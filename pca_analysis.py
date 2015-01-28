#!/usr/bin/env python

from __future__ import unicode_literals, print_function
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import sys
from mpl_toolkits.mplot3d import Axes3D
# from mpl_animations_from_static_3d import rotanimate


def main():
    """
    Runs the pca_analysis script
    """
    # Collect arguments
    csvPath, proj3D, show_flag, save_flag = parsing()

    # Prepare the data table
    df = pd.read_csv(filepath_or_buffer=csvPath, index_col=0, sep=",")
    ligs = df.index.values
    colors = df.color.values
    dfValues = df.drop("color", axis=1)
    dims = len(dfValues.columns)

    # Get the PCA for all dimentions (pc values),
    # and for the 2 dimentions (graph)
    X_r, pcValsAll = getPCA(dfValues, dims)
    if proj3D:
        X_r, pcVals = getPCA(dfValues, 3)
    else:
        X_r, pcVals = getPCA(dfValues, 2)

    # Displaying information to the terminal
    displayInfo(df, pcValsAll)

    # Run the plotting function
    plotPCA(proj3D, X_r, pcVals, ligs, colors, csvPath, show_flag, save_flag)


def parsing():
    """
    Parsing the parameters for script execution
    """

    descr = "Calculate PCA from data stored in a .csv file," \
        " and display it in a 2D plot (optional 3D)"
    descr_csvPath = "Path of the .csv file, each line is a" \
        " instance and the first value is its name followed" \
        " by its variables"
    descr_proj3D = "Use this flag for a 3D projection of the data"
    descr_show = "Use this flag if you want to show the figures upon execution"
    descr_save = "Use this flag if you want to save the figures upon execution"

    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument("csvPath", help=descr_csvPath)
    parser.add_argument("-proj3D", action="store_true", help=descr_proj3D)
    parser.add_argument("-show", action="store_true", help=descr_show)
    parser.add_argument("-save", action="store_true", help=descr_save)

    args = parser.parse_args()

    csvPath = args.csvPath
    proj3D = args.proj3D
    show_flag = args.show
    save_flag = args.save

    if not show_flag and not save_flag:
        print("Select at least one of the flags -show or -save")
        sys.exit()

    return csvPath, proj3D, show_flag, save_flag


def getPCA(data, dims):
    """
    Return the PCA data and principal component values given a dataset and a
    number of dimentions to be returned
    """

    # Get the PCA of that data
    pca = PCA(n_components=dims)
    X_r = pca.fit(data).transform(data)
    pcVals = [str(round(pc, 3)) for pc in pca.explained_variance_ratio_]

    return X_r, pcVals


def displayInfo(df, pcVals):
    """
    Print out the information to the terminal
    """

    print("\n## Data table ##\n")
    print(df)

    print("\n## Principal Components ##\n")
    for i, pc in enumerate(pcVals):
        print("PC" + str(i+1) + " = " + pc)


def plotPCA(proj3D, X_r, pcVals, ligs, colors, csvPath, show_flag, save_flag):
    """
    Plot the PCA data on 2D plot
    """

    # Main figure
    fig = plt.figure(figsize=(13, 12), dpi=100)

    if proj3D:
        ax = fig.add_subplot(111, projection="3d")
        for label, col, x, y, z in zip(ligs, colors,
                                       X_r[:, 0], X_r[:, 1], X_r[:, 2]):
            newCol = makeColor(col)
            Axes3D.scatter(ax, x, y, z, label=label, color=newCol,
                           marker="o", lw=1, s=800)
        ax.set_xlabel("PC1 (" + pcVals[0] + ")", fontsize=30)
        ax.set_ylabel("PC2 (" + pcVals[1] + ")", fontsize=30)
        ax.set_zlabel("PC3 (" + pcVals[2] + ")", fontsize=30)
        ax.tick_params(axis="both", which="major", labelsize=20)
    else:
        ax = fig.add_subplot(111)
        for label, col, x, y in zip(ligs, colors, X_r[:, 0], X_r[:, 1]):
            newCol = makeColor(col)
            ax.scatter(x, y, label=label, color=newCol, marker="o", lw=1, s=800)
            # ax.annotate(label, xy=(x, y - 0.05), fontsize=10,
            #             ha='center', va='top')
        ax.set_xlabel("PC1 (" + pcVals[0] + ")", fontsize=30)
        ax.set_ylabel("PC2 (" + pcVals[1] + ")", fontsize=30)
        ax.tick_params(axis="both", which="major", labelsize=30)

    # figTitle = "PCA on " + csvPath + " (PC1=" + pcVals[0] + ", PC2=" +
    # pcVals[1] + ")"
    # ax.text(0.5, 1.04, figTitle, horizontalalignment="center", fontsize=30,
    #         transform=ax.transAxes)

    # Legend figure
    fig_legend = plt.figure(figsize=(13, 12), dpi=100)
    plt.figlegend(*ax.get_legend_handles_labels(), scatterpoints=1,
                  loc="center", fancybox=True,
                  shadow=True, prop={"size": 30})

    # Save figures
    if save_flag:
        print("\nSAVING figures\n")
        pngPath = csvPath.replace(".csv", ".png")
        fig.savefig(pngPath, bbox_inches="tight")
        fig_legend.savefig(pngPath.replace(".png", "_legend.png"))

        # Take 20 angles between 0 & 360
        # angles = np.linspace(0, 360, 21)[:-1]
        # create an animated gif (20ms between frames)
        # rotanimate(ax, angles, pngPath.replace(".png", "_movie.gif", delay=20)

    # Show figures
    if show_flag:
        print("\nSHOWING figures\n")
        plt.show()


def makeColor(colorRGB):
    """
    Get a RGB color (1-255) as input, in string format R:G:B
    And return a RGB color (0-1)
    """
    return [float(col)/255. for col in colorRGB.split(":")]


if __name__ == "__main__":
    main()
