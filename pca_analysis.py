#!/usr/bin/env python

from __future__ import unicode_literals, print_function
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
from mpl_toolkits.mplot3d import Axes3D


def main():
    """
    Runs the pca_analysis script
    """
    # Collect arguments
    csvPath, rounded, proj3D, save_flag = parsing()

    # Prepare the data table
    df = pd.read_csv(filepath_or_buffer=csvPath, index_col=0, sep=",")
    ligs = df.index.values
    colors = df.color.values
    dfValues = df.drop("color", axis=1)
    dims = len(dfValues.columns)

    # Get the PCA for all dimentions (pc values),
    # and for the 2 dimentions (graph)
    X_r, pcValsAll = getPCA(dfValues, dims, 6)
    if proj3D:
        X_r, pcVals = getPCA(dfValues, 3, rounded)
    else:
        X_r, pcVals = getPCA(dfValues, 2, rounded)

    # Displaying information to the terminal
    displayInfo(df, pcValsAll)

    # Run the plotting function
    plotPCA(proj3D, X_r, pcVals, ligs, colors, csvPath, save_flag)


def parsing():
    """
    Parsing the parameters for script execution
    """

    descr = "Calculate PCA from data stored in a .csv file," \
        " and display it in a 2D plot (optional 3D)"
    descr_csvPath = "Path of the .csv file, each line is a" \
        " instance and the first value is its name followed" \
        " by its variables"
    descr_rounded = "Decimal at which principal components are rounded" \
        " (default=2)"
    descr_proj3D = "Use this flag for a 3D projection of the data"
    descr_save = "Use this flag if you want to save the figures upon execution"

    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument("csvPath", help=descr_csvPath)
    parser.add_argument("--rounded", type=int, help=descr_rounded)
    parser.add_argument("-proj3D", action="store_true", help=descr_proj3D)
    parser.add_argument("-save", action="store_true", help=descr_save)

    args = parser.parse_args()

    csvPath = args.csvPath
    # Default to rounded at second decimal if nothing else is provided
    if args.rounded:
        rounded = args.rounded
    else:
        rounded = 0
    proj3D = args.proj3D
    save_flag = args.save

    return csvPath, rounded, proj3D, save_flag


def getPCA(data, dims, rounded):
    """
    Return the PCA data and principal component values given a dataset and a
    number of dimentions to be returned
    """

    # Get the PCA of that data
    pca = PCA(n_components=dims)
    X_r = pca.fit(data).transform(data)
    pcVals = [round(100 * pc, rounded) for pc in pca.explained_variance_ratio_]

    return X_r, pcVals


def displayInfo(df, pcVals):
    """
    Print out the information to the terminal
    """

    print("\n## Data table ##\n")
    print(df)

    print("\n## Principal Components ##\n")
    for i, pc in enumerate(pcVals):
        print("PC" + str(i+1) + " = " + str(pc) + " %")


def plotPCA(proj3D, X_r, pcVals, ligs, colors, csvPath, save_flag):
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
        ax.set_xlabel("PC1 (" + '{0:g}'.format(pcVals[0]) + " %)", fontsize=30)
        ax.set_ylabel("PC2 (" + '{0:g}'.format(pcVals[1]) + " %)", fontsize=30)
        ax.set_zlabel("PC3 (" + '{0:g}'.format(pcVals[2]) + " %)", fontsize=30)
        ax.tick_params(axis="both", which="major", labelsize=20)
    else:
        ax = fig.add_subplot(111)
        for label, col, x, y in zip(ligs, colors, X_r[:, 0], X_r[:, 1]):
            newCol = makeColor(col)
            ax.scatter(x, y, label=label, color=newCol, marker="o", lw=1, s=800)
            # ax.annotate(label, xy=(x, y - 0.05), fontsize=10,
            #             ha='center', va='top')
        ax.set_xlabel("PC1 (" + '{0:g}'.format(pcVals[0]) + " %)", fontsize=30)
        ax.set_ylabel("PC2 (" + '{0:g}'.format(pcVals[1]) + " %)", fontsize=30)
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

    # Save figures if save flag was used
    if save_flag:
        print("\nSAVING figures\n")
        pngPath = csvPath.replace(".csv", ".png")
        fig.savefig(pngPath, bbox_inches="tight")
        fig_legend.savefig(pngPath.replace(".png", "_legend.png"))
    # Otherwise show the plots
    else:
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
