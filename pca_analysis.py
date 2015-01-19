#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import sys


def main():
    """
    Runs the pca_analysis script
    """
    csvPath, show_flag, save_flag = parsing()

    # Create the array of data
    array = np.loadtxt(stripCols(csvPath, 2), delimiter=",")

    # Get the ligand names for that data
    ligs = getColumn(csvPath, ",", 0)
    colors = getColumn(csvPath, ",", 1)

    # Output info collected
    displayInfo(array, ligs, colors)

    # Get the PCA of that data, with 2 dimensions
    pca = PCA(n_components=2)
    X_r = pca.fit(array).transform(array)

    # print pca.explained_variance_ratio_
    pc1 = str(int(pca.explained_variance_ratio_[0] * 100) / 100.0)
    pc2 = str(int(pca.explained_variance_ratio_[1] * 100) / 100.0)
    # print array
    # print ligs
    # print X_r

    plotPCA(X_r, pc1, pc2, ligs, colors, csvPath, show_flag, save_flag)


def parsing():
    """
    Parsing the parameters for script execution
    """

    descr = "Calculate PCA from data stored in a .csv file," \
        " and display it in a 2D plot"
    descr_csvPath = "Path of the .csv file, each line is a" \
        " instance and the first value is its name followed" \
        " by its variables"
    descr_show = "Use this flag if you want to show the figures upon execution"
    descr_save = "Use this flag if you want to save the figures upon execution"

    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument("csvPath", help=descr_csvPath)
    parser.add_argument("-show", action="store_true", help=descr_show)
    parser.add_argument("-save", action="store_true", help=descr_save)

    args = parser.parse_args()

    csvPath = args.csvPath
    show_flag = args.show
    save_flag = args.save

    if not show_flag and not save_flag:
        print "Select at least one of the flags -show or -save"
        sys.exit()

    return csvPath, show_flag, save_flag


def displayInfo(array, ligs, colors):
    """
    Output the information collected to the console
    """
    print "\n## DATA ##"
    print array
    print "\n## Names ##"
    for lig in ligs:
        print lig
    print "\n## Colors ##"
    for color in colors:
        print color
    print


def plotPCA(X_r, pc1, pc2, ligs, colors, csvPath, show_flag, save_flag):
    """
    Plot the PCA data on 2D plot
    """

    # Main figure
    fig = plt.figure(figsize=(13, 12), dpi=100)
    ax = fig.add_subplot(111)

    for label, col, x, y in zip(ligs, colors, X_r[:, 0], X_r[:, 1]):
        newCol = makeColor(col)
        ax.scatter(x, y, label=label, color=newCol, marker="o", lw=0, s=800)
        # ax.annotate(label, xy=(x, y - 0.05), fontsize=10,
        #             ha='center', va='top')

    ax.set_xlabel("PC1 (" + pc1 + ")", fontsize=30)
    ax.set_ylabel("PC2 (" + pc2 + ")", fontsize=30)
    ax.tick_params(axis="both", which="major", labelsize=30)
    figTitle = "PCA on " + csvPath + " (PC1=" + pc1 + ", PC2=" + pc2 + ")"
    ax.text(0.5, 1.04, figTitle, horizontalalignment="center", fontsize=30,
            transform=ax.transAxes)

    # Legend figure
    fig_legend = plt.figure()
    plt.figlegend(*ax.get_legend_handles_labels(), scatterpoints=1,
                  loc="center", fancybox=True,
                  shadow=True, prop={"size": 30})

    # Save figures
    if save_flag:
        print
        print "SAVING figures"
        print
        pngPath = csvPath.replace(".csv", ".png")
        fig.savefig(pngPath, bbox_inches="tight")
        fig_legend.savefig(pngPath.replace(".png", "_legend.png"))

    # Show figures
    if show_flag:
        print
        print "SHOWING figures"
        print
        plt.show()


def makeColor(colorRGB):
    """
    Get a RGB color (1-255) as input, in string format R:G:B
    And return a RGB color (0-1)
    """
    return [float(col)/255. for col in colorRGB.split(":")]


def getColumn(csvPath, delimiter, col):
    """
    Get only the ligand names from the .csv file
    """
    csvFile = open(csvPath, "r")
    csvLines = csvFile.readlines()
    csvFile.close()

    ligs = []

    for line in csvLines:
        ligand = line.split(delimiter)[col]
        ligs.append(ligand)

    return ligs


def stripCols(fname, col, delimiter=","):
    with open(fname, 'r') as fin:
        for line in fin:
            try:
                yield line.split(delimiter, 2)[col]
            except IndexError:
                continue

if __name__ == "__main__":
    main()
