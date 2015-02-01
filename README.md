# pca_analysis: from multivariate dataset to PCA plot

### Description
* Calculate principal component analysis on multivariate data stored in a .csv
    file and display it in a 2D plot (optional 3D)

### Overview
* Takes a set a comma separated value (.csv) file as argument: this file should have a set of observations along with multiple variables on each of these observations
* Calculates principal component analysis (PCA): PCA is a dimentionality reduction algorithm with uses transformations of a dataset and projects it onto a lower set of variables called principal components (PCs). The PCs extract the important information from the data, revealing its internal structure in a way that best explains its variance.
* It plots the PCA data: choose 2D (default) or 3D projection and show (default) or save the PCA plot and legend.

### Citation
* v1.0 of this script was attributed a DOI:
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.14735.svg)](http://dx.doi.org/10.5281/zenodo.14735)

### Getting started
* Installation and prerequisites
    * The easiest way to get started is to download the Anaconda distribution, which includes Python along with the SciPy libraries (and other scientific Python libraries). Note that this is a >330 Mb file.
        * http://continuum.io/downloads
    * Then download the content of this repository and unpack on your computer. Or if you are already using git, use:
        
        ```bash
        git clone https://github.com/thomas-coudrat/pca_analysis.git
        ```

    * Example: navigate to the directory and run the following:
        
        ```bash
        python pca_analysis.py data/random_data.csv
        ```

    * Help: get more info about the optional arguments
        
        ```bash
        python pca_analysis.py --help
        ```

    * Data (.csv file): have a look at the file data/random_data.csv. The formatting of your data should follow the same rules: row and column names AND a column name 'color' with RGB colors as follows 'R:G:B'.

* Additional information
    * The script makes use of the following Scipy libraries: pandas, matplotlib, scikit-learn
    * The PCA is performed using scikit-learn which relies on singular value decomposition
