## pca_analysis
### projection of a multivariate dataset onto its the top 2 or 3 calculated principal components

**Calculate PCA from data stored in a .csv file and display it in a 2D plot (optional 3D)**

*How to use:*
  * Prepare you data in a comma separated file (csv) the following format, note that the column *color* is required and must be written as described (not Color, colour, COLOR etc...)
  * An example file is inculded in data/random_data.csv
  * Run the example using:
`python pca_analysis data/random_data.csv -show`

*Requirements*:
  * scipy (pandas, numpy, matplotlib, scikit-learn)
