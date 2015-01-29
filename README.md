## pca_analysis: projection of a multivariate dataset onto the top 2 or 3 principal components of dataset

Calculate PCA from data stored in a .csv file and display it in a 2D plot (optional 3D)

*How to use:*
  * Prepare you data in a comma separated file (csv) the following format, note that the column *color* is required and must be written as described (not Color, colour, COLOR etc...)
  * An example file is inculded, run using:
    * `python pca_analysis data/random_data.csv -show`

                 color  valueA  valueB  valueC  valueD
observation                                           
one              0:0:0     4.5       5      10      13
two          255:0:255     5.0      10      15      11
three        100:100:0     2.0      11      14       2
four          150:0:50     1.0       1      10      13
five           200:0:0     4.0      10       5      14
six          10:10:255     4.0       4      10      10


*Requirements*:
  * scipy (pandas, numpy, matplotlib, scikit-learn)
