# ClusteringML
Clustering task for Machine Learning discipline in Universidade Federal de São Carlos (UFSCar).

# Data source
- **UrbanGB, urban road accidents coordinates labelled by the urban center Data Set** [https://archive.ics.uci.edu/ml/datasets/UrbanGB%2C+urban+road+accidents+coordinates+labelled+by+the+urban+center], which is made available here under the Open Database License (ODbL);
- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

## Dataset description

The data consists of 360177 rows and 2 columns. Each row represents an instance where the columns indicate the longitude and the latitude, respectively, of a road accident within Great Britain urban areas.

## Preprocessing

### Considerations

As noted in the README included in the dataset folder, the data is expressed in coordinate form. This means that one should take into account the curvature of the Earth in order to properly compute the distance between a pair of distinct points. Therefore, the dataset donors recommend scaling down longitude values by a factor of 1.7.

### Steps

The preprocessing stage was divided in 2 steps:
1- Scaling longitude values down by a factor of 1.7;
2- Normalizing each column.
