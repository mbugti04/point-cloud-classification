# point-cloud-classification


# Info

This project aims to classify New York City Boroughs with point cloud data using a Graph Neural Network. The model is trained on segments of the five boroughs. The idea is that the network picks up on each borough's distinct features (building size, density, distribution). The PointNet++ Graph Convolutional Neural Network obtains a testing accuracy of <>%, while the Point Cloud Transformer achieves a testing accuracy of <>%



# Installation

## Requirements
A couple of GBs of free space. (It initally required around 50GB of free space in order to download all of the dataset. However, I now download and process one file at a time, thus greatly reducing the storage need)

1. Download `pytorch`
2. Download `pytorch-geometric` with additional dependencies


# How to Run
Run `download_data.py`, which installs, unzips, and formats the dataset provided from the official [NYC 3D model dataset](https://www.nyc.gov/site/planning/data-maps/open-data/dwn-nyc-3d-model-download.page)
Run `

# Methodology

I first found models of cities online. After downloading them, I wrote the function to import the files using `trimesh`.

Then, I split the models of the cities into different chunks. Then, I sample points on the surface.

Then, I created a dataset for these city models using `pytorch-geometric`. First, these point clouds are converted into graphs (with nodes and edges).

Some point clouds are used as training data while others is used as test data. I train a graph neural network using `pytorch-geometric`

# Methodology 2

I found the official [NYC 3D model dataset](https://www.nyc.gov/site/planning/data-maps/open-data/dwn-nyc-3d-model-download.page) divided into 59 Community Districts.
These were in a proprietary `.3dm` file format. However, I found a python library `rhino3dm` which allowed me to process the file format and obtain the vertices. The vertices obtained are plotted below.

## 3D View
![alt text](images/nyc_3d_plot1.png)
Point clound of Manhattan


## Top-down View
![alt text](images/nyc_3d_plot2.png)
Top-down view of Manhattan

# References
