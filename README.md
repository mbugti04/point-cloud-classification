# point-cloud-classification

# Installation

Download `pytorch`
Download `pytorch-geometric` with additional dependencies

# Info

This project aims to classify locations with point cloud data. It is trained on segments of various cities.

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


## Top-down View
![alt text](images/nyc_3d_plot2.png)
