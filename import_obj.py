from torch_geometric.data import Data
from torch_geometric.datasets import PointCloudDataset

import torch_geometric.transforms as T

# Define the path to the .obj file
path = '/home/mbugti/point-cloud-classification/bunny.obj'

# Read the .obj file and sample points
dataset = PointCloudDataset(path, transform=T.SamplePoints(num=1024))

# Access the first data sample
data = dataset[0]

# Print the data information
print(data)