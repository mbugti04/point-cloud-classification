import torch
from torch_geometric.data import Dataset
import process_3dm_file
from torch_geometric.datasets import GeometricShapes
import torch_geometric.transforms as T
from torch_geometric.transforms import KNNGraph


class NYCDataset(Dataset):
    def __init__(self, data_list):
        super(NYCDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]

        return data

if __name__ == "__main__":
    data_list = process_3dm_file.process_models()
    dataset = NYCDataset(data_list)

    data = dataset[0]

    print(data)

    dataset.transform = T.Compose([KNNGraph(k=6)])

    print(dataset[0])
