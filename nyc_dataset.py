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
        # Implement your data processing logic here
        # You can access the data from self.data_list[idx]
        # and convert it into a PyTorch Geometric Data object

        # Example:
        data = self.data_list[idx]
        # Process the data and convert it into a PyTorch Geometric Data object
        # ...

        return data

if __name__ == "__main__":
    # Assuming you have a data_list containing your data
    data_list = process_3dm_file.process_all_models()

    # Create an instance of your dataset
    dataset = NYCDataset(data_list)

    # Access the data using dataset[idx]
    # Example:
    data = dataset[0]

    print(data)

    dataset.transform = T.Compose([KNNGraph(k=6)])

    print(dataset[0])
