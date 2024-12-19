from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import process_3dm_file

data_list = process_3dm_file.process_all_models()
print(len(data_list))
print(data_list)
loader = DataLoader(data_list, batch_size=32)