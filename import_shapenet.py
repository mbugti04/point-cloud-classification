from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='/data/ShapeNet', categories=['Airplane'])

dataset[0]