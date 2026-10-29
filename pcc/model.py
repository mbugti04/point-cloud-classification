"""Point Transformer classifier.

Adapted from PyTorch Geometric's point_transformer_classification example
(Zhao et al., "Point Transformer", 2021).
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MLP, PointTransformerConv, fps, global_mean_pool, knn, knn_graph
from torch_geometric.utils import scatter


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_in = Linear(in_channels, in_channels)
        self.lin_out = Linear(out_channels, out_channels)
        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)
        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, plain_last=False)
        self.transformer = PointTransformerConv(
            in_channels, out_channels, pos_nn=self.pos_nn, attn_nn=self.attn_nn
        )

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        return self.lin_out(x).relu()


class TransitionDown(torch.nn.Module):
    """Farthest-point-sample `ratio` of the points, then max-pool features from each sample's k neighbors."""

    def __init__(self, in_channels: int, out_channels: int, ratio: float = 0.25, k: int = 16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        sub_batch = batch[id_clusters] if batch is not None else None
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)
        x = self.mlp(x)
        x_out = scatter(
            x[id_k_neighbor[1]], id_k_neighbor[0], dim=0, dim_size=id_clusters.size(0), reduce="max"
        )
        return x_out, pos[id_clusters], sub_batch


class PointTransformerClassifier(torch.nn.Module):
    """Point Transformer that takes each point's local xyz as its input feature.

    The original notebook fed constant features, so the network saw geometry
    only through relative offsets between neighbors and could not tell how
    tall a building is. Tiles are now in local coordinates (centered, height
    above ground), so passing xyz in directly no longer exposes city location.

    With `use_density`, the tile's log vertex count is embedded and joined to
    the pooled features before the classifier head. Every tile is resampled
    to the same number of points, so without it the network cannot tell a
    sparse block of houses from a dense block of towers.
    """

    def __init__(self, num_classes: int, dims: tuple[int, ...] = (32, 64, 128, 256, 512), k: int = 16,
                 use_density: bool = True, density_dim: int = 32):
        super().__init__()
        self.k = k
        self.use_density = use_density
        self.mlp_input = MLP([3, dims[0]], plain_last=False)
        self.transformer_input = TransformerBlock(dims[0], dims[0])
        self.transition_down = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            self.transition_down.append(TransitionDown(d_in, d_out, k=k))
            self.transformers_down.append(TransformerBlock(d_out, d_out))
        head_in = dims[-1]
        if use_density:
            # Set from the training tiles with `set_density_stats`; saved in checkpoints.
            self.register_buffer("density_mean", torch.tensor(0.0))
            self.register_buffer("density_std", torch.tensor(1.0))
            self.density_mlp = MLP([1, density_dim], norm=None, plain_last=False)
            head_in += density_dim
        self.mlp_output = MLP([head_in, 64, num_classes], norm=None)

    def set_density_stats(self, mean: float, std: float) -> None:
        self.density_mean.fill_(mean)
        self.density_std.fill_(std)

    def forward(self, pos, batch, density=None):
        x = self.mlp_input(pos)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)
        for down, block in zip(self.transition_down, self.transformers_down):
            x, pos, batch = down(x, pos, batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = block(x, pos, edge_index)
        x = global_mean_pool(x, batch)
        if self.use_density:
            density = (density - self.density_mean) / self.density_std
            x = torch.cat([x, self.density_mlp(density)], dim=1)
        return F.log_softmax(self.mlp_output(x), dim=-1)
