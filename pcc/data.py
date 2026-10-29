"""Tile dataset for training and evaluation."""

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from . import config


class TileStore:
    """All tiles of the listed districts, loaded into memory once."""

    def __init__(self, tiles_dir: Path, districts: set[str]):
        self.tiles = {}
        self.tile_size = None
        for district in sorted(districts):
            data = np.load(tiles_dir / f"{district}.npz")
            self.tiles[district] = {k: data[k] for k in ("points", "offsets", "n_points", "ij")}
            tile_size = float(data["tile_size"])
            if self.tile_size is not None and tile_size != self.tile_size:
                raise ValueError("Tiles were cut at different sizes; re-run `pcc tile`.")
            self.tile_size = tile_size

    def points(self, district: str, index: int) -> np.ndarray:
        t = self.tiles[district]
        return t["points"][t["offsets"][index]:t["offsets"][index + 1]]

    def n_points(self, district: str, index: int) -> int:
        return int(self.tiles[district]["n_points"][index])


class TileDataset(Dataset):
    """Each item is one tile resampled to `num_points` points.

    Coordinates are divided by half the tile size, so x and y fall in [-1, 1]
    and height keeps its real scale relative to the tile (taller buildings
    stay taller). Resampling hides how many vertices the tile had, so the
    original count is passed alongside as `density` (log scale). Training
    mode adds a random rotation about the vertical axis and small jitter. Eval mode resamples deterministically per tile.
    """

    def __init__(
        self,
        store: TileStore,
        tiles: list[tuple[str, int]],
        num_points: int,
        train: bool,
        seed: int = 0,
        rotate: bool = True,
        shuffle_labels: bool = False,
    ):
        super().__init__()
        self.store = store
        self.tiles = [(d, int(i)) for d, i in tiles]
        self.num_points = num_points
        self.train = train
        self.seed = seed
        self.rotate = rotate
        self.scale = store.tile_size / 2
        self.labels = np.array([config.label_of(d) for d, _ in self.tiles])
        if shuffle_labels:
            self.labels = np.random.default_rng(seed).permutation(self.labels)

    def len(self) -> int:
        return len(self.tiles)

    def get(self, idx: int) -> Data:
        district, index = self.tiles[idx]
        points = self.store.points(district, index)
        if self.train:
            # Drawn from torch's RNG so `torch.manual_seed` makes runs repeatable.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))
        else:
            rng = np.random.default_rng([self.seed, idx])
        choice = rng.choice(len(points), self.num_points, replace=len(points) < self.num_points)
        pos = points[choice] / self.scale
        if self.train:
            if self.rotate:
                theta = rng.uniform(0, 2 * np.pi)
                c, s = np.cos(theta), np.sin(theta)
                pos[:, :2] = pos[:, :2] @ np.array([[c, -s], [s, c]], dtype=np.float32)
            pos = pos + rng.normal(0, 0.005, pos.shape).astype(np.float32)
        density = np.log(self.store.n_points(district, index))
        return Data(pos=torch.from_numpy(np.ascontiguousarray(pos, dtype=np.float32)),
                    density=torch.tensor([[density]], dtype=torch.float),
                    y=torch.tensor([self.labels[idx]]))

    def class_counts(self, num_classes: int) -> np.ndarray:
        return np.bincount(self.labels, minlength=num_classes)
