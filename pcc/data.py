"""Tile dataset for training and evaluation."""

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from . import config
from .features import FEATURE_GROUPS, columns, tile_features


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

    x and y are divided by half the tile size, so they fall in [-1, 1].
    Height is divided by `height_scale` meters (default: also half the tile
    size). With 250 m tiles that default puts a 10 m house at 0.08, so a
    smaller scale such as 30 m spreads typical building heights over a range
    the network can tell apart.

    Resampling hides how many vertices the tile had and the network has to
    learn height percentiles on its own, so the tile's log vertex count
    (`density`) and its height statistics (`tile_stats`, computed from all
    stored points) can be passed alongside as `extra` per-tile features.

    Training mode adds a random rotation about the vertical axis and small
    jitter. Eval mode resamples deterministically per tile.
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
        height_scale: float | None = None,
        density: bool = False,
        tile_stats: bool = False,
    ):
        super().__init__()
        self.store = store
        self.tiles = [(d, int(i)) for d, i in tiles]
        self.num_points = num_points
        self.train = train
        self.seed = seed
        self.rotate = rotate
        self.scale = np.array([store.tile_size / 2] * 2 + [height_scale or store.tile_size / 2], dtype=np.float32)
        self.labels = np.array([config.label_of(d) for d, _ in self.tiles])
        if shuffle_labels:
            self.labels = np.random.default_rng(seed).permutation(self.labels)
        self.extra = self._extra_features(density, tile_stats)

    def _extra_features(self, density: bool, tile_stats: bool) -> np.ndarray | None:
        names = (FEATURE_GROUPS["density"] if density else []) + (FEATURE_GROUPS["height"] if tile_stats else [])
        if not names:
            return None
        cols = columns(keep=names)
        rows = [tile_features(self.store.points(d, i), self.store.n_points(d, i))[cols] for d, i in self.tiles]
        return np.stack(rows).astype(np.float32)

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
        data = Data(pos=torch.from_numpy(np.ascontiguousarray(pos, dtype=np.float32)),
                    y=torch.tensor([self.labels[idx]]))
        if self.extra is not None:
            data.extra = torch.from_numpy(self.extra[idx:idx + 1])
        return data

    def class_counts(self, num_classes: int) -> np.ndarray:
        return np.bincount(self.labels, minlength=num_classes)
