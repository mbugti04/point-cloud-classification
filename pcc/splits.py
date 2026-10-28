"""Train/val/test splits grouped by community district.

Every tile from a district lands in the same split, so the test set measures
how well the model handles neighborhoods it has never seen.

(Except for Staten Island where there is no download for the third district.)
"""

import json
from pathlib import Path

import numpy as np

from . import config

TEST_FRACTION = 0.25
VAL_FRACTION = 0.15
STRIP_FRACTION = 0.2


def assign_districts(districts: list[str], seed: int) -> dict[str, str]:
    """Map each district to 'train', 'val', 'test' or 'train+val_strip'."""
    rng = np.random.default_rng(seed)
    assignment = {}
    for borough in config.BOROUGHS:
        group = sorted(d for d in districts if config.borough_of(d) == borough)
        if not group:
            continue
        group = [group[i] for i in rng.permutation(len(group))]
        n = len(group)
        if n == 1:
            raise ValueError(f"{borough} has only one district; cannot hold one out for test")
        n_test = max(1, round(TEST_FRACTION * n))
        # A val district needs at least one train district left over.
        n_val = max(1, round(VAL_FRACTION * n)) if n - n_test >= 2 else 0
        for d in group[:n_test]:
            assignment[d] = "test"
        for d in group[n_test:n_test + n_val]:
            assignment[d] = "val"
        rest = group[n_test + n_val:]
        for d in rest:
            assignment[d] = "train"
        if n_val == 0:
            assignment[rest[0]] = "train+val_strip"
    return assignment


def strip_split(ij: np.ndarray, fraction: float = STRIP_FRACTION) -> np.ndarray:
    """Label tiles 'val', 'buffer' or 'train' by cutting a strip off the long axis."""
    axis = int(np.ptp(ij[:, 0]) < np.ptp(ij[:, 1]))
    coord = ij[:, axis]
    lo, hi = coord.min(), coord.max()
    cut = hi - max(1, int(round(fraction * (hi - lo + 1)))) + 1
    labels = np.full(len(ij), "train", dtype=object)
    labels[coord >= cut] = "val"
    labels[coord == cut - 1] = "buffer"
    return labels


def make_splits(tiles_dir: Path, seed: int) -> dict:
    """Build the split file contents: per-district assignment and tile indices."""
    districts = sorted(p.stem for p in tiles_dir.glob("*.npz"))
    assignment = assign_districts(districts, seed)
    tiles = {"train": [], "val": [], "test": []}
    for district in districts:
        role = assignment[district]
        ij = np.load(tiles_dir / f"{district}.npz")["ij"]
        if role == "train+val_strip":
            labels = strip_split(ij)
            for split in ("train", "val"):
                tiles[split] += [[district, int(i)] for i in np.flatnonzero(labels == split)]
        else:
            tiles[role] += [[district, i] for i in range(len(ij))]
    return {"seed": seed, "districts": assignment, "tiles": tiles}


def save_splits(splits: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=1))


def load_splits(path: Path) -> dict:
    return json.loads(path.read_text())
