"""Cut each district's vertices into square tiles on a citywide grid.

Each tile becomes one training sample. Coordinates are made local so the model
cannot read a tile's position in the city:

* x and y are measured from the tile center, and
* z becomes height above the nearest building footprint, which removes
  terrain elevation (e.g. Staten Island's hills) as a shortcut.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def height_above_ground(xyz: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """z minus the z of the nearest footprint vertex in the xy plane."""
    if len(ground) == 0:
        return xyz[:, 2] - xyz[:, 2].min()
    _, idx = cKDTree(ground[:, :2]).query(xyz[:, :2])
    return xyz[:, 2] - ground[idx, 2]


def tile_district(
    xyz: np.ndarray,
    ground: np.ndarray,
    tile_size: float,
    min_points: int,
    max_points: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Split one district into tiles.

    Returns arrays describing every kept tile:
      points   float32 [sum(n_i), 3]  local x, y and height, concatenated
      offsets  int64   [T + 1]        tile i is points[offsets[i]:offsets[i + 1]]
      ij       int32   [T, 2]         grid cell of each tile
      n_points int64   [T]            vertex count before any subsampling
    Tiles with fewer than `min_points` vertices are dropped; tiles above
    `max_points` are randomly subsampled to bound file size.
    """
    height = height_above_ground(xyz, ground)
    cell = np.floor(xyz[:, :2] / tile_size).astype(np.int64)

    order = np.lexsort((cell[:, 1], cell[:, 0]))
    cell, xyz, height = cell[order], xyz[order], height[order]
    starts = np.flatnonzero(np.r_[True, np.any(cell[1:] != cell[:-1], axis=1)])
    ends = np.r_[starts[1:], len(cell)]

    chunks, ij, counts = [], [], []
    for s, e in zip(starts, ends):
        n = e - s
        if n < min_points:
            continue
        center = (cell[s] + 0.5) * tile_size
        local = np.column_stack([xyz[s:e, 0] - center[0], xyz[s:e, 1] - center[1], height[s:e]])
        if n > max_points:
            local = local[np.sort(rng.choice(n, max_points, replace=False))]
        chunks.append(local.astype(np.float32))
        ij.append(cell[s])
        counts.append(n)

    offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(c) for c in chunks])
    return {
        "points": np.concatenate(chunks) if chunks else np.zeros((0, 3), np.float32),
        "offsets": offsets,
        "ij": np.asarray(ij, dtype=np.int32).reshape(-1, 2),
        "n_points": np.asarray(counts, dtype=np.int64),
    }


def tile_all(
    interim_dir: Path,
    tiles_dir: Path,
    tile_size: float,
    min_points: int,
    max_points: int,
    seed: int = 0,
) -> dict[str, int]:
    """Tile every extracted district. Returns {district: tile count}."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for path in sorted(interim_dir.glob("*.npz")):
        district = path.stem
        if district.endswith(".tmp"):  # left behind by an interrupted extraction
            continue
        data = np.load(path)
        # Seed per district so re-tiling one district does not change the others.
        rng = np.random.default_rng([seed, int.from_bytes(district.encode(), "little")])
        tiles = tile_district(data["xyz"], data["ground"], tile_size, min_points, max_points, rng)
        np.savez_compressed(
            tiles_dir / f"{district}.npz",
            tile_size=np.float64(tile_size),
            **tiles,
        )
        counts[district] = len(tiles["ij"])
    return counts
