"""Per-building features, for a transformer whose tokens are buildings.

The NYC model draws each building's outline as a closed polyline on the
`Building_FootPrint` layer, and each roof level as one on `Building_RoofTop`.
Each roof is matched to the footprint containing its center. A building's
height is its highest roof above its own footprint; buildings with no matched
roof (well under 1%) are dropped.

Building tiles reuse the point tiles' grid and order exactly, so a tile index
in `splits.json` means the same 250 m square for both models. A building
belongs to the tile containing its footprint centroid.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .extract import _UNIT_TO_METERS
from .tiles import cell_center, cell_of

FOOTPRINT_LAYER = "building_footprint"
ROOF_LAYER = "building_rooftop"

# Per-token features after the local position (x, y, height). None of them
# depend on where the tile is in the city or which way it faces.
TOKEN_FEATURES = ["log_area", "compactness", "log_n_vertices", "n_roofs", "roof_range", "log_nn_distance"]


def polygon_area_perimeter(xy: np.ndarray) -> tuple[float, float]:
    """Area and perimeter of a closed polygon given as [N, 2] (first point may repeat at the end)."""
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perimeter = np.linalg.norm(xy - np.roll(xy, -1, axis=0), axis=1).sum()
    return float(area), float(perimeter)


def _polylines(model, layer_name: str, scale: float) -> list[np.ndarray]:
    import rhino3dm

    layers = {i for i, layer in enumerate(model.Layers) if layer.FullPath.split("::")[-1].strip().lower() == layer_name}
    out = []
    for obj in model.Objects:
        geometry = obj.Geometry
        if obj.Attributes.LayerIndex in layers and isinstance(geometry, rhino3dm.PolylineCurve):
            pts = [geometry.Point(i) for i in range(geometry.PointCount)]
            out.append(np.array([(p.X, p.Y, p.Z) for p in pts]) * scale)
    return out


def extract_buildings(path_3dm: Path) -> dict[str, np.ndarray]:
    """One row per building, in meters, absolute coordinates (local ones come at tiling)."""
    import rhino3dm
    from matplotlib.path import Path as Polygon

    model = rhino3dm.File3dm.Read(str(path_3dm))
    if model is None:
        raise ValueError(f"rhino3dm could not read {path_3dm}")
    unit = str(model.Settings.ModelUnitSystem).split(".")[-1]
    if unit not in _UNIT_TO_METERS:
        raise ValueError(f"Unsupported unit system {unit!r} in {path_3dm}")
    scale = _UNIT_TO_METERS[unit]

    footprints = [p for p in _polylines(model, FOOTPRINT_LAYER, scale) if len(p) >= 3]
    roofs = _polylines(model, ROOF_LAYER, scale)

    centroids = np.array([p[:, :2].mean(axis=0) for p in footprints]).reshape(-1, 2)
    outlines = [Polygon(p[:, :2]) for p in footprints]
    tree = cKDTree(centroids)
    roof_z = [[] for _ in footprints]
    for roof in roofs:
        center = roof[:, :2].mean(axis=0)
        _, candidates = tree.query(center, k=min(8, len(footprints)))
        for i in np.atleast_1d(candidates):
            if outlines[i].contains_point(center):
                roof_z[i].append(roof[:, 2].max())
                break

    rows = []
    for footprint, centroid, zs in zip(footprints, centroids, roof_z):
        if not zs:
            continue
        base = footprint[:, 2].min()
        closed = np.allclose(footprint[0], footprint[-1])
        area, perimeter = polygon_area_perimeter(footprint[:-1, :2] if closed else footprint[:, :2])
        rows.append((*centroid, max(zs) - base, area, perimeter,
                     len(footprint) - closed, len(zs), max(zs) - min(zs)))
    rows = np.array(rows, dtype=np.float64).reshape(-1, 8)
    return {
        "centroid": rows[:, 0:2],
        "height": rows[:, 2],
        "area": rows[:, 3],
        "perimeter": rows[:, 4],
        "n_vertices": rows[:, 5],
        "n_roofs": rows[:, 6],
        "roof_range": rows[:, 7],
    }


def extract_buildings_to_file(path_3dm: Path, out_path: Path) -> int:
    """Extract one .3dm to an .npz of building rows. Returns the building count."""
    buildings = extract_buildings(path_3dm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **buildings)
    tmp.replace(out_path)
    return len(buildings["height"])


def token_features(buildings: dict[str, np.ndarray]) -> np.ndarray:
    """[N, len(TOKEN_FEATURES)] for every building of a district."""
    area = np.maximum(buildings["area"], 1e-3)
    compactness = 4 * np.pi * area / np.maximum(buildings["perimeter"], 1e-3) ** 2
    # Nearest other building over the whole district, so tile edges don't bias it.
    centroid = buildings["centroid"]
    if len(centroid) > 1:
        nn = cKDTree(centroid).query(centroid, k=2)[0][:, 1]
    else:
        nn = np.full(len(centroid), 100.0)
    return np.column_stack([
        np.log(area), compactness, np.log(buildings["n_vertices"]), buildings["n_roofs"],
        buildings["roof_range"], np.log1p(nn),
    ])


def tile_buildings(buildings: dict[str, np.ndarray], ij: np.ndarray, tile_size: float) -> dict[str, np.ndarray]:
    """Group a district's buildings into the given point tiles, in the same order.

    Returns
      tokens  float32 [sum(n_i), 3 + len(TOKEN_FEATURES)]  local x, y, height, then TOKEN_FEATURES
      offsets int64   [T + 1]                             tile t is tokens[offsets[t]:offsets[t + 1]]
    A tile can have no building whose centroid lies inside it; it stays, empty.
    """
    index = {tuple(c): t for t, c in enumerate(np.asarray(ij, dtype=np.int64))}
    cells = cell_of(buildings["centroid"], tile_size)
    tile_of = np.array([index.get(tuple(c), -1) for c in cells], dtype=np.int64)
    keep = tile_of >= 0
    order = np.argsort(tile_of[keep], kind="stable")
    rows = np.flatnonzero(keep)[order]
    tile_of = tile_of[rows]

    local = buildings["centroid"][rows] - cell_center(ij[tile_of], tile_size)
    tokens = np.column_stack([local, buildings["height"][rows], token_features(buildings)[rows]])
    offsets = np.zeros(len(ij) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.bincount(tile_of, minlength=len(ij)))
    return {"tokens": tokens.astype(np.float32), "offsets": offsets}


def tile_buildings_all(buildings_dir: Path, tiles_dir: Path, out_dir: Path) -> dict[str, tuple[int, int]]:
    """Building tiles for every district that has point tiles. Returns {district: (tiles, empty tiles)}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for tiles_path in sorted(tiles_dir.glob("*.npz")):
        district = tiles_path.stem
        buildings_path = buildings_dir / f"{district}.npz"
        if not buildings_path.exists():
            print(f"! {district}: no extracted buildings, skipping")
            continue
        point_tiles = np.load(tiles_path)
        tile_size = float(point_tiles["tile_size"])
        ij = point_tiles["ij"]
        tiles = tile_buildings(dict(np.load(buildings_path)), ij, tile_size)
        np.savez_compressed(out_dir / f"{district}.npz", tile_size=np.float64(tile_size), ij=ij, **tiles)
        counts[district] = (len(ij), int((np.diff(tiles["offsets"]) == 0).sum()))
    return counts


class BuildingStore:
    """Building tiles of the listed districts, loaded into memory once."""

    def __init__(self, building_tiles_dir: Path, districts: set[str]):
        self.tiles = {}
        self.tile_size = None
        for district in sorted(districts):
            data = np.load(building_tiles_dir / f"{district}.npz")
            self.tiles[district] = {k: data[k] for k in ("tokens", "offsets", "ij")}
            self.tile_size = float(data["tile_size"])

    def tokens(self, district: str, index: int) -> np.ndarray:
        t = self.tiles[district]
        return t["tokens"][t["offsets"][index]:t["offsets"][index + 1]]
