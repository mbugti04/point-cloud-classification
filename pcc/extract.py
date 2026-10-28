"""Extract building vertices from NYC .3dm files.

The NYC model stores each building as three kinds of Brep surfaces (facades,
rooftops, footprints), in millimeters on the NY State Plane. We keep every
Brep vertex, convert to meters, and drop exact duplicates: neighboring facade
quads share corners with each other and with the roof and footprint, so about
two thirds of the raw vertices are repeats.

Footprint vertices are also saved on their own as a ground reference, so later
steps can measure height above local ground instead of absolute elevation.
"""

from pathlib import Path

import numpy as np

FOOTPRINT_LAYER = "footprint surface"

_UNIT_TO_METERS = {
    "Millimeters": 1e-3,
    "Centimeters": 1e-2,
    "Meters": 1.0,
    "Feet": 0.3048,
    "Inches": 0.0254,
}


def _is_footprint(layer_path: str) -> bool:
    return layer_path.split("::")[-1].strip().lower() == FOOTPRINT_LAYER


def extract_vertices(path_3dm: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (all vertices, footprint vertices), deduplicated, in meters as float64 [N, 3]."""
    import rhino3dm

    model = rhino3dm.File3dm.Read(str(path_3dm))
    if model is None:
        raise ValueError(f"rhino3dm could not read {path_3dm}")

    unit = str(model.Settings.ModelUnitSystem).split(".")[-1]
    if unit not in _UNIT_TO_METERS:
        raise ValueError(f"Unsupported unit system {unit!r} in {path_3dm}")
    scale = _UNIT_TO_METERS[unit]

    footprint_layers = {i for i, layer in enumerate(model.Layers) if _is_footprint(layer.FullPath)}

    coords, ground = [], []
    for obj in model.Objects:
        geometry = obj.Geometry
        if not isinstance(geometry, rhino3dm.Brep):
            continue
        is_footprint = obj.Attributes.LayerIndex in footprint_layers
        for vertex in geometry.Vertices:
            loc = vertex.Location
            coords.append((loc.X, loc.Y, loc.Z))
            if is_footprint:
                ground.append((loc.X, loc.Y, loc.Z))

    xyz = np.asarray(coords, dtype=np.float64).reshape(-1, 3) * scale
    ground_xyz = np.asarray(ground, dtype=np.float64).reshape(-1, 3) * scale
    return dedupe(xyz), dedupe(ground_xyz)


def dedupe(xyz: np.ndarray, resolution: float = 0.01) -> np.ndarray:
    """Drop points that coincide within `resolution` meters, keeping a stable order."""
    if len(xyz) == 0:
        return xyz
    keys = np.round(xyz / resolution).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(first)]


def extract_to_file(path_3dm: Path, out_path: Path) -> int:
    """Extract a .3dm file to a compressed .npz. Returns the point count."""
    xyz, ground_xyz = extract_vertices(path_3dm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, xyz=xyz, ground=ground_xyz)
    tmp.replace(out_path)
    return len(xyz)
