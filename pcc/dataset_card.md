---
license: other
license_name: nyc-open-data
license_link: https://www.nyc.gov/home/terms-of-use.page
task_categories:
  - other
tags:
  - point-cloud
  - 3d
  - geospatial
  - new-york-city
pretty_name: NYC 3D Building Tiles
size_categories:
  - 1K<n<10K
---

# NYC 3D Building Tiles

Point clouds of New York City buildings, cut into {{tile_size}} m square tiles and
labeled with their borough. Built from the
[NYC 3D Model by Community District](https://www.nyc.gov/content/planning/pages/resources/datasets/nyc-3d-model)
(NYC Department of City Planning, based on 2014 aerial survey data). The
original files use Rhino's `.3dm` format; this dataset converts them into plain
NumPy arrays.

Code: https://github.com/mbugti04/point-cloud-classification

## Files

- `tiles/<DISTRICT>.npz`: one file per community district (e.g. `MN01`), with:
  - `points` float32 `[N, 3]`: every tile's points, concatenated
  - `offsets` int64 `[T + 1]`: tile `i` is `points[offsets[i]:offsets[i+1]]`
  - `ij` int32 `[T, 2]`: grid cell of each tile (NY State Plane, meters / tile size)
  - `n_points` int64 `[T]`: vertex count before subsampling (tiles above 8192 points are subsampled)
  - `tile_size`: tile edge length in meters
- `splits.json`: train/val/test tile lists and the district assignment.

## Coordinates

Each point is a deduplicated building vertex (facade, roof and footprint
corners). Coordinates are local to the tile:

- `x`, `y`: meters from the tile center, in `[-tile_size/2, tile_size/2]`
- `z`: meters above the nearest building footprint, so terrain elevation is removed

No absolute position is stored per point; `ij` keeps tile positions available for
mapping and visualization.

## Labels

`MN` Manhattan, `BX` Bronx, `BK` Brooklyn, `QN` Queens, `SI` Staten Island,
taken from the first two letters of the district code.

## Splits

Splits are grouped by community district (seed {{split_seed}}): every tile from a
district is in the same split, so test districts are neighborhoods the model
never saw during training.

- Val districts: {{val_districts}}
- Test districts: {{test_districts}}

The city's download for Staten Island district SI03 is currently a dead link,
leaving two SI districts. One is in test. The other is in train, except for a
strip of tiles at one end held out as SI's val data, separated from the train
tiles by a dropped one-tile buffer.

Tiles per split:

{{split_table}}

## Limitations

- The source model reflects buildings in 2014.
- Park, airport and other non-district files are excluded.
- Tiles on district edges are partial, since each district is tiled separately.
