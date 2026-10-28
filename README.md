# Point Cloud Classification

![](images/niv-rozenberg-ySNtJhDX-cw-unsplash1.jpg)
Photo by <a href="https://unsplash.com/@nivroz?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Niv Rozenberg</a> on <a href="https://unsplash.com/photos/aerial-view-of-city-during-day-time-ySNtJhDX-cw?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>

Can a neural network tell which New York City borough a neighborhood is in,
just from the 3D shape of its buildings? This project converts the city's
official 3D building model into labeled point clouds and trains a Point
Transformer to classify them by borough.

## Results

_Pending a full run on all districts. Report mean ± std across seeds from
`runs/pct/summary.md` and `runs/baseline/summary.md`._

| model | accuracy | balanced accuracy | macro-F1 |
|---|---|---|---|
| majority class | | | |
| gradient boosting on tile statistics | | | |
| Point Transformer | | | |

## Data

The [NYC 3D Model by Community District](https://www.nyc.gov/content/planning/pages/resources/datasets/nyc-3d-model)
covers every building in the city as of 2014, split into the 59 community
districts and stored in Rhino's `.3dm` format. The pipeline:

1. **Download and extract** (`pcc download`). Reads each `.3dm` with
   [`rhino3dm`](https://github.com/mcneel/rhino3dm), keeps building vertices
   (facade, roof and footprint corners), converts millimeters to meters, and
   drops duplicate vertices (about two thirds of the raw count). Districts are
   handled one at a time and the raw files deleted, so the ~8.6 GB of zips are
   never on disk together. Each file is parsed in a fresh subprocess to work
   around memory that `rhino3dm` never frees.
2. **Tile** (`pcc tile`). Cuts each district into 250 m tiles on a citywide grid.
   Each tile is one sample. Coordinates are made local: x and y relative to the
   tile center, z as height above the nearest building footprint. The model
   can't use where a tile is in the city or how high the ground is.
3. **Split** (`pcc split`). Groups by community district, so test
   neighborhoods are never seen in training. Per borough, about 25% of
   districts go to test and 15% to validation.

It should be noted that the city's download link for the third Staten Island district, SI03, is dead, leaving only two districts.

The processed tiles can be published to the Hugging Face Hub with `pcc publish`.

## Evaluation

- **Checkpoint selection on validation only.** Each seed keeps the epoch with
  the best validation macro-F1. The test set is scored once, with that
  checkpoint.
- **Multiple seeds.** Results are mean ± std over seeds (default 0, 1, 2).
- **Class balance.** Boroughs have very different tile counts, so the loss is
  weighted by inverse class frequency, and balanced accuracy and macro-F1 are
  reported alongside accuracy.
- **Baselines.** A majority-class predictor, and gradient boosting on twelve
  per-tile statistics (point count, height percentiles, spread). The Point
  Transformer is only interesting if it beats the latter.

## Setup

Requires Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev --extra hub
```

This pins torch 2.8 and installs CPU builds of `torch-cluster`, which work
on macOS and CPU-only Linux. Apple GPUs (MPS) aren't supported by
`torch-cluster`, so Macs train on CPU.

On a CUDA machine (e.g. Colab), install with pip so `torch-cluster` comes
from the CUDA 12.8 build that matches PyPI's Linux torch 2.8:

```bash
pip install -e ".[dev,hub]" --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

CUDA wheels exist for Python 3.10–3.12 on Linux x86_64. The GPU path hasn't
been tested yet.

## Usage

```bash
uv run pcc download                    # all districts, ~8.6 GB downloaded one at a time
uv run pcc tile                        # --tile-size 250 --min-points 256
uv run pcc split                       # writes data/splits.json
uv run pcc baseline                    # writes runs/baseline/summary.md
uv run pcc train --seeds 0 1 2         # writes runs/pct/summary.md
uv run pcc train --shuffle-labels --out runs/pct-shuffled --seeds 0
uv run pytest
```

Every command takes `--help`. For a quick check on a few districts:

```bash
uv run pcc download --districts MN06 MN09 BX02 BX03 BX05 BK13 BK16 QN03 QN04
uv run pcc tile && uv run pcc split
uv run pcc train --seeds 0 --epochs 2 --max-tiles-per-split 64
```

## Repository layout

```
pcc/            pipeline, model, training and evaluation (`pcc` CLI)
tests/          unit tests on synthetic data
legacy/         the original 2024 notebooks and scripts, with notes on how they were evaluated
images/         figures
```

## References

Guo, Meng-Hao, et al. "PCT: Point cloud transformer." Computational Visual Media 7 (2021): 187-199.

Zhao, Hengshuang, et al. "Point Transformer." ICCV 2021. The model here is
adapted from PyTorch Geometric's implementation of this paper.
