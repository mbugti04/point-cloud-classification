# Point Cloud Classification

![](images/niv-rozenberg-ySNtJhDX-cw-unsplash1.jpg)
Photo by <a href="https://unsplash.com/@nivroz?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Niv Rozenberg</a> on <a href="https://unsplash.com/photos/aerial-view-of-city-during-day-time-ySNtJhDX-cw?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>

Can a neural network tell which New York City borough a neighborhood is in,
just from the 3D shape of its buildings? This project converts the city's
official 3D building model into labeled samples and trains two transformers
to classify them by borough. The transformers include a Point Transformer on raw points, and a
transformer whose tokens are individual buildings.

This improves upon the 2024 version of my project, found in the [legacy](legacy/) folder.

## Results
The 2024 version reported 83% accuracy, chosen from the best epoch on a test set of about 30 samples. See [legacy/README.md](legacy/README.md).

The new version tests on community districts never seen in training. Scores are
on the test set, with chance at 20% balanced accuracy. Rows marked ± are mean ±
std over 3 seeds. Other deep-model rows are a single seed, and gradient boosting
rows are one fit.

| model | accuracy | balanced accuracy | macro-F1 |
|---|---|---|---|
| majority class | 34.7% | 20.0% | 10.3% |
| Point Transformer | 43.3% | 40.3% | 41.5% |
| Point Transformer + height statistics, regularized | 46.5% | 45.9% | 48.0% |
| gradient boosting on 12 tile statistics | 53.4% | 56.7% | 56.2% |
| building transformer | 61.0 ± 2.2% | 61.8 ± 3.4% | 61.9 ± 3.1% |
| gradient boosting on building + height statistics | 65.1% | 67.2% | 66.6% |
| **building transformer + height statistics** | **68.2 ± 0.5%** | **70.8 ± 0.8%** | **70.5 ± 0.3%** |

Apart from the majority class, these all trained on the train split only. With training labels shuffled,
the best model falls to 21.4% balanced accuracy, close to chance.

All rows share one split. 

## Data

The [NYC 3D Model by Community District](https://www.nyc.gov/content/planning/pages/resources/datasets/nyc-3d-model)
covers every building in the city as of 2014, split into the 59 community
districts and stored in Rhino's `.3dm` format. The pipeline:

1. **Download and extract** (`pcc download`). Reads each `.3dm` with
   [`rhino3dm`](https://github.com/mcneel/rhino3dm), keeps building vertices
   (facade, roof and footprint corners), converts millimeters to meters, and
   drops duplicate vertices. Districts are handled one at a time and the raw files deleted, so
   the ~7.8 GB of zips are never on disk together. Each file is parsed in a
   fresh subprocess to work around memory that `rhino3dm` never frees.
2. **Tile** (`pcc tile`). Cuts each district into 250 m tiles on a citywide grid.
   Each tile is one sample. Coordinates are made local: x and y relative to the
   tile center, z as height above the nearest building footprint. The model
   can't use where a tile is in the city or how high the ground is.
3. **Split** (`pcc split`). Groups by community district, so test
   neighborhoods are never seen in training. Per borough, about 25% of
   districts go to test and 15% to validation.

   Note that with Staten Island, the city's download link for its third
   district, SI03, is dead, leaving two districts. One goes to test and the
   other to train, so no whole district is left for validation. Staten
   Island's validation tiles are instead a strip cut from one end of the train
   district, with one row of tiles dropped between the strip and the rest so
   they don't touch.

## Evaluation

- **Checkpoint selection on validation only.** Each seed keeps the epoch with
  the best validation macro-F1. Nothing is chosen using the test set: it is
  only scored after training, with that checkpoint.
- **Multiple seeds.** Results are mean ± std over seeds (default 0, 1, 2).
- **Class balance.** Boroughs have very different tile counts, so the loss is
  weighted by inverse class frequency, and balanced accuracy and macro-F1 are
  reported alongside accuracy.
- **Baselines.** A majority-class predictor, and gradient boosting on twelve
  per-tile statistics (point count, height percentiles, spread), plus versions
  using building statistics. A deep model is only interesting if it beats these.
- **Shuffled-label control.** Retraining with scrambled training labels should
  drop scores to chance, which checks that nothing leaks into the score.

## Setup

Requires Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/).

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

CUDA wheels exist for Python 3.10–3.13 on Linux x86_64. Results above were
trained this way on a Kaggle T4 GPU.

## Usage

```bash
uv run pcc download                    # all districts, ~7.8 GB downloaded one at a time
uv run pcc tile                        # --tile-size 250 --min-points 256
uv run pcc split                       # writes data/splits.json
uv run pcc baseline                    # writes runs/baseline/summary.md
uv run pcc train --seeds 0 1 2         # writes runs/pct/summary.md
uv run pcc train --shuffle-labels --out runs/pct-shuffled --seeds 0   # control: shuffled labels, scores should fall to chance
uv run pytest
```

Every command takes `--help`. For a quick check on a few districts:

```bash
uv run pcc download --districts MN06 MN09 BX02 BX03 BX05 BK13 BK16 QN03 QN04
uv run pcc tile && uv run pcc split
uv run pcc train --seeds 0 --epochs 2 --max-tiles-per-split 64
```

### Building transformer

A second model treats each building as one token (local position, height,
footprint area, compactness, roof levels, distance to the nearest building)
and runs a transformer encoder over the buildings of a tile. Building tiles
reuse the point tiles' grid and indices, so the same `splits.json` applies.

```bash
uv run pcc download --what buildings    # skip if `pcc download` already ran (it extracts both)
uv run pcc tile-buildings               # writes data/building_tiles/, needs data/tiles/
uv run pcc baseline --building-tiles-dir data/building_tiles   # adds gradient boosting on building aggregates
uv run pcc train --model building-transformer --dropout 0.1 --out runs/buildings
uv run pcc train --model building-transformer --dropout 0.1 --tile-stats --out runs/buildings-stats   # also uses data/tiles/
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
