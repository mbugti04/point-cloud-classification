"""Train and evaluate a model over several seeds: the Point Transformer on
point tiles, or the building transformer on building tiles.

For each seed the checkpoint with the best validation macro-F1 is kept, and
the test set is scored exactly once, with that checkpoint.
"""

import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from . import config
from .buildings import BuildingStore
from .data import BuildingTileDataset, TileDataset, TileStore
from .metrics import classification_metrics, format_summary, summarize, write_json
from .model import BuildingTransformer, PointTransformerClassifier
from .splits import load_splits


MODELS = ("point-transformer", "building-transformer")


@dataclass
class TrainConfig:
    model: str = "point-transformer"
    tiles_dir: str = str(config.TILES_DIR)
    building_tiles_dir: str = str(config.BUILDING_TILES_DIR)
    splits_path: str = str(config.SPLITS_PATH)
    out_dir: str = str(config.RUNS_DIR / "pct")
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    epochs: int = 60
    batch_size: int = 16
    num_points: int = 1024
    lr: float = 1e-3
    lr_step: int = 20
    lr_gamma: float = 0.5
    dims: tuple[int, ...] = (32, 64, 128, 256, 512)
    k: int = 16
    rotate: bool = True
    height_scale: float | None = None
    density: bool = False
    tile_stats: bool = False
    dropout: float = 0.0
    weight_decay: float = 0.0
    balance_districts: bool = False
    # building-transformer only
    width: int = 64
    layers: int = 3
    heads: int = 4
    max_buildings: int = 512
    shuffle_labels: bool = False
    max_tiles_per_split: int | None = None
    num_workers: int = 0
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    # torch-cluster's fps/knn have no MPS kernels, so Apple GPUs fall back to CPU.
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def rotate_z(pos: torch.Tensor, theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=pos.device)
    return pos @ rot


@torch.no_grad()
def predict(model, loader, device, theta: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds, labels, total_loss = [], [], 0.0
    for batch in loader:
        batch = batch.to(device)
        if theta:
            batch.pos = rotate_z(batch.pos, theta)
        out = model(batch)
        total_loss += F.nll_loss(out, batch.y, reduction="sum").item()
        preds.append(out.argmax(dim=1).cpu().numpy())
        labels.append(batch.y.cpu().numpy())
    y_pred, y_true = np.concatenate(preds), np.concatenate(labels)
    return y_true, y_pred, total_loss / len(y_true)


def _subset(tiles: list, limit: int | None, seed: int) -> list:
    if limit is None or len(tiles) <= limit:
        return tiles
    idx = np.random.default_rng(seed).choice(len(tiles), limit, replace=False)
    return [tiles[i] for i in sorted(idx)]


def _std(values: np.ndarray) -> torch.Tensor:
    std = values.std(axis=0)
    return torch.from_numpy(np.where(std > 0, std, 1.0).astype(np.float32))


def make_datasets(cfg: TrainConfig, store, tiles: dict, seed: int) -> tuple:
    if cfg.model == "building-transformer":
        common = dict(seed=seed, height_scale=cfg.height_scale or 30.0, max_buildings=cfg.max_buildings)
        return (BuildingTileDataset(store, tiles["train"], train=True, rotate=cfg.rotate,
                                    shuffle_labels=cfg.shuffle_labels, **common),
                BuildingTileDataset(store, tiles["val"], train=False, **common),
                BuildingTileDataset(store, tiles["test"], train=False, **common))
    inputs = dict(height_scale=cfg.height_scale, density=cfg.density, tile_stats=cfg.tile_stats)
    return (TileDataset(store, tiles["train"], cfg.num_points, train=True, seed=seed,
                        rotate=cfg.rotate, shuffle_labels=cfg.shuffle_labels, **inputs),
            TileDataset(store, tiles["val"], cfg.num_points, train=False, seed=seed, **inputs),
            TileDataset(store, tiles["test"], cfg.num_points, train=False, seed=seed, **inputs))


def make_model(cfg: TrainConfig, train_ds, num_classes: int) -> torch.nn.Module:
    """Build the model, with feature standardization taken from the training tiles."""
    if cfg.model == "building-transformer":
        features = train_ds.all_features()
        model = BuildingTransformer(num_classes, features.shape[1], cfg.width, cfg.layers, cfg.heads, cfg.dropout)
        model.set_feature_stats(torch.from_numpy(features.mean(axis=0)), _std(features))
        return model
    extra_dim = 0 if train_ds.extra is None else train_ds.extra.shape[1]
    model = PointTransformerClassifier(num_classes, tuple(cfg.dims), cfg.k, extra_dim, dropout=cfg.dropout)
    if extra_dim:
        model.set_extra_stats(torch.from_numpy(train_ds.extra.mean(axis=0)), _std(train_ds.extra))
    return model


def train_one_seed(cfg: TrainConfig, seed: int, store, splits: dict, out_dir: Path) -> dict:
    set_seed(seed)
    device = resolve_device(cfg.device)
    num_classes = len(config.BOROUGHS)

    tiles = {k: _subset(v, cfg.max_tiles_per_split, seed) for k, v in splits["tiles"].items()}
    train_ds, val_ds, test_ds = make_datasets(cfg, store, tiles, seed)

    generator = torch.Generator().manual_seed(seed)
    if cfg.balance_districts:
        # Every district contributes equally per epoch, so large districts
        # can't dominate what the model learns about their borough.
        per_district = Counter(d for d, _ in train_ds.tiles)
        sample_weights = np.array([1 / per_district[d] for d, _ in train_ds.tiles])
        sampler = WeightedRandomSampler(sample_weights, len(train_ds), generator=generator)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    else:
        sample_weights = np.ones(len(train_ds))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # Weight classes by inverse frequency (as sampled) so big boroughs don't dominate.
    counts = np.bincount(train_ds.labels, weights=sample_weights, minlength=num_classes)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / num_classes, dtype=torch.float, device=device)

    model = make_model(cfg, train_ds, num_classes).to(device)
    # AdamW with weight_decay=0 is the same as Adam.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best = {"macro_f1": -1.0, "epoch": None}
    history = []

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = F.nll_loss(model(batch), batch.y, weight=weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        scheduler.step()

        y_true, y_pred, val_loss = predict(model, val_loader, device)
        val = classification_metrics(y_true, y_pred)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_ds),
            "val_loss": val_loss,
            "val_accuracy": val["accuracy"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_macro_f1": val["macro_f1"],
            "seconds": time.time() - start,
        })
        improved = val["macro_f1"] > best["macro_f1"]
        if improved:
            best = {"macro_f1": val["macro_f1"], "epoch": epoch}
            torch.save(model.state_dict(), seed_dir / "best.pt")
        print(f"[seed {seed}] epoch {epoch:03d}  train_loss {history[-1]['train_loss']:.4f}  "
              f"val_loss {val_loss:.4f}  val_macro_f1 {val['macro_f1']:.3f}{'  *' if improved else ''}")

    model.load_state_dict(torch.load(seed_dir / "best.pt", map_location=device))
    y_true, y_pred, _ = predict(model, test_loader, device)
    test = classification_metrics(y_true, y_pred)
    test["best_epoch"] = best["epoch"]
    test["best_val_macro_f1"] = best["macro_f1"]

    # Rotating every test tile should barely change predictions if the model
    # learned building shapes rather than which way the street grid points.
    # With --no-rotate a drop here shows how much it leans on grid direction.
    rotated = []
    for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False)[1:]:
        _, y_rot, _ = predict(model, test_loader, device, theta=float(theta))
        rotated.append(classification_metrics(y_true, y_rot)["macro_f1"])
    test["rotated_macro_f1_mean"] = float(np.mean(rotated))

    write_json(seed_dir / "history.json", history)
    write_json(seed_dir / "test_metrics.json", test)
    print(f"[seed {seed}] best epoch {best['epoch']}  test acc {test['accuracy']:.3f}  "
          f"balanced {test['balanced_accuracy']:.3f}  macro_f1 {test['macro_f1']:.3f}  "
          f"rotated macro_f1 {test['rotated_macro_f1_mean']:.3f}")
    return test


def run(cfg: TrainConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    splits = load_splits(Path(cfg.splits_path))
    districts = {d for split in splits["tiles"].values() for d, _ in split}
    if cfg.model not in MODELS:
        raise ValueError(f"Unknown model {cfg.model!r}; choose from {MODELS}")
    if cfg.model == "building-transformer":
        store = BuildingStore(Path(cfg.building_tiles_dir), districts)
    else:
        store = TileStore(Path(cfg.tiles_dir), districts)
    write_json(out_dir / "config.json", {**asdict(cfg), "split_seed": splits["seed"],
                                         "split_sizes": {k: len(v) for k, v in splits["tiles"].items()}})

    per_seed = [train_one_seed(cfg, seed, store, splits, out_dir) for seed in cfg.seeds]
    summary = summarize(per_seed)
    write_json(out_dir / "summary.json", summary)
    name = "Building transformer" if cfg.model == "building-transformer" else "Point Transformer"
    md = format_summary(f"{name}, test set{' (SHUFFLED LABELS)' if cfg.shuffle_labels else ''}", summary)
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)
    return summary
