"""Train and evaluate the Point Transformer over several seeds.

For each seed the checkpoint with the best validation macro-F1 is kept, and
the test set is scored exactly once, with that checkpoint.
"""

import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from . import config
from .data import TileDataset, TileStore
from .metrics import classification_metrics, format_summary, summarize, write_json
from .model import PointTransformerClassifier
from .splits import load_splits


@dataclass
class TrainConfig:
    tiles_dir: str = str(config.TILES_DIR)
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
        pos = rotate_z(batch.pos, theta) if theta else batch.pos
        out = model(pos, batch.batch)
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


def train_one_seed(cfg: TrainConfig, seed: int, store: TileStore, splits: dict, out_dir: Path) -> dict:
    set_seed(seed)
    device = resolve_device(cfg.device)
    num_classes = len(config.BOROUGHS)

    tiles = {k: _subset(v, cfg.max_tiles_per_split, seed) for k, v in splits["tiles"].items()}
    train_ds = TileDataset(store, tiles["train"], cfg.num_points, train=True, seed=seed,
                           rotate=cfg.rotate, shuffle_labels=cfg.shuffle_labels)
    val_ds = TileDataset(store, tiles["val"], cfg.num_points, train=False, seed=seed)
    test_ds = TileDataset(store, tiles["test"], cfg.num_points, train=False, seed=seed)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # Weight classes by inverse training frequency so big boroughs don't dominate.
    counts = train_ds.class_counts(num_classes)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / num_classes, dtype=torch.float, device=device)

    model = PointTransformerClassifier(num_classes, tuple(cfg.dims), cfg.k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
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
            loss = F.nll_loss(model(batch.pos, batch.batch), batch.y, weight=weights)
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
    store = TileStore(Path(cfg.tiles_dir), districts)
    write_json(out_dir / "config.json", {**asdict(cfg), "split_seed": splits["seed"],
                                         "split_sizes": {k: len(v) for k, v in splits["tiles"].items()}})

    per_seed = [train_one_seed(cfg, seed, store, splits, out_dir) for seed in cfg.seeds]
    summary = summarize(per_seed)
    write_json(out_dir / "summary.json", summary)
    md = format_summary(f"Point Transformer, test set{' (SHUFFLED LABELS)' if cfg.shuffle_labels else ''}", summary)
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)
    return summary
