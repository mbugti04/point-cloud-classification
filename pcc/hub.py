"""Publish the processed tiles to the Hugging Face Hub as a dataset repo."""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from . import config

CARD_TEMPLATE = Path(__file__).with_name("dataset_card.md")


def _stats(tiles_dir: Path, splits: dict) -> str:
    rows = ["| split | " + " | ".join(config.BOROUGHS) + " | total |", "|---" * (len(config.BOROUGHS) + 2) + "|"]
    for name, tiles in splits["tiles"].items():
        counts = {b: 0 for b in config.BOROUGHS}
        for district, _ in tiles:
            counts[config.borough_of(district)] += 1
        rows.append(f"| {name} | " + " | ".join(str(counts[b]) for b in config.BOROUGHS) + f" | {len(tiles)} |")
    return "\n".join(rows)


def build_card(tiles_dir: Path, splits: dict) -> str:
    tile_size = float(np.load(next(tiles_dir.glob("*.npz")))["tile_size"])
    held_out = {r: sorted(d for d, role in splits["districts"].items() if role.startswith(r))
                for r in ("train", "val", "test")}
    return (CARD_TEMPLATE.read_text()
            .replace("{{tile_size}}", f"{tile_size:g}")
            .replace("{{split_seed}}", str(splits["seed"]))
            .replace("{{split_table}}", _stats(tiles_dir, splits))
            .replace("{{val_districts}}", ", ".join(held_out["val"]) or "none")
            .replace("{{test_districts}}", ", ".join(held_out["test"])))


def publish(tiles_dir: Path, splits_path: Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi

    splits = json.loads(splits_path.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        shutil.copytree(tiles_dir, staging / "tiles")
        shutil.copy(splits_path, staging / "splits.json")
        (staging / "README.md").write_text(build_card(tiles_dir, splits))
        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(folder_path=staging, repo_id=repo_id, repo_type="dataset",
                          commit_message="Upload NYC 3D tiles")
    return f"https://huggingface.co/datasets/{repo_id}"
