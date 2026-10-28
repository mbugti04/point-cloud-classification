"""Classification metrics and multi-seed summaries."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from . import config


def classification_metrics(y_true, y_pred) -> dict:
    labels = list(range(len(config.BOROUGHS)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = cm.diagonal() / cm.sum(axis=1)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # Average over boroughs present in y_true, matching balanced accuracy.
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0)),
        "per_class_recall": {b: (None if np.isnan(r) else float(r)) for b, r in zip(config.BOROUGHS, recall)},
        "confusion_matrix": cm.tolist(),
    }


SUMMARY_KEYS = ("accuracy", "balanced_accuracy", "macro_f1")


def summarize(per_seed: list[dict]) -> dict:
    """Mean and sample std of the headline metrics across seeds."""
    out = {"seeds": len(per_seed)}
    for key in SUMMARY_KEYS:
        values = np.array([m[key] for m in per_seed])
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return out


def format_summary(name: str, summary: dict) -> str:
    rows = [f"### {name} ({summary['seeds']} seed{'s' if summary['seeds'] != 1 else ''})", "",
            "| metric | mean ± std |", "|---|---|"]
    for key in SUMMARY_KEYS:
        s = summary[key]
        rows.append(f"| {key} | {100 * s['mean']:.1f} ± {100 * s['std']:.1f}% |")
    return "\n".join(rows) + "\n"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
