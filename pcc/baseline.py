"""Non-deep baselines on simple per-tile statistics.

If gradient boosting on a dozen numbers (point count, height percentiles,
spread) matches the Point Transformer, the transformer is not learning much
beyond "how tall and how dense is this area".

Ablations retrain gradient boosting with one feature group removed, or on
randomly rotated tiles, to show where its accuracy comes from and so which
signal the Point Transformer is failing to pick up.
"""

from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from .data import TileStore
from .metrics import classification_metrics, format_summary, summarize, write_json
from .splits import load_splits

FEATURE_NAMES = [
    "log_n_points", "height_mean", "height_p50", "height_p75", "height_p90", "height_p99",
    "height_max", "frac_over_10m", "frac_over_30m", "x_std", "y_std", "xy_radius_mean",
]
FEATURE_GROUPS = {
    "density": ["log_n_points"],
    "height": ["height_mean", "height_p50", "height_p75", "height_p90", "height_p99",
               "height_max", "frac_over_10m", "frac_over_30m"],
    # x_std and y_std depend on which way the street grid points; xy_radius_mean does not.
    "spread": ["x_std", "y_std", "xy_radius_mean"],
}


def tile_features(points: np.ndarray, n_points: int) -> np.ndarray:
    h = points[:, 2]
    xy = points[:, :2]
    return np.array([
        np.log(n_points), h.mean(), *np.percentile(h, [50, 75, 90, 99]), h.max(),
        (h > 10).mean(), (h > 30).mean(), xy[:, 0].std(), xy[:, 1].std(),
        np.linalg.norm(xy, axis=1).mean(),
    ])


def rotate_z(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    out = points.copy()
    out[:, :2] = points[:, :2] @ np.array([[c, -s], [s, c]])
    return out


def featurize(store: TileStore, tiles: list, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Features for each tile; with `rng`, each tile is first rotated by a random angle about z."""
    rows = []
    for d, i in tiles:
        points = store.points(d, i)
        if rng is not None:
            points = rotate_z(points, rng.uniform(0, 2 * np.pi))
        rows.append(tile_features(points, store.n_points(d, i)))
    y = np.array([config.label_of(d) for d, _ in tiles])
    return np.stack(rows), y


def columns(drop: list[str] = (), keep: list[str] | None = None) -> list[int]:
    names = keep if keep is not None else [n for n in FEATURE_NAMES if n not in drop]
    return [FEATURE_NAMES.index(n) for n in names]


def fit_score(X_train, y_train, X_test, y_test) -> dict:
    # Deterministic without early stopping, so one run is enough.
    clf = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=0)
    clf.fit(X_train, y_train)
    test = classification_metrics(y_test, clf.predict(X_test))
    return {**summarize([test]), "confusion_matrix": test["confusion_matrix"]}


def comparison_table(results: dict) -> str:
    rows = ["### Baselines, test set", "",
            "| model | balanced accuracy | macro-F1 | " + " | ".join(config.BOROUGHS) + " |",
            "|---|---|---|" + "---|" * len(config.BOROUGHS)]
    for name, r in results.items():
        recall = ["–" if r["per_class_recall"][b] is None else f"{100 * r['per_class_recall'][b]:.1f}%"
                  for b in config.BOROUGHS]
        rows.append(f"| {name} | {100 * r['balanced_accuracy']['mean']:.1f}% | "
                    f"{100 * r['macro_f1']['mean']:.1f}% | " + " | ".join(recall) + " |")
    return "\n".join(rows) + "\n"


def run(tiles_dir: Path, splits_path: Path, out_dir: Path) -> dict:
    splits = load_splits(splits_path)
    districts = {d for split in splits["tiles"].values() for d, _ in split}
    store = TileStore(tiles_dir, districts)
    # Val is not needed for tuning here, so it joins train.
    train_tiles = splits["tiles"]["train"] + splits["tiles"]["val"]
    X_train, y_train = featurize(store, train_tiles)
    X_test, y_test = featurize(store, splits["tiles"]["test"])

    results = {}
    majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    test = classification_metrics(y_test, majority.predict(X_test))
    results["majority"] = {**summarize([test]), "confusion_matrix": test["confusion_matrix"]}

    results["gradient_boosting"] = fit_score(X_train, y_train, X_test, y_test)
    for group, names in FEATURE_GROUPS.items():
        cols = columns(drop=names)
        results[f"gb_no_{group}"] = fit_score(X_train[:, cols], y_train, X_test[:, cols], y_test)
    cols = columns(keep=FEATURE_GROUPS["height"])
    results["gb_height_only"] = fit_score(X_train[:, cols], y_train, X_test[:, cols], y_test)

    # Same features, but every train and test tile is rotated at random first,
    # like the Point Transformer's training augmentation. Only x_std and y_std
    # can change, so any drop measures how much the street grid's direction helps.
    X_train_rot, _ = featurize(store, train_tiles, np.random.default_rng(0))
    X_test_rot, _ = featurize(store, splits["tiles"]["test"], np.random.default_rng(1))
    results["gb_rotated_tiles"] = fit_score(X_train_rot, y_train, X_test_rot, y_test)

    write_json(out_dir / "summary.json", results)
    md = comparison_table(results) + "\n" + "".join(
        format_summary(f"Baseline: {name}, test set", s) + "\n" for name, s in results.items())
    (out_dir / "summary.md").write_text(md)
    print(md)
    return results
