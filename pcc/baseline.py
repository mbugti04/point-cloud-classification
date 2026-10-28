"""Non-deep baselines on simple per-tile statistics.

If gradient boosting on a dozen numbers (point count, height percentiles,
spread) matches the Point Transformer, the transformer is not learning much
beyond "how tall and how dense is this area".
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


def tile_features(points: np.ndarray, n_points: int) -> np.ndarray:
    h = points[:, 2]
    xy = points[:, :2]
    return np.array([
        np.log(n_points), h.mean(), *np.percentile(h, [50, 75, 90, 99]), h.max(),
        (h > 10).mean(), (h > 30).mean(), xy[:, 0].std(), xy[:, 1].std(),
        np.linalg.norm(xy, axis=1).mean(),
    ])


def featurize(store: TileStore, tiles: list) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([tile_features(store.points(d, i), store.n_points(d, i)) for d, i in tiles])
    y = np.array([config.label_of(d) for d, _ in tiles])
    return X, y


def run(tiles_dir: Path, splits_path: Path, out_dir: Path) -> dict:
    splits = load_splits(splits_path)
    districts = {d for split in splits["tiles"].values() for d, _ in split}
    store = TileStore(tiles_dir, districts)
    # Val is not needed for tuning here, so it joins train.
    X_train, y_train = featurize(store, splits["tiles"]["train"] + splits["tiles"]["val"])
    X_test, y_test = featurize(store, splits["tiles"]["test"])

    results = {}
    majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    results["majority"] = summarize([classification_metrics(y_test, majority.predict(X_test))])

    # Deterministic without early stopping, so one run is enough.
    clf = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=0)
    clf.fit(X_train, y_train)
    results["gradient_boosting"] = summarize([classification_metrics(y_test, clf.predict(X_test))])

    write_json(out_dir / "summary.json", results)
    md = "".join(format_summary(f"Baseline: {name}, test set", s) + "\n" for name, s in results.items())
    (out_dir / "summary.md").write_text(md)
    print(md)
    return results
