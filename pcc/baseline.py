"""Non-deep baselines on simple per-tile statistics.

If gradient boosting on a dozen numbers (point count, height percentiles,
spread) matches the Point Transformer, the transformer is not learning much
beyond "how tall and how dense is this area".

Ablations retrain gradient boosting with one feature group removed, or on
randomly rotated tiles, to show where its accuracy comes from and so which
signal the Point Transformer is failing to pick up.
"""

import warnings
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from . import config
from .data import TileStore
from .features import FEATURE_GROUPS, columns, tile_features
from .metrics import classification_metrics, format_summary, summarize, write_json
from .splits import load_splits

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


def fit_score(X_train, y_train, X_test, y_test) -> dict:
    # Deterministic without early stopping, so one run is enough.
    clf = HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=0)
    clf.fit(X_train, y_train)
    test = classification_metrics(y_test, clf.predict(X_test))
    return {**summarize([test]), "confusion_matrix": test["confusion_matrix"]}


def mlp_fit_score(X_train, y_train, X_test, y_test, seeds=(0, 1, 2)) -> dict:
    """A small neural network on the same features, class-balanced like the others."""
    scaler = StandardScaler().fit(X_train)
    weights = compute_sample_weight("balanced", y_train)
    per_seed = []
    for seed in seeds:
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=seed)
        with warnings.catch_warnings():
            # A fixed iteration budget is fine here; the loss is still creeping down at the end.
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(scaler.transform(X_train), y_train, sample_weight=weights)
        per_seed.append(classification_metrics(y_test, clf.predict(scaler.transform(X_test))))
    return summarize(per_seed)


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
    n_train_only = len(splits["tiles"]["train"])  # the train split comes first
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

    # The Point Transformer never trains on val (it picks its checkpoint
    # there), so these match its training data exactly.
    only = slice(n_train_only)
    height = columns(keep=FEATURE_GROUPS["height"])
    results["gb_train_only"] = fit_score(X_train[only], y_train[only], X_test, y_test)
    results["gb_height_only_train_only"] = fit_score(X_train[only][:, height], y_train[only], X_test[:, height], y_test)

    # A neural network on the 8 height statistics: if it matches gradient
    # boosting, the Point Transformer's gap is not about neural nets per se.
    results["mlp_height_only"] = mlp_fit_score(X_train[:, height], y_train, X_test[:, height], y_test)
    results["mlp_height_only_train_only"] = mlp_fit_score(X_train[only][:, height], y_train[only], X_test[:, height], y_test)

    write_json(out_dir / "summary.json", results)
    md = comparison_table(results) + "\n" + "".join(
        format_summary(f"Baseline: {name}, test set", s) + "\n" for name, s in results.items())
    (out_dir / "summary.md").write_text(md)
    print(md)
    return results
