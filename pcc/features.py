"""Per-tile summary statistics, shared by the baselines and the hybrid model."""

import numpy as np

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


def columns(drop: list[str] = (), keep: list[str] | None = None) -> list[int]:
    names = keep if keep is not None else [n for n in FEATURE_NAMES if n not in drop]
    return [FEATURE_NAMES.index(n) for n in names]
