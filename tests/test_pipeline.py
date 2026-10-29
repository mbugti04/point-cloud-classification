import numpy as np
import pytest
import torch

from pcc import config
from pcc.extract import dedupe
from pcc.splits import assign_districts, make_splits, strip_split
from pcc.tiles import height_above_ground, tile_all, tile_district


def test_dedupe_drops_coincident_points_and_keeps_order():
    xyz = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.001], [5.0, 5.0, 5.0]])
    out = dedupe(xyz)
    np.testing.assert_array_equal(out, xyz[[0, 1, 3]])


def test_height_is_measured_from_nearest_footprint():
    ground = np.array([[0.0, 0.0, 10.0], [100.0, 0.0, 50.0]])
    xyz = np.array([[1.0, 0.0, 30.0], [99.0, 0.0, 60.0]])
    np.testing.assert_allclose(height_above_ground(xyz, ground), [20.0, 10.0])


def test_tiles_are_local_and_sparse_tiles_are_dropped():
    rng = np.random.default_rng(0)
    dense = rng.uniform([1000, 2000, 0], [1250, 2250, 40], size=(500, 3))  # one full tile
    sparse = rng.uniform([1500, 2000, 0], [1750, 2250, 40], size=(10, 3))  # below min_points
    xyz = np.vstack([dense, sparse])
    ground = np.column_stack([xyz[:, :2], np.zeros(len(xyz))])

    tiles = tile_district(xyz, ground, tile_size=250, min_points=100, max_points=300, rng=rng)

    assert len(tiles["ij"]) == 1
    assert tiles["ij"][0].tolist() == [4, 8]
    assert tiles["n_points"][0] == 500
    assert tiles["offsets"].tolist() == [0, 300]  # capped at max_points
    assert np.abs(tiles["points"][:, :2]).max() <= 125  # centered on the tile
    # Ground is at z=0, so height above ground equals the original z.
    assert np.isin(tiles["points"][:, 2], dense[:, 2].astype(np.float32)).all()


def test_every_borough_gets_a_test_district_and_districts_never_straddle_splits():
    districts = [d for d in config.DISTRICTS if d != "SI03"]
    assignment = assign_districts(districts, seed=0)
    assert set(assignment) == set(districts)
    for borough in config.BOROUGHS:
        roles = [r for d, r in assignment.items() if config.borough_of(d) == borough]
        assert "test" in roles
        assert any(r.startswith("train") for r in roles)
    # Staten Island has two districts, so it falls back to a val strip.
    si = {d: r for d, r in assignment.items() if d.startswith("SI")}
    assert sorted(si.values()) == ["test", "train+val_strip"]
    # Boroughs with enough districts get a whole val district.
    assert "val" in [r for d, r in assignment.items() if d.startswith("MN")]


def test_split_is_reproducible():
    districts = config.DISTRICTS
    assert assign_districts(districts, seed=3) == assign_districts(districts, seed=3)
    assert assign_districts(districts, seed=3) != assign_districts(districts, seed=4)


def test_strip_split_leaves_a_buffer_row():
    ij = np.array([[x, y] for x in range(10) for y in range(3)])
    labels = strip_split(ij, fraction=0.2)
    assert set(labels[ij[:, 0] >= 8]) == {"val"}
    assert set(labels[ij[:, 0] == 7]) == {"buffer"}
    assert set(labels[ij[:, 0] < 7]) == {"train"}


@pytest.fixture
def tiny_tiles(tmp_path):
    rng = np.random.default_rng(0)
    interim = tmp_path / "interim"
    interim.mkdir()
    for district in ["MN01", "MN02", "MN03", "SI01", "SI02"]:
        xyz = rng.uniform([0, 0, 0], [1000, 500, 30], size=(4000, 3))
        np.savez(interim / f"{district}.npz", xyz=xyz, ground=np.column_stack([xyz[:, :2], np.zeros(4000)]))
    tiles_dir = tmp_path / "tiles"
    tile_all(interim, tiles_dir, tile_size=250, min_points=50, max_points=1000)
    return tiles_dir


def test_make_splits_keeps_districts_whole(tiny_tiles):
    splits = make_splits(tiny_tiles, seed=0)
    seen = {}
    for name, tiles in splits["tiles"].items():
        for district, _ in tiles:
            seen.setdefault(district, set()).add(name)
    for district, names in seen.items():
        role = splits["districts"][district]
        if role == "train+val_strip":
            assert names == {"train", "val"}
        else:
            assert names == {role}


def test_dataset_and_model_shapes(tiny_tiles):
    from torch_geometric.loader import DataLoader

    from pcc.data import TileDataset, TileStore
    from pcc.model import PointTransformerClassifier

    store = TileStore(tiny_tiles, {"MN01", "SI01"})
    tiles = [("MN01", 0), ("MN01", 1), ("SI01", 0)]
    eval_ds = TileDataset(store, tiles, num_points=128, train=False)
    item = eval_ds[0]
    assert item.pos.shape == (128, 3)
    assert torch.equal(item.pos, eval_ds[0].pos)  # eval sampling is deterministic
    assert eval_ds[2].y.item() == config.BOROUGHS.index("SI")
    # Density is the original vertex count, not the resampled one.
    assert item.density.item() == pytest.approx(np.log(store.n_points("MN01", 0)))

    batch = next(iter(DataLoader(TileDataset(store, tiles, 128, train=True), batch_size=3)))
    assert batch.density.shape == (3, 1)
    for use_density in (True, False):
        model = PointTransformerClassifier(len(config.BOROUGHS), dims=(16, 32), k=8, use_density=use_density)
        out = model(batch.pos, batch.batch, batch.density)
        assert out.shape == (3, len(config.BOROUGHS))


def test_rotation_only_changes_direction_dependent_features():
    from pcc.baseline import FEATURE_NAMES, rotate_z, tile_features

    rng = np.random.default_rng(0)
    points = rng.uniform([-125, -20, 0], [125, 20, 40], size=(500, 3))  # long in x
    before = tile_features(points, 500)
    after = tile_features(rotate_z(points, np.pi / 2), 500)
    changed = {name for name, a, b in zip(FEATURE_NAMES, before, after) if not np.isclose(a, b)}
    assert changed == {"x_std", "y_std"}
