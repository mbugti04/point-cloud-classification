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
    assert "extra" not in item

    # Heights are scaled separately from x and y.
    scaled = TileDataset(store, tiles, 128, train=False, height_scale=25.0)[0]
    torch.testing.assert_close(scaled.pos[:, 2], item.pos[:, 2] * store.tile_size / 2 / 25.0)

    # Extra per-tile features: log vertex count (from the original count, not
    # the resampled one) followed by the 8 height statistics.
    hybrid = TileDataset(store, tiles, 128, train=True, density=True, tile_stats=True)
    assert hybrid[0].extra.shape == (1, 9)
    assert hybrid[0].extra[0, 0].item() == pytest.approx(np.log(store.n_points("MN01", 0)))

    for ds, extra_dim in ((TileDataset(store, tiles, 128, train=True), 0), (hybrid, 9)):
        batch = next(iter(DataLoader(ds, batch_size=3)))
        model = PointTransformerClassifier(len(config.BOROUGHS), dims=(16, 32), k=8, extra_dim=extra_dim)
        out = model(batch)
        assert out.shape == (3, len(config.BOROUGHS))


def test_rotation_only_changes_direction_dependent_features():
    from pcc.baseline import rotate_z
    from pcc.features import FEATURE_NAMES, tile_features

    rng = np.random.default_rng(0)
    points = rng.uniform([-125, -20, 0], [125, 20, 40], size=(500, 3))  # long in x
    before = tile_features(points, 500)
    after = tile_features(rotate_z(points, np.pi / 2), 500)
    changed = {name for name, a, b in zip(FEATURE_NAMES, before, after) if not np.isclose(a, b)}
    assert changed == {"x_std", "y_std"}


def test_polygon_area_and_perimeter():
    from pcc.buildings import polygon_area_perimeter

    square = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float)
    assert polygon_area_perimeter(square) == pytest.approx((50.0, 30.0))


def _fake_buildings(rng, n, lo, hi):
    return {
        "centroid": rng.uniform(lo, hi, size=(n, 2)),
        "height": rng.uniform(3, 40, n), "area": rng.uniform(50, 500, n), "perimeter": rng.uniform(30, 100, n),
        "n_vertices": rng.integers(4, 12, n).astype(float), "n_roofs": rng.integers(1, 3, n).astype(float),
        "roof_range": rng.uniform(0, 5, n),
    }


def test_building_tiles_follow_point_tiles_exactly():
    from pcc.buildings import TOKEN_FEATURES, tile_buildings

    rng = np.random.default_rng(0)
    # Buildings fill cells (4, 8) and (5, 8); point tiles also include the empty cell (6, 8).
    buildings = _fake_buildings(rng, 40, [1000, 2000], [1500, 2250])
    ij = np.array([[5, 8], [6, 8], [4, 8]])  # order as stored in the point tiles
    tiles = tile_buildings(buildings, ij, tile_size=250)

    counts = np.diff(tiles["offsets"])
    assert len(counts) == len(ij)
    assert counts[1] == 0  # empty tile kept, not dropped
    assert counts.sum() == 40
    assert tiles["tokens"].shape == (40, 3 + len(TOKEN_FEATURES))
    assert np.abs(tiles["tokens"][:, :2]).max() <= 125  # local to the tile center
    first = tiles["tokens"][tiles["offsets"][0]:tiles["offsets"][1]]
    expected = np.sort(buildings["height"][np.floor(buildings["centroid"][:, 0] / 250) == 5])
    np.testing.assert_allclose(np.sort(first[:, 2]), expected, rtol=1e-5)


def test_building_transformer_handles_empty_tiles(tmp_path):
    from torch_geometric.loader import DataLoader

    from pcc.buildings import BuildingStore, tile_buildings
    from pcc.data import BuildingTileDataset, TileStore
    from pcc.features import BUILDING_FEATURE_NAMES, building_features
    from pcc.model import BuildingTransformer

    rng = np.random.default_rng(0)
    ij = np.array([[4, 8], [6, 8]])
    tiles = tile_buildings(_fake_buildings(rng, 30, [1000, 2000], [1250, 2250]), ij, tile_size=250)
    np.savez(tmp_path / "MN01.npz", tile_size=250.0, ij=ij, **tiles)
    store = BuildingStore(tmp_path, {"MN01"})

    ds = BuildingTileDataset(store, [("MN01", 0), ("MN01", 1)], train=True, max_buildings=16)
    assert ds[0].pos.shape == (16, 3)  # capped
    assert ds[1].num_nodes == 0
    feats = ds.all_features()
    model = BuildingTransformer(len(config.BOROUGHS), feats.shape[1], width=16, layers=1, heads=2)
    model.set_feature_stats(torch.from_numpy(feats.mean(0)), torch.from_numpy(feats.std(0)))
    model.eval()
    out = model(next(iter(DataLoader(ds, batch_size=2))))  # empty tile last
    assert out.shape == (2, len(config.BOROUGHS))
    assert torch.isfinite(out).all()

    # Height statistics from the point tiles join the [CLS] output, even for the empty tile.
    points = np.random.default_rng(1).uniform(0, 30, (600, 3)).astype(np.float32)
    np.savez(tmp_path / "points.npz", tile_size=250.0, ij=ij, points=points, offsets=np.array([0, 300, 600]),
             n_points=np.array([300, 300]))
    (tmp_path / "pts").mkdir()
    (tmp_path / "points.npz").rename(tmp_path / "pts" / "MN01.npz")
    hybrid_store = BuildingStore(tmp_path, {"MN01"}, TileStore(tmp_path / "pts", {"MN01"}))
    hybrid = BuildingTileDataset(hybrid_store, [("MN01", 0), ("MN01", 1)], train=False, tile_stats=True)
    model = BuildingTransformer(len(config.BOROUGHS), feats.shape[1], width=16, layers=1, heads=2,
                                extra_dim=hybrid.extra.shape[1])
    model.set_extra_stats(torch.from_numpy(hybrid.extra.mean(0)), torch.from_numpy(hybrid.extra.std(0) + 1e-6))
    model.eval()
    out = model(next(iter(DataLoader(hybrid, batch_size=2))))
    assert out.shape == (2, len(config.BOROUGHS)) and torch.isfinite(out).all()

    empty = building_features(store.tokens("MN01", 1), 250.0)
    assert len(empty) == len(BUILDING_FEATURE_NAMES) and empty[0] == 0
