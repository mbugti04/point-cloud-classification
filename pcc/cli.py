"""Command-line entry point: `pcc <command>` (or `python -m pcc <command>`)."""

import argparse
from pathlib import Path

from . import config


def _download(args):
    from .download import download_all

    districts = [d.upper() for d in args.districts] if args.districts else config.DISTRICTS
    download_all(districts, args.raw_dir, args.interim_dir, keep_raw=args.keep_raw)


def _tile(args):
    from .tiles import tile_all

    counts = tile_all(args.interim_dir, args.tiles_dir, args.tile_size, args.min_points, args.max_points, args.seed)
    by_borough = {}
    for district, n in counts.items():
        by_borough[config.borough_of(district)] = by_borough.get(config.borough_of(district), 0) + n
    print(f"{sum(counts.values())} tiles from {len(counts)} districts: {by_borough}")


def _split(args):
    from .splits import make_splits, save_splits

    splits = make_splits(args.tiles_dir, args.seed)
    save_splits(splits, args.out)
    for name, tiles in splits["tiles"].items():
        per_borough = {b: 0 for b in config.BOROUGHS}
        for district, _ in tiles:
            per_borough[config.borough_of(district)] += 1
        print(f"{name:5s} {len(tiles):6d} tiles  {per_borough}")
    print(f"Wrote {args.out}")


def _train(args):
    from .train import TrainConfig, run

    cfg = TrainConfig(
        tiles_dir=str(args.tiles_dir), splits_path=str(args.splits), out_dir=str(args.out),
        seeds=args.seeds, epochs=args.epochs, batch_size=args.batch_size, num_points=args.num_points,
        lr=args.lr, rotate=not args.no_rotate, use_density=not args.no_density, shuffle_labels=args.shuffle_labels,
        max_tiles_per_split=args.max_tiles_per_split, num_workers=args.num_workers, device=args.device,
    )
    run(cfg)


def _baseline(args):
    from .baseline import run

    run(args.tiles_dir, args.splits, args.out)


def _card(args):
    from .hub import build_card
    from .splits import load_splits

    args.out.write_text(build_card(args.tiles_dir, load_splits(args.splits)))
    print(f"Wrote {args.out}")


def _publish(args):
    from .hub import publish

    print(publish(args.tiles_dir, args.splits, args.repo_id, private=not args.public))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pcc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download districts and extract vertices")
    p.add_argument("--districts", nargs="*", help="e.g. MN01 BX02 (default: all 59)")
    p.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    p.add_argument("--interim-dir", type=Path, default=config.INTERIM_DIR)
    p.add_argument("--keep-raw", action="store_true", help="keep the downloaded zips")
    p.set_defaults(func=_download)

    p = sub.add_parser("tile", help="cut districts into fixed-size tiles")
    p.add_argument("--interim-dir", type=Path, default=config.INTERIM_DIR)
    p.add_argument("--tiles-dir", type=Path, default=config.TILES_DIR)
    p.add_argument("--tile-size", type=float, default=250.0, help="tile edge in meters")
    p.add_argument("--min-points", type=int, default=256, help="drop tiles with fewer vertices")
    p.add_argument("--max-points", type=int, default=8192, help="subsample tiles with more vertices")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_tile)

    p = sub.add_parser("split", help="assign districts to train/val/test")
    p.add_argument("--tiles-dir", type=Path, default=config.TILES_DIR)
    p.add_argument("--out", type=Path, default=config.SPLITS_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_split)

    p = sub.add_parser("train", help="train the Point Transformer over several seeds")
    p.add_argument("--tiles-dir", type=Path, default=config.TILES_DIR)
    p.add_argument("--splits", type=Path, default=config.SPLITS_PATH)
    p.add_argument("--out", type=Path, default=config.RUNS_DIR / "pct")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-rotate", action="store_true", help="disable random rotation augmentation")
    p.add_argument("--no-density", action="store_true",
                   help="don't give the model the tile's vertex count (the setup of the first full run)")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="sanity check: train on shuffled labels; test scores should drop to chance")
    p.add_argument("--max-tiles-per-split", type=int, help="cap tiles per split, for quick smoke tests")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=_train)

    p = sub.add_parser("baseline", help="majority-class and gradient-boosting baselines")
    p.add_argument("--tiles-dir", type=Path, default=config.TILES_DIR)
    p.add_argument("--splits", type=Path, default=config.SPLITS_PATH)
    p.add_argument("--out", type=Path, default=config.RUNS_DIR / "baseline")
    p.set_defaults(func=_baseline)

    p = sub.add_parser("card", help="render the dataset card locally for review")
    p.add_argument("--tiles-dir", type=Path, default=config.TILES_DIR)
    p.add_argument("--splits", type=Path, default=config.SPLITS_PATH)
    p.add_argument("--out", type=Path, default=config.DATA_DIR / "DATASET_CARD.md")
    p.set_defaults(func=_card)

    p = sub.add_parser("publish", help="upload tiles, splits and dataset card to the Hugging Face Hub")
    p.add_argument("repo_id", help="e.g. your-username/nyc-3d-tiles")
    p.add_argument("--tiles-dir", type=Path, default=config.TILES_DIR)
    p.add_argument("--splits", type=Path, default=config.SPLITS_PATH)
    p.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    p.set_defaults(func=_publish)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
