"""Download NYC 3D model districts and convert each to an .npz of vertices.

Districts are processed one at a time (download, unzip, extract, delete) so the
full ~8.6 GB of zips never needs to be on disk at once. Extraction runs in a
fresh subprocess per file because rhino3dm does not release memory between
reads.
"""

import multiprocessing as mp
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from . import config
from .extract import extract_to_file


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def _extract_worker(path_3dm: str, out_path: str, queue) -> None:
    try:
        queue.put(("ok", extract_to_file(Path(path_3dm), Path(out_path))))
    except Exception as e:  # reported back to the parent
        queue.put(("error", repr(e)))


def _extract_in_subprocess(path_3dm: Path, out_path: Path) -> int:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_extract_worker, args=(str(path_3dm), str(out_path), queue))
    proc.start()
    status, value = queue.get()
    proc.join()
    if status != "ok":
        raise RuntimeError(f"Extraction failed for {path_3dm.name}: {value}")
    return value


def process_district(district: str, raw_dir: Path, interim_dir: Path, keep_raw: bool = False) -> int | None:
    """Download and extract one district. Returns its point count, or None if unavailable."""
    out_path = interim_dir / f"{district}.npz"
    if out_path.exists():
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"nyc_3dmodel_{district.lower()}.zip"
    work_dir = raw_dir / district

    if not zip_path.exists():
        try:
            _download(config.DOWNLOAD_URL.format(code=district.lower()), zip_path)
        except requests.HTTPError as e:
            tqdm.write(f"! {district}: download failed ({e.response.status_code}), skipping")
            return None

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)
    path_3dm = next(work_dir.rglob("*.3dm"))

    n_points = _extract_in_subprocess(path_3dm, out_path)

    shutil.rmtree(work_dir)
    if not keep_raw:
        zip_path.unlink()
    return n_points


def download_all(districts: list[str], raw_dir: Path, interim_dir: Path, keep_raw: bool = False) -> None:
    for district in tqdm(districts, desc="districts"):
        if (interim_dir / f"{district}.npz").exists():
            tqdm.write(f"  {district}: already extracted")
            continue
        n_points = process_district(district, raw_dir, interim_dir, keep_raw)
        if n_points is not None:
            tqdm.write(f"  {district}: {n_points:,} unique vertices")
