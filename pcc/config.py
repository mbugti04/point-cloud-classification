"""Shared constants and default paths."""

from pathlib import Path

BOROUGHS = ["MN", "BX", "BK", "QN", "SI"]
BOROUGH_NAMES = {
    "MN": "Manhattan",
    "BX": "Bronx",
    "BK": "Brooklyn",
    "QN": "Queens",
    "SI": "Staten Island",
}

# The 59 community districts. Park/airport files on the NYC site are skipped.
DISTRICT_COUNTS = {"MN": 12, "BX": 12, "BK": 18, "QN": 14, "SI": 3}
DISTRICTS = [f"{b}{i:02d}" for b, n in DISTRICT_COUNTS.items() for i in range(1, n + 1)]

DOWNLOAD_URL = (
    "https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/"
    "nyc-3d-model/nyc_3dmodel_{code}.zip"
)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
TILES_DIR = DATA_DIR / "tiles"
SPLITS_PATH = DATA_DIR / "splits.json"
RUNS_DIR = Path("runs")


def borough_of(district: str) -> str:
    return district[:2].upper()


def label_of(district: str) -> int:
    return BOROUGHS.index(borough_of(district))
