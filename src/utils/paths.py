"""Centralized path constants for the project."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_EXTRACT = DATA_DIR / "raw_extract"
INTERIM = DATA_DIR / "interim"
PROCESSED = DATA_DIR / "processed"
OUTPUTS = REPO_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"

RAW_ZIPS = REPO_ROOT.parents[1]

VDEM_CSV = RAW_EXTRACT / "V-Dem-CY-Full+Others-v16.csv"
SWIID_SUMMARY_CSV = RAW_EXTRACT / "swiid-9.7" / "data" / "swiid_summary.csv"
SWIID_DRAWS_DTA = RAW_EXTRACT / "swiid-9.7" / "data" / "swiid9_3.dta"
BARRO_LEE_CSV = RAW_ZIPS / "barro-lee-bl2012_MF1599_v2.2.csv"
WID_ZIP = RAW_ZIPS / "wid_all_data.zip"

SEED = 42

for d in (INTERIM, PROCESSED, FIGURES, TABLES):
    d.mkdir(parents=True, exist_ok=True)
