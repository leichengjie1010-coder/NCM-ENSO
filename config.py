"""Input locations and run settings. Replace DATA_FILE before fitting."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = Path("path/to/processed_fields.nc")
FORECAST_FILE = None
OUTPUT_DIR = ROOT / "results"

FORECAST_ORIGIN = "2026-08"
