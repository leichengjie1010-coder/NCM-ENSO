"""Input locations and the selected final model's training/replay settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = Path("path/to/processed_fields.nc")
FORECAST_FILE = None
OUTPUT_DIR = ROOT / "results"
RECIPE_DIR = ROOT / "training_recipe"

FINAL_MODEL_SHA256 = "05d21a64041969aa95632bfa8746a4d30603ee35786c76bd726ab83c598824d3"
DATA_ROOT = Path("/Volumes/Data/datas")
DATA_VOLUME_UUID = "192DB495-FD02-48EA-AE9B-538A4737217B"

FORECAST_ORIGIN = "2026-09"
