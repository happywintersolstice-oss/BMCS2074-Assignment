"""
Central place for paths used by the project.

Keeping paths here makes the rest of the code easier to read and update.
"""

from pathlib import Path


# BASE_DIR points to the main assignment folder.
BASE_DIR = Path(__file__).resolve().parent.parent

# Common project folders.
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
