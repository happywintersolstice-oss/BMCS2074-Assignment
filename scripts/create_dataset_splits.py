"""
Create separate training and testing datasets from the current
e-wallet review master dataset.

What this file does:
- reads the current cleaned dataset
- creates two non-overlapping CSV files
- keeps label distribution roughly consistent across the splits
- writes a small summary file so the team can cite the split sizes later

How it works:
- the script starts from the shared master dataset path in `src/ewallet_review_constants.py`
- it creates a stable `review_id` for each row
- it uses stratified sampling so both splits keep all issue categories
- it saves the outputs into `data/splits/`
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ewallet_review_constants import MASTER_DATA_PATH, TESTING_DATA_PATH, TRAINING_DATA_PATH


OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

SOURCE_DATA_PATH = MASTER_DATA_PATH
TRAINING_OUTPUT_PATH = TRAINING_DATA_PATH
TESTING_OUTPUT_PATH = TESTING_DATA_PATH
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "dataset_split_summary.md"

TESTING_ROWS = 800
RANDOM_STATE = 42


def load_source_dataset() -> pd.DataFrame:
    """Read the current master dataset and validate the required columns."""
    if not SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_DATA_PATH}")

    dataframe = pd.read_csv(SOURCE_DATA_PATH)
    required_columns = {"text", "label"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(f"Source dataset is missing required columns: {missing_display}")

    dataframe = dataframe.dropna(subset=["text", "label"]).copy()
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe["label"] = dataframe["label"].astype(str).str.strip()
    dataframe = dataframe[dataframe["text"] != ""].reset_index(drop=True)

    if len(dataframe) <= TESTING_ROWS:
        raise ValueError(
            "Source dataset is too small for the requested split sizes. "
            f"Need more than {TESTING_ROWS} rows."
        )

    dataframe.insert(0, "review_id", [f"ewallet_{index:05d}" for index in range(1, len(dataframe) + 1)])
    return dataframe


def make_split(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create non-overlapping training and testing splits."""
    training_rows, testing_rows = train_test_split(
        dataframe,
        test_size=TESTING_ROWS,
        stratify=dataframe["label"],
        random_state=RANDOM_STATE,
    )

    training_rows = training_rows.copy()
    testing_rows = testing_rows.copy()

    training_rows["dataset_role"] = "training"
    testing_rows["dataset_role"] = "testing_manual"

    return (
        training_rows.sort_values("review_id").reset_index(drop=True),
        testing_rows.sort_values("review_id").reset_index(drop=True),
    )


def write_csv_files(training_rows: pd.DataFrame, testing_rows: pd.DataFrame) -> None:
    """Save all split datasets into the output folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_rows.to_csv(TRAINING_OUTPUT_PATH, index=False)
    testing_rows.to_csv(TESTING_OUTPUT_PATH, index=False)


def build_label_table(dataframe: pd.DataFrame) -> str:
    """Render a small markdown table of label counts for one split."""
    counts = dataframe["label"].value_counts().sort_index()
    header = "| Label | Rows |\n|---|---:|"
    rows = [f"| `{label}` | {count} |" for label, count in counts.items()]
    return "\n".join([header, *rows])


def write_summary_file(
    source_rows: pd.DataFrame,
    training_rows: pd.DataFrame,
    testing_rows: pd.DataFrame,
) -> None:
    """Write a markdown summary that explains the dataset split outputs."""
    summary_lines = [
        "# Dataset Split Summary",
        "",
        "This file was generated from `data/raw/ewallet_reviews_final.csv`.",
        "",
        "## Split Sizes",
        "",
        f"- Source rows: {len(source_rows)}",
        f"- Training rows: {len(training_rows)}",
        f"- Testing rows: {len(testing_rows)}",
        "",
        "## Training Label Counts",
        "",
        build_label_table(training_rows),
        "",
        "## Testing Label Counts",
        "",
        build_label_table(testing_rows),
        "",
        "## Intended Usage",
        "",
        "- `ewallet_reviews_training.csv`: use for model development and training.",
        "- `ewallet_reviews_testing_manual.csv`: keep as a held-out evaluation set for report results.",
    ]

    SUMMARY_OUTPUT_PATH.write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    """Generate all split files and print a short console summary."""
    source_rows = load_source_dataset()
    training_rows, testing_rows = make_split(source_rows)
    write_csv_files(training_rows, testing_rows)
    write_summary_file(source_rows, training_rows, testing_rows)

    print(f"Created: {TRAINING_OUTPUT_PATH}")
    print(f"Created: {TESTING_OUTPUT_PATH}")
    print(f"Created: {SUMMARY_OUTPUT_PATH}")
    print()
    print("Row counts")
    print(f"- source: {len(source_rows)}")
    print(f"- training: {len(training_rows)}")
    print(f"- testing: {len(testing_rows)}")


if __name__ == "__main__":
    main()
