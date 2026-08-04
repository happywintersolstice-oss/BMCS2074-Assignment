"""
Combine team-labeled review files into one validated master dataset.

What this file does:
- reads multiple labeled CSV files from the team
- keeps only valid labels from the shared taxonomy
- removes empty and duplicate rows
- writes one combined master dataset for future splitting

How it works:
- each CSV is tagged with its source filename
- invalid labels are discarded
- exact duplicate text and label pairs are removed
- the final CSV keeps useful metadata columns when present
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID_LABELS = {
    "payment_failure",
    "account_access_issue",
    "transfer_issue",
    "security_concern",
    "feature_request",
}

DEFAULT_INPUT_DIR = Path("data/labeling_batches")
DEFAULT_OUTPUT_PATH = Path("data/raw/ewallet_reviews_master_labeled.csv")


def parse_arguments() -> argparse.Namespace:
    """Read command-line options for master dataset building."""
    parser = argparse.ArgumentParser(description="Combine labeled review files into one master dataset.")
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing team-labeled CSV files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Destination CSV path for the combined master dataset.",
    )
    return parser.parse_args()


def normalize_label(value: object) -> str:
    """Normalize label spelling to the shared machine-friendly taxonomy."""
    normalized = str(value).strip().lower()
    return normalized.replace(" ", "_")


def load_labeled_files(input_dir: str) -> pd.DataFrame:
    """Read all CSV files from the labeling folder."""
    source_dir = Path(input_dir)
    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {source_dir}")

    dataframes: list[pd.DataFrame] = []
    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)
        dataframe["source_file"] = csv_path.name
        dataframes.append(dataframe)

    return pd.concat(dataframes, ignore_index=True)


def validate_and_clean(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep only valid labeled rows and remove exact duplicates."""
    if "text" not in dataframe.columns or "label" not in dataframe.columns:
        raise ValueError("All labeled files must contain text and label columns.")

    dataframe = dataframe.copy()
    dataframe["text"] = dataframe["text"].fillna("").astype(str).str.strip()
    dataframe["label"] = dataframe["label"].apply(normalize_label)
    dataframe = dataframe[dataframe["text"] != ""].copy()
    dataframe = dataframe[dataframe["label"].isin(VALID_LABELS)].copy()

    preferred_columns = [
        "text",
        "label",
        "app_name",
        "app_id",
        "store_source",
        "rating",
        "review_date",
        "review_id",
        "assigned_to",
        "label_notes",
        "label_status",
        "source_file",
    ]
    existing_columns = [column for column in preferred_columns if column in dataframe.columns]
    dataframe = dataframe[existing_columns].copy()
    dataframe = dataframe.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("No valid labeled rows remained after cleaning.")

    return dataframe


def main() -> None:
    """Build the master labeled dataset."""
    arguments = parse_arguments()
    dataframe = load_labeled_files(arguments.input_dir)
    cleaned_dataframe = validate_and_clean(dataframe)

    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Created: {output_path}")
    print(f"Rows kept: {len(cleaned_dataframe)}")
    print("Rows by label:")
    print(cleaned_dataframe["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
