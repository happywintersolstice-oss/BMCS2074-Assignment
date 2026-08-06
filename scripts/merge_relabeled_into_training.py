"""
Merge high-confidence relabeled rows into the training dataset.

What this file does:
- reads one or more relabeled CSV files
- keeps only rows with supported labels
- aligns their columns with the training dataset
- appends only new non-duplicate rows into training

How it works:
- duplicate checking uses the pair of `text` and `label`
- rows missing required values are ignored
- the script prints how many rows were added per label
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Add the project root so this script can import from `src` when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ewallet_review_constants import LABEL_NAMES, TRAINING_DATA_PATH


def parse_arguments() -> argparse.Namespace:
    """Read command-line inputs for training merge."""
    parser = argparse.ArgumentParser(description="Merge relabeled rows into the training dataset.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more relabeled CSV files to merge into training.",
    )
    parser.add_argument(
        "--training",
        default=str(TRAINING_DATA_PATH),
        help="Training CSV file to update.",
    )
    return parser.parse_args()


def load_training_dataframe(training_path: str) -> pd.DataFrame:
    """Load the current training dataset."""
    path = Path(training_path)
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    return pd.read_csv(path)


def load_relabeled_rows(input_paths: list[str]) -> pd.DataFrame:
    """Load and combine relabeled files."""
    dataframes: list[pd.DataFrame] = []
    for input_path in input_paths:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Relabeled file not found: {path}")
        dataframe = pd.read_csv(path)
        dataframe["merge_source_file"] = path.name
        dataframes.append(dataframe)

    combined = pd.concat(dataframes, ignore_index=True)
    if "text" not in combined.columns or "label" not in combined.columns:
        raise ValueError("Every relabeled input must contain text and label columns.")

    combined["text"] = combined["text"].fillna("").astype(str).str.strip()
    combined["label"] = combined["label"].fillna("").astype(str).str.strip()
    combined = combined[(combined["text"] != "") & (combined["label"].isin(LABEL_NAMES))].copy()
    combined = combined.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return combined


def merge_rows(training_df: pd.DataFrame, relabeled_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append only unseen text-label pairs into training."""
    existing_pairs = set(
        zip(
            training_df["text"].fillna("").astype(str).str.strip(),
            training_df["label"].fillna("").astype(str).str.strip(),
        )
    )

    is_new = [pair not in existing_pairs for pair in zip(relabeled_df["text"], relabeled_df["label"])]
    new_rows = relabeled_df[is_new].copy()
    if new_rows.empty:
        return training_df, new_rows

    for required_column in training_df.columns:
        if required_column not in new_rows.columns:
            new_rows[required_column] = ""

    new_rows = new_rows[training_df.columns].copy()
    merged_df = pd.concat([training_df, new_rows], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return merged_df, new_rows


def main() -> None:
    """Merge relabeled rows into the training CSV."""
    arguments = parse_arguments()
    training_df = load_training_dataframe(arguments.training)
    relabeled_df = load_relabeled_rows(arguments.inputs)
    merged_df, new_rows = merge_rows(training_df, relabeled_df)

    output_path = Path(arguments.training)
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Training rows before merge: {len(training_df)}")
    print(f"Candidate relabeled rows loaded: {len(relabeled_df)}")
    print(f"New rows added: {len(new_rows)}")
    print(f"Training rows after merge: {len(merged_df)}")
    if not new_rows.empty:
        print("Added rows by label:")
        print(new_rows["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
