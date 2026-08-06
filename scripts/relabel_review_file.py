"""
Apply the shared strict relabel rules to any review CSV file.

What this file does:
- reads a CSV file with a `text` column
- applies the existing strict relabel rules
- saves high-confidence rows and manual-review rows separately

How it works:
- it reuses the same `relabel_row` function as the main relabel script
- output files are passed in as command-line arguments
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.relabel_ewallet_review_dataset import relabel_row


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Relabel a review CSV with the shared strict rules.")
    parser.add_argument("--input", required=True, help="Input CSV file with a text column.")
    parser.add_argument("--output", required=True, help="Output CSV for high-confidence relabeled rows.")
    parser.add_argument("--manual-output", required=True, help="Output CSV for manual-review rows.")
    return parser.parse_args()


def main() -> None:
    """Run the relabel flow for an arbitrary review file."""
    arguments = parse_arguments()
    input_path = Path(arguments.input)
    output_path = Path(arguments.output)
    manual_output_path = Path(arguments.manual_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataframe = pd.read_csv(input_path)
    if "text" not in dataframe.columns:
        raise ValueError("Input CSV must contain a text column.")

    relabel_results = dataframe["text"].apply(relabel_row)
    dataframe["suggested_label"] = relabel_results.map(lambda item: item[0])
    dataframe["rule_score"] = relabel_results.map(lambda item: item[1])
    dataframe["review_status"] = relabel_results.map(lambda item: item[2])

    relabeled = dataframe[dataframe["suggested_label"].notna()].copy()
    relabeled["label"] = relabeled["suggested_label"]
    relabeled["label_method"] = "strict_rule_relabel"

    manual = dataframe[dataframe["suggested_label"].isna()].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manual_output_path.parent.mkdir(parents=True, exist_ok=True)
    relabeled.to_csv(output_path, index=False, encoding="utf-8-sig")
    manual.to_csv(manual_output_path, index=False, encoding="utf-8-sig")

    print(f"Input rows: {len(dataframe)}")
    print(f"Relabeled rows: {len(relabeled)}")
    if not relabeled.empty:
        print("Relabeled counts:")
        print(relabeled["label"].value_counts().to_string())
    print(f"Manual review rows: {len(manual)}")


if __name__ == "__main__":
    main()
