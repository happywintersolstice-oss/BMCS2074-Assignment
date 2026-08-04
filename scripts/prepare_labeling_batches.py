"""
Create manual labeling batches from a collected review CSV.

What this file does:
- reads a label-ready review CSV
- removes rows that already have labels
- splits the remaining rows into balanced team batches
- writes one CSV per team member for easier manual labeling

How it works:
- the script shuffles rows with a fixed random seed
- it distributes rows round-robin across members
- each output file keeps the original metadata for traceability
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/raw/ewallet_reviews_for_labeling.csv")
DEFAULT_OUTPUT_DIR = Path("data/labeling_batches")
DEFAULT_MEMBERS = ["member_1", "member_2", "member_3"]
RANDOM_STATE = 42


def parse_arguments() -> argparse.Namespace:
    """Read command-line options for batch generation."""
    parser = argparse.ArgumentParser(description="Split review rows into manual labeling batches.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the label-ready review CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where member batch CSV files will be written.",
    )
    parser.add_argument(
        "--members",
        nargs="+",
        default=DEFAULT_MEMBERS,
        help="Member names to assign labeling batches to.",
    )
    return parser.parse_args()


def load_unlabeled_reviews(input_path: str) -> pd.DataFrame:
    """Load only rows that still need manual labeling."""
    dataframe = pd.read_csv(input_path)
    if "text" not in dataframe.columns:
        raise ValueError("Input CSV must contain a text column.")

    if "label" not in dataframe.columns:
        dataframe["label"] = ""

    dataframe["label"] = dataframe["label"].fillna("").astype(str).str.strip()
    dataframe = dataframe[dataframe["label"] == ""].copy()
    if dataframe.empty:
        raise ValueError("No unlabeled rows were found in the input CSV.")

    return dataframe.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def assign_batches(dataframe: pd.DataFrame, members: list[str]) -> dict[str, pd.DataFrame]:
    """Distribute rows evenly across the given member names."""
    batches: dict[str, list[dict[str, object]]] = {member: [] for member in members}

    for index, row in enumerate(dataframe.to_dict(orient="records")):
        member = members[index % len(members)]
        row["assigned_to"] = member
        row["label_status"] = "pending"
        batches[member].append(row)

    return {member: pd.DataFrame(rows) for member, rows in batches.items()}


def save_batches(batches: dict[str, pd.DataFrame], output_dir: str) -> None:
    """Write one CSV file per member."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    for member, dataframe in batches.items():
        output_path = destination / f"{member}_labeling_batch.csv"
        dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Created: {output_path}")


def main() -> None:
    """Run the batch preparation flow."""
    arguments = parse_arguments()
    dataframe = load_unlabeled_reviews(arguments.input)
    batches = assign_batches(dataframe, arguments.members)
    save_batches(batches, arguments.output_dir)

    print()
    print(f"Total unlabeled rows distributed: {len(dataframe)}")
    for member, member_dataframe in batches.items():
        print(f"- {member}: {len(member_dataframe)}")


if __name__ == "__main__":
    main()
