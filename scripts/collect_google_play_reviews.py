"""
Collect real Google Play reviews for selected e-wallet apps.

What this script does:
- fetches review text from Google Play for one or more target app lists
- saves one raw CSV file for archive and traceability
- saves one label-ready CSV file for your team to annotate manually

How it works:
- each target app is identified by its Google Play app id
- the script can load targets from the built-in list or from a CSV file
- the script requests a configurable number of reviews per app
- raw review metadata is preserved for reporting and cleaning
- a second CSV is created with a blank `label` column for team labeling
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews


@dataclass(frozen=True)
class AppTarget:
    """
    Store the app name and Google Play id for one review source target.
    """

    app_name: str
    app_id: str


DEFAULT_TARGETS = [
    AppTarget("Touch 'n Go eWallet", "my.com.tngdigital.ewallet"),
    AppTarget("Grab", "com.grabtaxi.passenger"),
    AppTarget("Boost", "my.com.myboost"),
    AppTarget("Shopee", "com.shopee.my"),
]

RAW_OUTPUT_PATH = Path("data/raw/ewallet_reviews_collected_raw.csv")
LABELING_OUTPUT_PATH = Path("data/raw/ewallet_reviews_for_labeling.csv")
DEFAULT_TARGETS_FILE = Path("data/raw/google_play_targets.csv")


def parse_arguments() -> argparse.Namespace:
    """
    Read command-line options so the script stays reusable.
    """
    parser = argparse.ArgumentParser(
        description="Collect Google Play reviews for selected e-wallet apps."
    )
    parser.add_argument(
        "--count-per-app",
        type=int,
        default=500,
        help="Maximum number of reviews to request per app.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code for Google Play review collection. Example: en",
    )
    parser.add_argument(
        "--country",
        default="my",
        help="Country code for Google Play review collection. Example: my",
    )
    parser.add_argument(
        "--raw-output",
        default=str(RAW_OUTPUT_PATH),
        help="Path for the raw collected review CSV.",
    )
    parser.add_argument(
        "--labeling-output",
        default=str(LABELING_OUTPUT_PATH),
        help="Path for the label-ready CSV.",
    )
    parser.add_argument(
        "--targets-file",
        default=str(DEFAULT_TARGETS_FILE),
        help="Optional CSV file with app_name and app_id columns.",
    )
    parser.add_argument(
        "--sort",
        choices=["newest", "most_relevant"],
        default="newest",
        help="Google Play review ordering to request.",
    )
    return parser.parse_args()


def load_targets(targets_file: str) -> list[AppTarget]:
    """
    Load app targets from CSV when provided, otherwise use the built-in defaults.
    """
    csv_path = Path(targets_file)
    if not csv_path.exists():
        return DEFAULT_TARGETS

    targets_dataframe = pd.read_csv(csv_path)
    required_columns = {"app_name", "app_id"}
    missing_columns = required_columns.difference(targets_dataframe.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(f"Targets file is missing required columns: {missing_display}")

    targets: list[AppTarget] = []
    for row in targets_dataframe.itertuples(index=False):
        app_name = str(getattr(row, "app_name")).strip()
        app_id = str(getattr(row, "app_id")).strip()
        if app_name and app_id:
            targets.append(AppTarget(app_name=app_name, app_id=app_id))

    if not targets:
        raise ValueError("Targets file did not contain any valid app rows.")

    return targets


def fetch_reviews_for_target(
    target: AppTarget,
    count_per_app: int,
    lang: str,
    country: str,
    sort_mode: str,
) -> list[dict[str, object]]:
    """
    Fetch reviews for one app and convert them into plain row dictionaries.
    """
    selected_sort = Sort.NEWEST if sort_mode == "newest" else Sort.MOST_RELEVANT
    review_items, _ = reviews(
        target.app_id,
        lang=lang,
        country=country,
        sort=selected_sort,
        count=count_per_app,
    )

    rows: list[dict[str, object]] = []
    for item in review_items:
        rows.append(
            {
                "review_id": item.get("reviewId"),
                "text": (item.get("content") or "").strip(),
                "app_name": target.app_name,
                "app_id": target.app_id,
                "store_source": "Google Play",
                "rating": item.get("score"),
                "review_date": item.get("at"),
                "thumbs_up_count": item.get("thumbsUpCount"),
                "review_created_version": item.get("reviewCreatedVersion"),
                "app_version": item.get("appVersion"),
            }
        )
    return rows


def build_raw_dataframe(
    targets: list[AppTarget],
    count_per_app: int,
    lang: str,
    country: str,
    sort_mode: str,
) -> pd.DataFrame:
    """
    Collect reviews from every target app and return one combined DataFrame.
    """
    all_rows: list[dict[str, object]] = []
    for target in targets:
        target_rows = fetch_reviews_for_target(
            target=target,
            count_per_app=count_per_app,
            lang=lang,
            country=country,
            sort_mode=sort_mode,
        )
        all_rows.extend(target_rows)

    dataframe = pd.DataFrame(all_rows)
    if dataframe.empty:
        raise ValueError("No reviews were collected. Check the app ids or network connection.")

    # Remove empty rows and exact duplicate reviews before saving.
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe = dataframe[dataframe["text"] != ""].copy()
    dataframe = dataframe.drop_duplicates(subset=["review_id", "text"]).reset_index(drop=True)
    return dataframe


def build_labeling_dataframe(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simpler CSV that the team can label manually.
    """
    labeling_dataframe = raw_dataframe[
        ["text", "app_name", "app_id", "store_source", "rating", "review_date", "review_id"]
    ].copy()
    labeling_dataframe.insert(1, "label", "")
    labeling_dataframe["label_notes"] = ""
    labeling_dataframe["assigned_to"] = ""
    labeling_dataframe["label_status"] = "unlabeled"
    return labeling_dataframe


def save_dataframe(dataframe: pd.DataFrame, output_path: str) -> None:
    """
    Save one DataFrame to CSV, creating the folder if needed.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False, encoding="utf-8-sig")


def main() -> None:
    """
    Run the full review collection flow.
    """
    arguments = parse_arguments()
    targets = load_targets(arguments.targets_file)
    raw_dataframe = build_raw_dataframe(
        targets=targets,
        count_per_app=arguments.count_per_app,
        lang=arguments.lang,
        country=arguments.country,
        sort_mode=arguments.sort,
    )
    labeling_dataframe = build_labeling_dataframe(raw_dataframe)

    save_dataframe(raw_dataframe, arguments.raw_output)
    save_dataframe(labeling_dataframe, arguments.labeling_output)

    print("Collection complete.")
    print(f"Raw reviews saved to: {arguments.raw_output}")
    print(f"Labeling file saved to: {arguments.labeling_output}")
    print(f"Total collected rows after cleanup: {len(raw_dataframe)}")
    print("Collected rows by app:")
    print(raw_dataframe["app_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
