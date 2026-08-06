"""
Find likely security concern reviews from a collected review CSV.

What this file does:
- reads a collected or label-ready review CSV
- scores rows using security-related phrases and tokens
- optionally favors lower-rated reviews
- writes a smaller candidate CSV for manual security labeling

How it works:
- each review text is normalized to lowercase plain text
- security keywords and phrases add to a row score
- rows with higher scores are sorted to the top
- the output keeps the original metadata plus matched keywords
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("data/raw/ewallet_reviews_for_labeling.csv")
DEFAULT_OUTPUT_PATH = Path("data/raw/security_concern_candidates.csv")

SECURITY_PHRASES = [
    "unauthorized access",
    "suspicious activity",
    "suspicious login",
    "unknown device",
    "security issue",
    "security concern",
    "privacy concern",
    "fraud claim",
    "fraudulent transaction",
    "account takeover",
    "not safe",
    "hacked account",
    "my account was hacked",
    "someone accessed my account",
    "unusual activity",
    "stolen account",
]

SECURITY_TOKENS = {
    "security",
    "privacy",
    "fraud",
    "fraudulent",
    "scam",
    "unsafe",
    "hack",
    "hacked",
    "hacker",
    "suspicious",
    "unauthorized",
    "compromised",
    "breach",
    "phishing",
    "paranoid",
    "stolen",
    "protect",
    "protection",
}


def parse_arguments() -> argparse.Namespace:
    """Read command-line options for candidate extraction."""
    parser = argparse.ArgumentParser(description="Find likely security concern reviews in a CSV file.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Source CSV file containing collected or label-ready reviews.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Destination CSV path for security concern candidates.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=2,
        help="Minimum security score required to keep a row.",
    )
    parser.add_argument(
        "--max-rating",
        type=float,
        default=3.0,
        help="Prefer rows with rating at or below this value. Use 5 to disable filtering.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=300,
        help="Maximum number of candidate rows to save after sorting.",
    )
    return parser.parse_args()


def normalize_text(text: object) -> str:
    """Convert review text into a simple searchable format."""
    normalized = str(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def score_review_security(text: object) -> tuple[int, list[str]]:
    """Give a review a security score and return the matched terms."""
    normalized = normalize_text(text)
    tokens = set(normalized.split())

    score = 0
    matched_terms: list[str] = []

    for phrase in SECURITY_PHRASES:
        if phrase in normalized:
            score += 2
            matched_terms.append(phrase)

    for token in SECURITY_TOKENS:
        if token in tokens:
            score += 1
            matched_terms.append(token)

    return score, sorted(set(matched_terms))


def load_reviews(input_path: str) -> pd.DataFrame:
    """Read the source review CSV and validate the required text column."""
    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    dataframe = pd.read_csv(source_path)
    if "text" not in dataframe.columns:
        raise ValueError("Input CSV must contain a text column.")
    return dataframe


def build_candidate_dataframe(
    dataframe: pd.DataFrame,
    min_score: int,
    max_rating: float,
    top: int,
) -> pd.DataFrame:
    """Create the filtered and ranked security candidate queue."""
    results = dataframe["text"].apply(score_review_security)
    candidate_dataframe = dataframe.copy()
    candidate_dataframe["security_score"] = results.map(lambda item: item[0])
    candidate_dataframe["matched_terms"] = results.map(lambda item: ", ".join(item[1]))

    candidate_dataframe = candidate_dataframe[candidate_dataframe["security_score"] >= min_score].copy()

    if "rating" in candidate_dataframe.columns and max_rating < 5:
        numeric_ratings = pd.to_numeric(candidate_dataframe["rating"], errors="coerce")
        candidate_dataframe = candidate_dataframe[numeric_ratings.fillna(max_rating) <= max_rating].copy()

    candidate_dataframe = candidate_dataframe.sort_values(
        by=["security_score", "rating"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    if top > 0:
        candidate_dataframe = candidate_dataframe.head(top).copy()

    if "label" not in candidate_dataframe.columns:
        candidate_dataframe.insert(1, "label", "")
    if "label_notes" not in candidate_dataframe.columns:
        candidate_dataframe["label_notes"] = ""
    if "label_status" not in candidate_dataframe.columns:
        candidate_dataframe["label_status"] = "security_candidate"

    return candidate_dataframe


def main() -> None:
    """Run the security candidate extraction flow."""
    arguments = parse_arguments()
    dataframe = load_reviews(arguments.input)
    candidate_dataframe = build_candidate_dataframe(
        dataframe=dataframe,
        min_score=arguments.min_score,
        max_rating=arguments.max_rating,
        top=arguments.top,
    )

    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate_dataframe.to_csv(destination, index=False, encoding="utf-8-sig")

    print("Security candidate extraction complete.")
    print(f"Input rows: {len(dataframe)}")
    print(f"Candidate rows saved: {len(candidate_dataframe)}")
    print(f"Output file: {destination}")
    if not candidate_dataframe.empty:
        print("Top matched terms:")
        print(candidate_dataframe["matched_terms"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
