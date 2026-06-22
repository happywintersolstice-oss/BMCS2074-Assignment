"""
Dataset loading helpers.
"""

from __future__ import annotations

import pandas as pd

from src.constants import (
    DATA_PATH,
    LABEL_COLUMN_CANDIDATES,
    MAX_TRAIN_TEXT_LENGTH,
    TEXT_COLUMN_CANDIDATES,
)
from src.text_processing import preprocess_text


LABEL_MAPPING = {
    "spam": "spam",
    "scam": "spam",
    "phishing": "spam",
    "fraud": "spam",
    "malicious": "spam",
    "junk": "spam",
    "junk mail": "spam",
    "unsolicited": "spam",
    "unsolicited message": "spam",
    "1": "spam",
    "ham": "legitimate",
    "legitimate": "legitimate",
    "legit": "legitimate",
    "benign": "legitimate",
    "not spam": "legitimate",
    "non-spam": "legitimate",
    "safe": "legitimate",
    "normal": "legitimate",
    "0": "legitimate",
}


def normalize_label(value: object) -> str | None:
    """
    Convert different label formats into the shared app label set.
    """
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if not normalized:
        return None

    return LABEL_MAPPING.get(normalized)


def detect_column_name(columns: list[str], candidates: list[str], column_role: str) -> str:
    """
    Find a usable dataset column from a list of accepted candidate names.
    """
    lowered_to_original = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered_to_original:
            return lowered_to_original[candidate]

    accepted = ", ".join(candidates)
    available = ", ".join(columns)
    raise ValueError(
        f"Dataset is missing a supported {column_role} column. "
        f"Accepted names: {accepted}. Available columns: {available}"
    )


def read_dataset_file() -> pd.DataFrame:
    """
    Read the dataset with a small set of encoding fallbacks.
    """
    read_attempts = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1"},
    ]
    last_error: Exception | None = None

    for options in read_attempts:
        try:
            return pd.read_csv(DATA_PATH, **options)
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(f"Could not read dataset with supported encodings: {last_error}")


def load_dataset() -> pd.DataFrame:
    """
    Load the local dataset and add a cleaned text column.
    """
    dataframe, _ = load_dataset_with_summary()
    return dataframe


def load_dataset_with_summary() -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Load the local dataset, clean it, and return a small summary of the cleanup steps.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataframe = read_dataset_file()
    original_rows = len(dataframe)
    column_names = dataframe.columns.tolist()
    text_column = detect_column_name(column_names, TEXT_COLUMN_CANDIDATES, "text")
    label_column = detect_column_name(column_names, LABEL_COLUMN_CANDIDATES, "label")

    # Keep only the columns the app actually uses and work on a copy.
    dataframe = dataframe[[label_column, text_column]].copy()
    dataframe = dataframe.rename(columns={label_column: "label", text_column: "text"})

    # Drop rows that do not have both label and text.
    dataframe = dataframe.dropna(subset=["label", "text"])
    rows_after_null_drop = len(dataframe)

    # Normalize labels into the shared scheme used by the app.
    dataframe["label"] = dataframe["label"].apply(normalize_label)
    dataframe = dataframe.dropna(subset=["label"])
    rows_after_label_clean = len(dataframe)

    # Clean and trim the raw text before feature preprocessing.
    dataframe["text"] = dataframe["text"].astype(str).str.strip().str.slice(0, MAX_TRAIN_TEXT_LENGTH)
    dataframe = dataframe[dataframe["text"] != ""].copy()
    rows_after_text_clean = len(dataframe)

    # Remove exact duplicate message rows so metrics are less inflated.
    dataframe = dataframe.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    rows_after_dedup = len(dataframe)

    dataframe["clean_text"] = dataframe["text"].apply(preprocess_text)
    dataframe = dataframe[dataframe["clean_text"] != ""].copy()
    final_rows = len(dataframe)

    if len(dataframe) < 20:
        raise ValueError("Dataset is too small after cleaning. Provide at least 20 valid labeled rows.")

    class_counts = dataframe["label"].value_counts()
    if set(class_counts.index) != {"spam", "legitimate"}:
        raise ValueError("Dataset must contain both 'spam' and 'legitimate' classes after label normalization.")

    if (class_counts < 2).any():
        raise ValueError("Each class must have at least 2 rows after cleaning.")

    minority_count = int(class_counts.min())
    majority_count = int(class_counts.max())
    imbalance_ratio = round(majority_count / minority_count, 2) if minority_count else 0.0

    summary = {
        "original_rows": original_rows,
        "rows_after_null_drop": rows_after_null_drop,
        "rows_after_label_clean": rows_after_label_clean,
        "rows_after_text_clean": rows_after_text_clean,
        "rows_after_dedup": rows_after_dedup,
        "final_rows": final_rows,
        "duplicates_removed": rows_after_text_clean - rows_after_dedup,
        "spam_count": int(class_counts.get("spam", 0)),
        "legitimate_count": int(class_counts.get("legitimate", 0)),
        "minority_count": minority_count,
        "majority_count": majority_count,
        "imbalance_ratio": imbalance_ratio,
    }

    return dataframe, summary
