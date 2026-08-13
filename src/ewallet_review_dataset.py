"""
Dataset loading helpers for the e-wallet review classification app.

What this file does:
- reads the CSV datasets
- finds the correct text and label columns
- cleans and validates the rows
- creates a `clean_text` column for model input
- returns a small summary for the UI

How it works:
- raw labels are normalized into the five official project labels
- invalid, empty, or duplicate rows are removed
- the cleaned text is produced by the preprocessing module
- training data can be balanced, while testing data stays untouched
"""

from io import BytesIO

import pandas as pd

from src.ewallet_review_constants import (
    BALANCE_CLASSES_FOR_TRAINING,
    DATA_PATH,
    LABEL_COLUMN_CANDIDATES,
    LABEL_NAMES,
    MAX_TRAIN_TEXT_LENGTH,
    TESTING_DATA_PATH,
    TEXT_COLUMN_CANDIDATES,
)
from src.ewallet_review_text_processing import preprocess_review_text


LABEL_MAPPING = {
    # This mapping lets the app accept small label wording differences in a CSV file.
    "payment_failure": "payment_failure",
    "payment failure": "payment_failure",
    "failed payment": "payment_failure",
    "transaction failed": "payment_failure",
    "declined payment": "payment_failure",
    "account_access_issue": "account_access_issue",
    "account access issue": "account_access_issue",
    "login issue": "account_access_issue",
    "verification issue": "account_access_issue",
    "otp issue": "account_access_issue",
    "transfer_issue": "transfer_issue",
    "transfer issue": "transfer_issue",
    "bank transfer issue": "transfer_issue",
    "money transfer issue": "transfer_issue",
    "security_concern": "security_concern",
    "security concern": "security_concern",
    "security issue": "security_concern",
    "privacy concern": "security_concern",
    "feature_request": "feature_request",
    "feature request": "feature_request",
    "request": "feature_request",
}


def normalize_review_label(value: object) -> str | None:
    """
    Convert different label formats into the shared e-wallet label set.
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
    # Compare column names in lowercase so matching is more forgiving.
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


def read_review_dataset_file(dataset_path) -> pd.DataFrame:
    """
    Read the dataset with a small set of encoding fallbacks.
    """
    # Try a few common encodings because CSV files often vary by source.
    read_attempts = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"encoding": "latin-1"},
    ]
    last_error: Exception | None = None

    for options in read_attempts:
        try:
            return pd.read_csv(dataset_path, **options)
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(f"Could not read dataset with supported encodings: {last_error}")


def load_uploaded_review_file(file_bytes: bytes) -> pd.DataFrame:
    """Load a user-uploaded CSV containing a supported review-text column."""
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    read_attempts = ["utf-8", "utf-8-sig", "latin-1"]
    dataframe: pd.DataFrame | None = None
    last_error: Exception | None = None
    for encoding in read_attempts:
        try:
            dataframe = pd.read_csv(BytesIO(file_bytes), encoding=encoding)
            break
        except UnicodeDecodeError as error:
            last_error = error
        except pd.errors.EmptyDataError as error:
            raise ValueError("The uploaded CSV contains no rows.") from error

    if dataframe is None:
        raise ValueError(f"Could not read uploaded CSV with supported encodings: {last_error}")

    text_column = detect_column_name(dataframe.columns.tolist(), TEXT_COLUMN_CANDIDATES, "text")
    dataframe = dataframe[[text_column]].rename(columns={text_column: "text"}).copy()
    dataframe["text"] = dataframe["text"].fillna("").astype(str).str.strip().str.slice(0, MAX_TRAIN_TEXT_LENGTH)
    dataframe = dataframe[dataframe["text"] != ""].copy()
    dataframe["clean_text"] = dataframe["text"].apply(preprocess_review_text)
    dataframe = dataframe[dataframe["clean_text"] != ""].reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("The uploaded CSV has no usable review text after cleaning.")

    return dataframe


def load_ewallet_review_dataset(
    apply_balancing: bool | None = None,
    excluded_clean_texts: set[str] | None = None,
) -> pd.DataFrame:
    """
    Load the training dataset and add a cleaned review text column.
    """
    dataframe, _ = load_ewallet_review_dataset_with_summary(
        apply_balancing=apply_balancing,
        excluded_clean_texts=excluded_clean_texts,
    )
    return dataframe


def load_testing_review_dataset() -> pd.DataFrame:
    """
    Load the held-out testing dataset without balancing it.
    """
    dataframe, _ = load_review_dataset_with_summary(
        dataset_path=TESTING_DATA_PATH,
        apply_balancing=False,
    )
    return dataframe


def load_ewallet_review_dataset_with_summary(
    apply_balancing: bool | None = None,
    excluded_clean_texts: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float | dict[str, int]]]:
    """
    Load the training dataset, clean it, and return a small summary of the cleanup steps.
    """
    if apply_balancing is None:
        apply_balancing = BALANCE_CLASSES_FOR_TRAINING

    return load_review_dataset_with_summary(
        dataset_path=DATA_PATH,
        apply_balancing=apply_balancing,
        excluded_clean_texts=excluded_clean_texts,
    )


def load_review_dataset_with_summary(
    dataset_path,
    apply_balancing: bool,
    excluded_clean_texts: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float | dict[str, int]]]:
    """
    Load any project dataset, clean it, and optionally balance it.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataframe = read_review_dataset_file(dataset_path)
    original_rows = len(dataframe)
    column_names = dataframe.columns.tolist()
    text_column = detect_column_name(column_names, TEXT_COLUMN_CANDIDATES, "text")
    label_column = detect_column_name(column_names, LABEL_COLUMN_CANDIDATES, "label")

    # Keep only the columns used by the training pipeline.
    dataframe = dataframe[[label_column, text_column]].copy()
    dataframe = dataframe.rename(columns={label_column: "label", text_column: "text"})
    dataframe = dataframe.dropna(subset=["label", "text"])
    rows_after_null_drop = len(dataframe)

    # Convert raw label wording into the one shared label scheme for the project.
    dataframe["label"] = dataframe["label"].apply(normalize_review_label)
    dataframe = dataframe.dropna(subset=["label"])
    rows_after_label_clean = len(dataframe)

    # Trim very long reviews so they stay manageable for the prototype.
    dataframe["text"] = dataframe["text"].astype(str).str.strip().str.slice(0, MAX_TRAIN_TEXT_LENGTH)
    dataframe = dataframe[dataframe["text"] != ""].copy()
    rows_after_text_clean = len(dataframe)

    # Remove exact duplicates so repeated rows do not inflate the model scores.
    dataframe = dataframe.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    rows_after_dedup = len(dataframe)

    # Build the cleaned text version that the models will actually learn from.
    dataframe["clean_text"] = dataframe["text"].apply(preprocess_review_text)
    dataframe = dataframe[dataframe["clean_text"] != ""].copy()
    rows_before_overlap_removal = len(dataframe)

    # Exclude test-set text before balancing so it cannot leak into model training.
    if excluded_clean_texts:
        dataframe = dataframe[~dataframe["clean_text"].isin(excluded_clean_texts)].copy()
    overlap_rows_removed = rows_before_overlap_removal - len(dataframe)
    final_rows = len(dataframe)

    if len(dataframe) < 20:
        raise ValueError("Dataset is too small after cleaning. Provide at least 20 valid labeled rows.")

    # Count how many rows belong to each issue category after cleaning.
    class_counts = dataframe["label"].value_counts()
    missing_labels = [label for label in LABEL_NAMES if label not in class_counts.index]
    if missing_labels:
        missing_display = ", ".join(missing_labels)
        raise ValueError(
            "Dataset must contain all supported e-wallet issue labels after label normalization. "
            f"Missing labels: {missing_display}"
        )

    if (class_counts < 2).any():
        raise ValueError("Each class must have at least 2 rows after cleaning.")

    pre_balance_counts = {label: int(class_counts.get(label, 0)) for label in LABEL_NAMES}
    minority_count = int(class_counts.min())
    majority_count = int(class_counts.max())
    imbalance_ratio = round(majority_count / minority_count, 2) if minority_count else 0.0

    balanced_rows = len(dataframe)
    balanced_counts = pre_balance_counts.copy()
    applied_balancing = False
    if apply_balancing:
        target_count = minority_count
        balanced_parts: list[pd.DataFrame] = []
        for label in LABEL_NAMES:
            label_rows = dataframe[dataframe["label"] == label]
            balanced_parts.append(label_rows.sample(n=target_count, random_state=42))

        dataframe = (
            pd.concat(balanced_parts, ignore_index=True)
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        balanced_rows = len(dataframe)
        balanced_class_counts = dataframe["label"].value_counts()
        balanced_counts = {label: int(balanced_class_counts.get(label, 0)) for label in LABEL_NAMES}
        applied_balancing = True

    # This summary is shown in the sidebar so users can inspect dataset health.
    summary = {
        "original_rows": original_rows,
        "rows_after_null_drop": rows_after_null_drop,
        "rows_after_label_clean": rows_after_label_clean,
        "rows_after_text_clean": rows_after_text_clean,
        "rows_after_dedup": rows_after_dedup,
        "final_rows": final_rows,
        "balanced_rows": balanced_rows,
        "duplicates_removed": rows_after_text_clean - rows_after_dedup,
        "overlap_rows_removed": overlap_rows_removed,
        "label_counts_before_balance": pre_balance_counts,
        "label_counts": balanced_counts,
        "applied_balancing": applied_balancing,
        "minority_count": minority_count,
        "majority_count": majority_count,
        "imbalance_ratio": imbalance_ratio,
    }

    return dataframe, summary
