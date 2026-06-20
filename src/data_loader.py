"""
Dataset loading helpers.
"""

import pandas as pd

from src.constants import DATA_PATH
from src.text_processing import preprocess_text


def load_dataset() -> pd.DataFrame:
    """
    Load the local dataset and add a cleaned text column.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)
    required_columns = {"label", "text"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    dataframe["clean_text"] = dataframe["text"].astype(str).apply(preprocess_text)
    return dataframe
