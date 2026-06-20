"""
Text cleaning helpers for preprocessing.
"""

import re
import string


def preprocess_text(text: str) -> str:
    """
    Clean text before TF-IDF feature extraction.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
