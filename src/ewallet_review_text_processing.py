"""
Text cleaning helpers for e-wallet review preprocessing.

What this file does:
- prepares raw review text before it is sent into TF-IDF

How it works:
- converts text to lowercase
- removes punctuation and special characters
- removes extra spaces

This makes similar reviews look more consistent to the models.
"""

import re
import string


def preprocess_review_text(text: str) -> str:
    """
    Clean review text before TF-IDF feature extraction.
    """
    # Apply the same deterministic cleanup during training and prediction.
    # Lowercasing helps the model treat "Payment" and "payment" as the same word.
    text = text.lower()
    # Remove punctuation so the model focuses on words rather than symbols.
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Keep only letters, numbers, and spaces in the final cleaned text.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse repeated spaces created by earlier cleanup steps.
    text = re.sub(r"\s+", " ", text).strip()
    return text
