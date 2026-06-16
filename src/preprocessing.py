"""
Basic text preprocessing helpers.

This file will later contain the steps that clean your text before training.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean a piece of text using simple rules.

    What this function does:
    - converts text to lowercase
    - removes non-letter characters except spaces
    - removes extra spaces
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    sample = "This FOOD is Amazing!!! 10/10"
    print(clean_text(sample))
