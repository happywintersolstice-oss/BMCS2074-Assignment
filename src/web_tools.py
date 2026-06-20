"""
Helpers for webpage text extraction.
"""

from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup


def extract_text_from_url(url: str) -> str:
    """
    Fetch a page and extract readable text.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.extract()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError("No readable text was found on the page.")

    return text
