"""Google Play review collection and live operational triage helpers."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pandas as pd
from google_play_scraper import Sort, reviews
from langdetect import DetectorFactory, LangDetectException, detect

from src.ewallet_review_constants import LABEL_DISPLAY_NAMES
from src.ewallet_review_text_processing import preprocess_review_text


# Keep language detection repeatable across runs for the same review text.
DetectorFactory.seed = 0


def is_english_review(text: str) -> bool:
    """Return whether language detection identifies the review as English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def extract_google_play_app_id(app_reference: str) -> str:
    """Accept either a Google Play URL or an Android package id."""
    reference = app_reference.strip()
    if not reference:
        raise ValueError("Enter a Google Play URL or Android package id.")
    app_id = parse_qs(urlparse(reference).query).get("id", [""])[0].strip() if "play.google.com" in reference else reference
    if not app_id or " " in app_id or "." not in app_id:
        raise ValueError("Enter a valid Google Play URL or package id, such as my.com.tngdigital.ewallet.")
    return app_id


def collect_google_play_reviews(app_id: str, count: int, language: str, country: str) -> pd.DataFrame:
    """Collect and retain only English Google Play reviews for one application."""
    review_items, _ = reviews(app_id, lang=language, country=country, sort=Sort.NEWEST, count=count)
    dataframe = pd.DataFrame(
        {
            "Review ID": [item.get("reviewId") for item in review_items],
            "Review Date": [item.get("at") for item in review_items],
            "Rating": [item.get("score") for item in review_items],
            "Text": [(item.get("content") or "").strip() for item in review_items],
            "Thumbs Up": [item.get("thumbsUpCount") for item in review_items],
        }
    )
    if dataframe.empty:
        raise ValueError("No reviews were returned. Check the app id, country, or network connection.")
    dataframe = dataframe[dataframe["Text"] != ""].drop_duplicates(subset=["Review ID", "Text"])
    dataframe = dataframe[dataframe["Text"].map(is_english_review)]
    if dataframe.empty:
        raise ValueError("No usable English review text was returned.")
    return dataframe.reset_index(drop=True)


def triage_google_play_reviews(model, reviews_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Classify only negative reviews; positive and neutral reviews remain untouched."""
    results = reviews_dataframe.copy()
    results["Sentiment"] = results["Rating"].map(lambda rating: "Negative" if rating <= 2 else "Positive" if rating >= 4 else "Neutral")
    results["Action"] = results["Sentiment"].map({"Negative": "Classified for follow-up", "Positive": "Left unchanged", "Neutral": "Left unchanged"})
    results["Predicted Issue"] = ""
    results["Confidence"] = None
    negative_rows = results["Sentiment"] == "Negative"
    if negative_rows.any():
        cleaned_texts = results.loc[negative_rows, "Text"].map(preprocess_review_text).tolist()
        predictions = list(model.predict(cleaned_texts))
        results.loc[negative_rows, "Predicted Issue"] = [LABEL_DISPLAY_NAMES.get(label, str(label).replace("_", " ").title()) for label in predictions]
        if hasattr(model, "predict_proba"):
            results.loc[negative_rows, "Confidence"] = [float(max(row)) for row in model.predict_proba(cleaned_texts)]
    summary = {"total": int(len(results)), "negative": int(negative_rows.sum()), "positive": int((results["Sentiment"] == "Positive").sum()), "neutral": int((results["Sentiment"] == "Neutral").sum())}
    return results, summary
