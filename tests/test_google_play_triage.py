"""Tests for rating-based live Google Play review triage."""

import unittest

import pandas as pd

from src.ewallet_review_google_play import extract_google_play_app_id, triage_google_play_reviews


class FakeModel:
    """Small predictable model used to test triage without network access."""

    def predict(self, texts: list[str]) -> list[str]:
        return ["payment_failure"] * len(texts)

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.9, 0.1] for _ in texts]


class GooglePlayTriageTests(unittest.TestCase):
    def test_extracts_app_id_from_google_play_url(self) -> None:
        url = "https://play.google.com/store/apps/details?id=my.com.tngdigital.ewallet"
        self.assertEqual(extract_google_play_app_id(url), "my.com.tngdigital.ewallet")

    def test_classifies_only_negative_reviews(self) -> None:
        reviews = pd.DataFrame(
            {"Rating": [1, 3, 5], "Text": ["payment failed", "okay app", "great app"]}
        )
        results, summary = triage_google_play_reviews(FakeModel(), reviews)
        self.assertEqual(results["Predicted Issue"].tolist(), ["Payment Failure", "", ""])
        self.assertEqual(results["Action"].tolist(), ["Classified for follow-up", "Left unchanged", "Left unchanged"])
        self.assertEqual(summary, {"total": 3, "negative": 1, "positive": 1, "neutral": 1})


if __name__ == "__main__":
    unittest.main()
