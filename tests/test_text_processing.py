"""Regression checks for the shared review-text preprocessing function."""

import unittest

from src.ewallet_review_dataset import load_uploaded_review_file
from src.ewallet_review_text_processing import preprocess_review_text


class PreprocessReviewTextTests(unittest.TestCase):
    def test_normalizes_case_punctuation_and_spacing(self) -> None:
        self.assertEqual(
            preprocess_review_text("  Payment FAILED!!!  "),
            "payment failed",
        )

    def test_keeps_numbers_used_in_review_text(self) -> None:
        self.assertEqual(preprocess_review_text("OTP 1234 not received"), "otp 1234 not received")

    def test_loads_uploaded_csv_using_a_supported_text_column(self) -> None:
        dataset = load_uploaded_review_file(b"review\nPayment failed!\nOTP not received\n")
        self.assertEqual(dataset["clean_text"].tolist(), ["payment failed", "otp not received"])


if __name__ == "__main__":
    unittest.main()
