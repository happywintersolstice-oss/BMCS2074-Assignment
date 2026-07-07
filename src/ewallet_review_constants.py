"""
Shared constants for the e-wallet review classification app.

What this file does:
- stores fixed values used across the whole project
- keeps labels, titles, paths, and dataset column candidates in one place

How it works:
- other files import values from here instead of repeating them
- this reduces mistakes and makes future updates easier
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "ewallet_reviews_demo.csv"

MODEL_NAMES = ["Naive Bayes", "SVM", "Logistic Regression"]

TFIDF_SETTINGS = {
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
}

# These are the exact machine-friendly labels used for training and prediction.
LABEL_NAMES = [
    "payment_failure",
    "account_access_issue",
    "transfer_issue",
    "security_concern",
    "feature_request",
]

LABEL_DISPLAY_NAMES = {
    "payment_failure": "Payment Failure",
    "account_access_issue": "Account Access Issue",
    "transfer_issue": "Transfer Issue",
    "security_concern": "Security Concern",
    "feature_request": "Feature Request",
}

# These help the loader accept slightly different dataset column names.
TEXT_COLUMN_CANDIDATES = [
    "text",
    "content",
    "review",
    "comment",
    "feedback",
    "body",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "category",
    "class",
    "target",
    "type",
]

MAX_TRAIN_TEXT_LENGTH = 2000

TARGET_REVIEWS_PER_CLASS = {
    "payment_failure": 60,
    "account_access_issue": 60,
    "transfer_issue": 60,
    "security_concern": 60,
    "feature_request": 60,
}

APP_TITLE = "E-Wallet App Review Issue Classification Using NLP"
APP_SUBTITLE = (
    "A Streamlit app that uses TF-IDF with classic NLP models to classify e-wallet app "
    "reviews into actionable issue categories."
)
