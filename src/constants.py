"""
Shared constants for the NLP app.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "sms_spam_demo.csv"

MODEL_NAMES = ["Naive Bayes", "SVM", "Logistic Regression"]

TEXT_COLUMN_CANDIDATES = [
    "text",
    "message",
    "email_text",
    "email_body",
    "body",
    "content",
    "sms",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "category",
    "class",
    "target",
    "type",
]

MAX_TRAIN_TEXT_LENGTH = 2000
MAX_URL_TEXT_LENGTH = 4000

APP_TITLE = "AI-Powered Scam Message and Suspicious Web Content Detection Using NLP"
APP_SUBTITLE = (
    "A Streamlit app that uses TF-IDF with classic NLP models to classify SMS messages, "
    "email text, and webpage text as Scam/Spam or Legitimate."
)
