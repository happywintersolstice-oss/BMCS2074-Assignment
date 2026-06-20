"""
Shared constants for the NLP app.
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "sms_spam_demo.csv"

MODEL_NAMES = ["Naive Bayes", "SVM", "Logistic Regression"]

APP_TITLE = "AI-Powered Scam Message and Suspicious Web Content Detection Using NLP"
APP_SUBTITLE = (
    "A Streamlit prototype that uses TF-IDF with classic NLP models to classify messages "
    "and webpage text as Scam/Spam or Legitimate."
)
