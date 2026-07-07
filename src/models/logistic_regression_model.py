"""
Logistic Regression model pipeline and training helpers.

What this file does:
- builds the Logistic Regression text-classification pipeline
- trains that pipeline
- returns its evaluation metrics

How it works:
- TF-IDF converts review text into numbers
- Logistic Regression learns weighted linear signals for each issue class
"""

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.ewallet_review_constants import TFIDF_SETTINGS


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Build the Logistic Regression pipeline with TF-IDF features.
    """
    # Put TF-IDF and the classifier into one reusable pipeline object.
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_logistic_regression(
    x_train: list[str],
    x_test: list[str],
    y_train: list[str],
    y_test: list[str],
) -> tuple[str, Any, dict[str, float | str]]:
    """
    Train Logistic Regression and return the trained pipeline with metrics.
    """
    # Train on the shared training split so results stay comparable with the other models.
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "Model": "Logistic Regression",
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }
    return "Logistic Regression", pipeline, metrics
