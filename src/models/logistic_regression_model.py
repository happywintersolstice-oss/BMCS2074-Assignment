"""
Logistic Regression model pipeline and training helpers.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Build the Logistic Regression pipeline with TF-IDF features.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
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
    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "Model": "Logistic Regression",
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
    }
    return "Logistic Regression", pipeline, metrics
