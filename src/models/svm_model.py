"""
SVM model pipeline and training helpers.
"""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_svm_pipeline() -> Pipeline:
    """
    Build the SVM pipeline with TF-IDF features.
    """
    classifier = CalibratedClassifierCV(LinearSVC(random_state=42), cv=3)
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", classifier),
        ]
    )


def train_svm(
    x_train: list[str],
    x_test: list[str],
    y_train: list[str],
    y_test: list[str],
) -> tuple[str, Any, dict[str, float | str]]:
    """
    Train SVM and return the trained pipeline with metrics.
    """
    pipeline = build_svm_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "Model": "SVM",
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
    }
    return "SVM", pipeline, metrics
