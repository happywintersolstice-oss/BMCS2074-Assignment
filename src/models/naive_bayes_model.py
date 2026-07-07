"""
Naive Bayes model pipeline and training helpers.

What this file does:
- builds the Naive Bayes text-classification pipeline
- trains that pipeline
- returns its evaluation metrics

How it works:
- TF-IDF converts review text into numbers
- Multinomial Naive Bayes learns word-probability patterns for each label
"""

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.ewallet_review_constants import TFIDF_SETTINGS


def build_naive_bayes_pipeline() -> Pipeline:
    """
    Build the Naive Bayes pipeline with shared TF-IDF features.
    """
    # Put TF-IDF and the classifier into one reusable pipeline object.
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)),
            ("classifier", MultinomialNB()),
        ]
    )


def train_naive_bayes(
    x_train: list[str],
    x_test: list[str],
    y_train: list[str],
    y_test: list[str],
) -> tuple[str, Any, dict[str, float | str]]:
    """
    Train Naive Bayes and return the trained pipeline with metrics.
    """
    # Train on the shared training split so results stay comparable with the other models.
    pipeline = build_naive_bayes_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "Model": "Naive Bayes",
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }
    return "Naive Bayes", pipeline, metrics
