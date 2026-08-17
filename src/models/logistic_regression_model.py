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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.ewallet_review_constants import TFIDF_SETTINGS


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Build the Logistic Regression pipeline with TF-IDF features.
    """
    # Keep feature extraction and classifier configuration together for reuse.
    # Put TF-IDF and the classifier into one reusable pipeline object.
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
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
    # Select the best training configuration before measuring unseen-test performance.
    # Tune a small Logistic Regression search space so the model stays suitable for text classification.
    class_counts = {label: y_train.count(label) for label in set(y_train)}
    cv_splits = min(3, min(class_counts.values()))
    pipeline = tune_logistic_regression_pipeline(x_train, y_train, cv_splits)
    predictions = pipeline.predict(x_test)

    metrics = {
        "Model": "Logistic Regression",
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }
    return "Logistic Regression", pipeline, metrics


def tune_logistic_regression_pipeline(
    x_train: list[str],
    y_train: list[str],
    cv_splits: int,
) -> Pipeline:
    """
    Tune Logistic Regression with a small grid search and return the best pipeline.
    """
    # Try a controlled set of TF-IDF and regularization values using cross-validation.
    pipeline = build_logistic_regression_pipeline()
    if cv_splits < 2:
        pipeline.fit(x_train, y_train)
        return pipeline

    parameter_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "classifier__C": [0.5, 1.0, 2.0],
    }
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=splitter,
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_
