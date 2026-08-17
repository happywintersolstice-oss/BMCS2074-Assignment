"""
SVM model pipeline and training helpers.

What this file does:
- builds the SVM text-classification pipeline
- trains that pipeline
- returns its evaluation metrics

How it works:
- TF-IDF converts review text into numbers
- LinearSVC learns class boundaries between issue categories
- calibration is added so the app can show confidence scores
"""

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.ewallet_review_constants import TFIDF_SETTINGS


def build_svm_pipeline() -> Pipeline:
    """
    Build the SVM pipeline with TF-IDF features.
    """
    # Calibration adds probability estimates so the interface can show confidence.
    # CalibratedClassifierCV wraps LinearSVC so the app can read probabilities later.
    classifier = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", random_state=42),
        cv=3,
    )
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)),
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
    # Fit the tuned SVM once and calculate comparable metrics on the test split.
    # Tune a small linear SVM search space so the model stays strong but still understandable.
    class_counts = {label: y_train.count(label) for label in set(y_train)}
    cv_splits = min(3, min(class_counts.values()))
    pipeline = tune_svm_pipeline(x_train, y_train, cv_splits)
    predictions = pipeline.predict(x_test)

    metrics = {
        "Model": "SVM",
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }
    return "SVM", pipeline, metrics


def tune_svm_pipeline(x_train: list[str], y_train: list[str], cv_splits: int) -> Pipeline:
    """
    Tune the linear SVM with a small grid search and return the best pipeline.
    """
    # Search only lightweight linear-SVM settings to keep training practical.
    pipeline = build_svm_pipeline()
    if cv_splits < 2:
        pipeline.fit(x_train, y_train)
        return pipeline

    parameter_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "classifier__estimator__C": [0.5, 1.0, 2.0],
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
