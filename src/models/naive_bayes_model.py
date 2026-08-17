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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.ewallet_review_constants import TFIDF_SETTINGS


NAIVE_BAYES_TFIDF_SETTINGS = {
    **TFIDF_SETTINGS,
    "min_df": 1,
    "sublinear_tf": True,
}


def build_naive_bayes_pipeline() -> Pipeline:
    """
    Build the Naive Bayes pipeline with shared TF-IDF features.
    """
    # Bundle text vectorization and classification so both steps stay consistent.
    # Naive Bayes benefits from keeping rare issue-specific words in small datasets.
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**NAIVE_BAYES_TFIDF_SETTINGS)),
            ("classifier", MultinomialNB(alpha=0.5)),
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
    # Tune on training data, then score the selected pipeline only on held-out data.
    # Tune only lightweight Naive Bayes and TF-IDF settings on the training split.
    class_counts = {label: y_train.count(label) for label in set(y_train)}
    cv_splits = min(3, min(class_counts.values()))
    pipeline = tune_naive_bayes_pipeline(x_train, y_train, cv_splits)
    predictions = pipeline.predict(x_test)

    metrics = {
        "Model": "Naive Bayes",
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }
    return "Naive Bayes", pipeline, metrics


def tune_naive_bayes_pipeline(x_train: list[str], y_train: list[str], cv_splits: int) -> Pipeline:
    """
    Tune Naive Bayes with a small grid search and return the best pipeline.
    """
    # Use cross-validation to select settings without touching the final test set.
    pipeline = build_naive_bayes_pipeline()
    if cv_splits < 2:
        pipeline.fit(x_train, y_train)
        return pipeline

    parameter_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__sublinear_tf": [False, True],
        "classifier__alpha": [0.1, 0.3, 0.5, 1.0],
        "classifier__fit_prior": [True, False],
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
