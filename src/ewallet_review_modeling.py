"""
Model training, evaluation, and prediction helpers for the e-wallet review classifier.

What this file does:
- trains all three models
- compares their metrics
- predicts the issue category for a new review
- explains the prediction using top TF-IDF terms

How it works:
- the cleaned dataset is split into train and test sets
- each model trains on the same split for fair comparison
- the prediction flow cleans new text, transforms it with TF-IDF, and asks the selected model for a label
"""

from typing import Any

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.ewallet_review_constants import LABEL_DISPLAY_NAMES, MODEL_NAMES
from src.ewallet_review_dataset import load_ewallet_review_dataset
from src.ewallet_review_text_processing import preprocess_review_text
from src.models.logistic_regression_model import build_logistic_regression_pipeline, train_logistic_regression
from src.models.naive_bayes_model import build_naive_bayes_pipeline, train_naive_bayes
from src.models.svm_model import build_svm_pipeline, train_svm


MODEL_BUILDERS = {
    "Naive Bayes": build_naive_bayes_pipeline,
    "SVM": build_svm_pipeline,
    "Logistic Regression": build_logistic_regression_pipeline,
}

MODEL_TRAINERS = {
    "Naive Bayes": train_naive_bayes,
    "SVM": train_svm,
    "Logistic Regression": train_logistic_regression,
}

LABEL_REASON_MAP = {
    "payment_failure": "The model leaned toward Payment Failure because the review contains strong wording about declined transactions, failed checkout, or payment problems.",
    "account_access_issue": "The model leaned toward Account Access Issue because the review contains strong wording about login trouble, verification problems, or OTP access issues.",
    "transfer_issue": "The model leaned toward Transfer Issue because the review contains strong wording about sending money, bank transfers, delays, or failed transfer actions.",
    "security_concern": "The model leaned toward Security Concern because the review contains strong wording about suspicious activity, privacy worries, or account safety concerns.",
    "feature_request": "The model leaned toward Feature Request because the review contains strong wording asking for improvements, new functions, or missing app features.",
}


def train_single_model(
    model_name: str,
    x_train: list[str],
    x_test: list[str],
    y_train: list[str],
    y_test: list[str],
) -> tuple[str, Any, dict[str, float | str]]:
    """
    Train one model through its dedicated module and return its pipeline with metrics.
    """
    if model_name not in MODEL_TRAINERS:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_TRAINERS[model_name](x_train, x_test, y_train, y_test)


def cross_validate_model(model_name: str, x: list[str], y: list[str]) -> dict[str, float]:
    """
    Run stratified cross-validation so evaluation is less sensitive to one small split.
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model: {model_name}")

    class_counts = pd.Series(y).value_counts()
    cv_splits = min(5, int(class_counts.min()))
    if cv_splits < 2:
        raise ValueError("Cross-validation requires at least 2 examples in every class.")

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = cross_validate(
        MODEL_BUILDERS[model_name](),
        x,
        y,
        cv=splitter,
        scoring=scoring,
        n_jobs=1,
    )
    return {
        "CV Accuracy": float(results["test_accuracy"].mean()),
        "CV Precision": float(results["test_precision_macro"].mean()),
        "CV Recall": float(results["test_recall_macro"].mean()),
        "CV F1 Score": float(results["test_f1_macro"].mean()),
    }


@st.cache_resource
def train_models() -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Train all supported models and return them with a comparison table.
    """
    # Load one shared cleaned dataset so every model is trained fairly.
    dataframe = load_ewallet_review_dataset()
    x = dataframe["clean_text"].tolist()
    y = dataframe["label"].tolist()

    class_counts = dataframe["label"].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Training requires at least two classes in the dataset.")

    if (class_counts < 2).any():
        raise ValueError("Each class must have at least 2 examples for a stratified train/test split.")

    if len(dataframe) < 10:
        raise ValueError("Dataset is too small for stable model training.")

    # Stratified split keeps the label balance similar in train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    models: dict[str, Any] = {}
    metrics_rows: list[dict[str, float | str]] = []

    # Train each model on the same data and collect its evaluation scores.
    for model_name in MODEL_NAMES:
        trained_name, pipeline, metrics = train_single_model(
            model_name=model_name,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        metrics.update(cross_validate_model(model_name=model_name, x=x, y=y))
        models[trained_name] = pipeline
        metrics_rows.append(metrics)

    return models, pd.DataFrame(metrics_rows)


def explain_prediction(model: Any, raw_text: str) -> dict[str, Any]:
    """
    Predict the label for a review text and explain it.
    """
    # Clean the user input using the same logic used during training.
    cleaned = preprocess_review_text(raw_text)
    prediction = model.predict([cleaned])[0]

    confidence = 0.0
    # Not every model gives probabilities, so check first before reading confidence.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        confidence = float(max(probabilities))

    # Reuse the trained TF-IDF step to inspect which input words received the strongest weights.
    tfidf = model.named_steps["tfidf"]
    transformed = tfidf.transform([cleaned])
    feature_names = tfidf.get_feature_names_out()
    non_zero_indices = transformed.nonzero()[1]

    weighted_terms: list[tuple[str, float]] = []
    for index in non_zero_indices:
        weighted_terms.append((feature_names[index], float(transformed[0, index])))

    weighted_terms.sort(key=lambda item: item[1], reverse=True)
    top_terms = [{"term": term, "score": score} for term, score in weighted_terms[:5]]

    display_label = LABEL_DISPLAY_NAMES.get(prediction, prediction.replace("_", " ").title())
    return {
        "label": display_label,
        "confidence": confidence,
        "cleaned_text": cleaned,
        "top_terms": top_terms,
        "label_reason": LABEL_REASON_MAP.get(
            prediction,
            "The model chose this issue category based on the strongest weighted terms in the cleaned review text.",
        ),
    }
