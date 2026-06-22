"""
Model training, evaluation, and prediction helpers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.constants import MODEL_NAMES
from src.data_loader import load_dataset
from src.models.logistic_regression_model import build_logistic_regression_pipeline, train_logistic_regression
from src.models.naive_bayes_model import build_naive_bayes_pipeline, train_naive_bayes
from src.models.svm_model import build_svm_pipeline, train_svm
from src.text_processing import preprocess_text


def get_model_builders() -> dict[str, Any]:
    """
    Return a mapping from display names to their dedicated builder functions.
    """
    return {
        "Naive Bayes": build_naive_bayes_pipeline,
        "SVM": build_svm_pipeline,
        "Logistic Regression": build_logistic_regression_pipeline,
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
    trainers = {
        "Naive Bayes": train_naive_bayes,
        "SVM": train_svm,
        "Logistic Regression": train_logistic_regression,
    }
    if model_name not in trainers:
        raise ValueError(f"Unsupported model: {model_name}")
    return trainers[model_name](x_train, x_test, y_train, y_test)


@st.cache_resource
def train_models() -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Train all supported models and return them with a comparison table.
    """
    dataframe = load_dataset()
    x = dataframe["clean_text"].tolist()
    y = dataframe["label"].tolist()

    class_counts = dataframe["label"].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Training requires at least two classes in the dataset.")

    if (class_counts < 2).any():
        raise ValueError("Each class must have at least 2 examples for a stratified train/test split.")

    if len(dataframe) < 10:
        raise ValueError("Dataset is too small for stable model training.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    models: dict[str, Any] = {}
    metrics_rows: list[dict[str, float | str]] = []

    for model_name in MODEL_NAMES:
        trained_name, pipeline, metrics = train_single_model(
            model_name=model_name,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        models[trained_name] = pipeline
        metrics_rows.append(metrics)

    return models, pd.DataFrame(metrics_rows)


def explain_prediction(model: Any, raw_text: str) -> dict[str, Any]:
    """
    Predict the label for a message or extracted webpage text and explain it.
    """
    cleaned = preprocess_text(raw_text)
    prediction = model.predict([cleaned])[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        confidence = float(max(probabilities))

    tfidf = model.named_steps["tfidf"]
    transformed = tfidf.transform([cleaned])
    feature_names = tfidf.get_feature_names_out()
    non_zero_indices = transformed.nonzero()[1]

    weighted_terms: list[tuple[str, float]] = []
    for index in non_zero_indices:
        weighted_terms.append((feature_names[index], float(transformed[0, index])))

    weighted_terms.sort(key=lambda item: item[1], reverse=True)
    top_terms = [
        {"term": term, "score": score}
        for term, score in weighted_terms[:5]
    ]

    display_label = "Scam/Spam" if prediction == "spam" else "Legitimate"
    label_reason = (
        "The model leaned toward Scam/Spam because these terms carried the strongest weight in the input text."
        if prediction == "spam"
        else "The model leaned toward Legitimate because the overall wording looked closer to legitimate training examples."
    )
    return {
        "label": display_label,
        "confidence": confidence,
        "cleaned_text": cleaned,
        "top_terms": top_terms,
        "label_reason": label_reason,
    }


def predict_text(model: Any, raw_text: str) -> tuple[str, float]:
    """
    Backward-compatible helper for simple prediction usage.
    """
    result = explain_prediction(model, raw_text)
    return result["label"], result["confidence"]
