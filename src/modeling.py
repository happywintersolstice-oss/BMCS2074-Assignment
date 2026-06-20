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


def predict_text(model: Any, raw_text: str) -> tuple[str, float]:
    """
    Predict the label for a message or extracted webpage text.
    """
    cleaned = preprocess_text(raw_text)
    prediction = model.predict([cleaned])[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        confidence = float(max(probabilities))

    display_label = "Scam/Spam" if prediction == "spam" else "Legitimate"
    return display_label, confidence
