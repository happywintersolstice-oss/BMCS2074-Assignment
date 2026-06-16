"""
Streamlit prototype for scam/spam message detection.

The app trains two simple NLP models on startup:
- Naive Bayes
- Support Vector Machine (SVM)

It supports:
- message analysis
- URL text extraction and analysis
- model comparison metrics
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# DATA_PATH points to the demo dataset bundled with this project.
DATA_PATH = Path("data/raw/sms_spam_demo.csv")


def preprocess_text(text: str) -> str:
    """
    Clean text before feature extraction.

    Steps:
    - convert to lowercase
    - remove punctuation
    - remove special characters
    - remove extra spaces
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset() -> pd.DataFrame:
    """
    Load the local demo dataset.

    The dataset uses:
    - spam: suspicious or scam-like content
    - ham: legitimate content
    """
    dataframe = pd.read_csv(DATA_PATH)
    dataframe["clean_text"] = dataframe["text"].astype(str).apply(preprocess_text)
    return dataframe


def build_pipeline(model_name: str) -> Any:
    """
    Build a TF-IDF + classifier pipeline based on the selected model.
    """
    if model_name == "Naive Bayes":
        classifier = MultinomialNB()
    else:
        classifier = SVC(kernel="linear", probability=True, random_state=42)

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", classifier),
        ]
    )


@st.cache_resource
def train_models() -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Train both models once and cache the result.

    Returns:
    - trained models
    - comparison metrics table
    """
    dataframe = load_dataset()
    # Convert to plain lists so scikit-learn and VS Code type checking agree more easily.
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

    for model_name in ["Naive Bayes", "SVM"]:
        pipeline = build_pipeline(model_name)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        models[model_name] = pipeline
        metrics_rows.append(
            {
                "Model": model_name,
                "Accuracy": float(accuracy_score(y_test, predictions)),
                "Precision": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
                "Recall": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
                "F1 Score": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    return models, metrics_df


def predict_text(model: Any, raw_text: str) -> tuple[str, float]:
    """
    Predict whether a message is spam/scam or legitimate.

    Returns:
    - display label
    - confidence score
    """
    cleaned = preprocess_text(raw_text)
    prediction = model.predict([cleaned])[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned])[0]
        confidence = float(max(probabilities))

    display_label = "Scam/Spam" if prediction == "spam" else "Legitimate"
    return display_label, confidence


def extract_text_from_url(url: str) -> str:
    """
    Download page content and extract readable text from it.

    A friendly error is raised if the page cannot be reached or parsed.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    # Remove elements that usually add noise to the extracted text.
    for element in soup(["script", "style", "noscript"]):
        element.extract()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError("No readable text was found on the page.")

    return text


def show_message_analysis(models: dict[str, Any]) -> None:
    """
    Render the message analysis section.
    """
    st.subheader("Message Analysis")
    st.write("Paste a message and choose which NLP model should analyze it.")

    model_choice = st.selectbox("Choose a model", ["Naive Bayes", "SVM"])
    message = st.text_area(
        "Enter a message",
        height=180,
        placeholder="Example: Congratulations! You have won a prize. Click this link now...",
    )

    if st.button("Analyze Message", use_container_width=True):
        if not message.strip():
            st.warning("Please enter a message before clicking Analyze.")
            return

        prediction, confidence = predict_text(models[model_choice], message)

        if prediction == "Scam/Spam":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.info(f"Confidence score: {confidence:.2%}")
        st.caption(f"Selected model: {model_choice}")


def show_url_analysis(models: dict[str, Any]) -> None:
    """
    Render the optional URL analysis section.
    """
    st.subheader("URL Analysis")
    st.write("Enter a URL to extract the web page text and analyze whether it looks suspicious.")

    model_choice = st.selectbox("Choose a model for URL analysis", ["Naive Bayes", "SVM"])
    url = st.text_input(
        "Enter a URL",
        placeholder="https://example.com",
    )

    if st.button("Analyze URL", use_container_width=True):
        if not url.strip():
            st.warning("Please enter a URL before clicking Analyze.")
            return

        try:
            extracted_text = extract_text_from_url(url)
            preview = extracted_text[:800] + ("..." if len(extracted_text) > 800 else "")
            st.write("Extracted page text preview:")
            st.code(preview)

            prediction, confidence = predict_text(models[model_choice], extracted_text)

            if prediction == "Scam/Spam":
                st.error(f"Prediction: {prediction}")
            else:
                st.success(f"Prediction: {prediction}")

            st.info(f"Confidence score: {confidence:.2%}")
            st.caption(f"Selected model: {model_choice}")
        except Exception as error:
            st.warning(
                "Sorry, the app could not extract readable content from that URL. "
                "Please try another page or check the link."
            )
            st.caption(f"Technical detail: {error}")


def show_model_comparison(metrics_df: pd.DataFrame) -> None:
    """
    Render the model comparison table.
    """
    st.subheader("Model Comparison")
    st.write("This table compares the two models using the current demo dataset.")

    formatted_df = metrics_df.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        formatted_df[column] = formatted_df[column].map(lambda value: f"{value:.2%}")

    st.dataframe(formatted_df, use_container_width=True, hide_index=True)


def main() -> None:
    """
    Main Streamlit app entry point.
    """
    st.set_page_config(
        page_title="Scam Message Detection",
        layout="wide",
    )

    st.title("AI-Powered Scam Message and Suspicious Web Content Detection Using NLP")
    st.write(
        "This prototype uses TF-IDF with Naive Bayes and SVM to classify text as scam/spam or legitimate."
    )

    models, metrics_df = train_models()

    page = st.sidebar.radio(
        "Choose a section",
        ["Message Analysis", "URL Analysis", "Model Comparison"],
    )

    if page == "Message Analysis":
        show_message_analysis(models)
    elif page == "URL Analysis":
        show_url_analysis(models)
    else:
        show_model_comparison(metrics_df)

    with st.expander("Demo dataset details"):
        dataset = load_dataset()
        st.write(f"Number of rows: {len(dataset)}")
        st.write(dataset[["label", "text"]].head(10))


if __name__ == "__main__":
    main()
