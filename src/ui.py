"""
Presentation layer for the Streamlit app.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.constants import APP_TITLE, MODEL_NAMES
from src.data_loader import load_dataset
from src.modeling import explain_prediction, train_models
from src.web_tools import extract_text_from_url


def render_app() -> None:
    """
    Render the full Streamlit application.
    """
    apply_theme()

    try:
        models, metrics_df = train_models()
        dataset = load_dataset()
    except Exception as error:
        st.error("The app could not start because dataset loading or model setup failed.")
        st.code(str(error))
        st.stop()

    render_header(dataset)
    render_sidebar()

    message_tab, url_tab, comparison_tab, dataset_tab = st.tabs(
        ["Message Analysis", "URL Analysis", "Model Comparison", "Dataset Preview"]
    )

    with message_tab:
        render_message_page(models)
    with url_tab:
        render_url_page(models)
    with comparison_tab:
        render_comparison_page(metrics_df)
    with dataset_tab:
        render_dataset_page(dataset)


def apply_theme() -> None:
    """
    Replace the default Streamlit look with a cleaner visual style.
    """
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, #dcefff 0%, transparent 24%),
                radial-gradient(circle at top right, #edf7ff 0%, transparent 22%),
                linear-gradient(180deg, #f6fbff 0%, #ffffff 100%);
            color: #1d2d3d;
        }
        .block-container {
            max-width: 980px;
            padding-top: 3rem;
            padding-bottom: 1.5rem;
        }
        .page-header {
            margin-bottom: 0.55rem;
        }
        .page-header h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
            line-height: 1.06;
            letter-spacing: -0.02em;
            color: #1a3148;
        }
        .page-header p {
            margin: 0;
            max-width: 760px;
            color: #59738b;
            font-size: 0.98rem;
        }
        .mini-stat {
            background: rgba(248, 252, 255, 0.96);
            border: 1px solid rgba(88, 139, 188, 0.14);
            border-radius: 14px;
            padding: 0.7rem 0.85rem;
            box-shadow: none;
        }
        .mini-stat-label {
            color: #67819a;
            font-size: 0.82rem;
        }
        .mini-stat-value {
            color: #1e3952;
            font-size: 1.35rem;
            font-weight: 650;
        }
        .section-card {
            background: rgba(250, 253, 255, 0.97);
            border: 1px solid rgba(88, 139, 188, 0.12);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: none;
        }
        .soft-note {
            background: rgba(225, 241, 255, 0.72);
            border-radius: 14px;
            border: 1px solid rgba(88, 139, 188, 0.12);
            padding: 0.9rem 1rem;
            color: #486276;
        }
        .soft-note strong {
            color: #1f3d57;
        }
        .result-card {
            background: rgba(248, 252, 255, 0.98);
            border: 1px solid rgba(88, 139, 188, 0.16);
            border-radius: 16px;
            padding: 1rem 1.1rem;
        }
        .result-card h4 {
            margin: 0 0 0.35rem 0;
            color: #34506a;
            font-size: 0.92rem;
            font-weight: 600;
        }
        .result-value {
            color: #14324d;
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.05;
        }
        .explanation-list {
            margin-top: 0.4rem;
            color: #506b82;
        }
        .token-chip {
            display: inline-block;
            margin: 0.2rem 0.35rem 0 0;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            background: rgba(215, 236, 252, 0.9);
            border: 1px solid rgba(88, 139, 188, 0.16);
            color: #27506f;
            font-size: 0.86rem;
        }
        div[data-testid="stSidebar"] {
            background: #f7fbff;
            border-right: 1px solid rgba(88, 139, 188, 0.08);
        }
        div[data-testid="stMetric"] {
            background: rgba(248, 252, 255, 0.95);
            border: 1px solid rgba(88, 139, 188, 0.14);
            padding: 0.9rem 1rem;
            border-radius: 16px;
        }
        div[data-testid="stTabs"] button {
            color: #627c95;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #2f79b7;
        }
        details[data-testid="stExpander"] {
            border: 1px solid rgba(88, 139, 188, 0.12);
            border-radius: 14px;
            background: rgba(250, 253, 255, 0.92);
            overflow: hidden;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(88, 139, 188, 0.12);
            border-radius: 14px;
            background: rgba(250, 253, 255, 0.92);
            overflow: hidden;
        }
        details[data-testid="stExpander"] summary {
            background: rgba(244, 250, 255, 0.96);
            color: #2f79b7 !important;
            border-radius: 14px;
        }
        div[data-testid="stExpander"] summary {
            background: rgba(244, 250, 255, 0.96) !important;
            color: #2f79b7 !important;
            border-radius: 14px;
        }
        details[data-testid="stExpander"] summary:hover {
            background: rgba(232, 244, 255, 0.98);
            color: #235f94 !important;
        }
        div[data-testid="stExpander"] summary:hover {
            background: rgba(232, 244, 255, 0.98) !important;
            color: #235f94 !important;
        }
        details[data-testid="stExpander"] summary span,
        details[data-testid="stExpander"] summary p,
        details[data-testid="stExpander"] summary div {
            color: inherit !important;
        }
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary div,
        div[data-testid="stExpander"] summary svg {
            color: inherit !important;
            fill: currentColor !important;
        }
        .stButton > button {
            background: #4f97d1;
            color: white;
            border: none;
            border-radius: 999px;
            padding: 0.55rem 1.15rem;
            font-weight: 600;
        }
        .stButton > button:hover {
            background: #3f86bf;
            color: white;
        }
        .stButton > button p,
        .stButton > button span,
        .stButton > button div {
            color: #ffffff !important;
        }
        .stSelectbox label, .stTextInput label, .stTextArea label {
            color: #5b7389 !important;
        }
        .stMarkdown h2 {
            margin-top: 0.15rem;
            margin-bottom: 0.55rem;
        }
        .stMarkdown p {
            margin-bottom: 0.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(dataset: pd.DataFrame) -> None:
    """
    Render the app heading and top-level stats.
    """
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{APP_TITLE}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    spam_count = int((dataset["label"] == "spam").sum())
    legit_count = int((dataset["label"] == "legitimate").sum())

    col1, col2, col3 = st.columns(3)
    with col1:
        render_stat("Messages", str(len(dataset)))
    with col2:
        render_stat("Scam/Spam Samples", str(spam_count))
    with col3:
        render_stat("Legitimate Samples", str(legit_count))

def render_sidebar() -> None:
    """
    Keep the sidebar minimal and informational only.
    """
    dataset = load_dataset()
    spam_count = int((dataset["label"] == "spam").sum())
    legit_count = int((dataset["label"] == "legitimate").sum())

    st.sidebar.markdown("### About")
    st.sidebar.caption("This app checks whether pasted SMS text, email text, or extracted webpage text looks like Scam/Spam or Legitimate.")

    st.sidebar.markdown("### Model Guide")
    st.sidebar.caption("Naive Bayes: simple probability baseline.")
    st.sidebar.caption("SVM: stronger at separating harder cases.")
    st.sidebar.caption("Logistic Regression: useful linear comparison model.")

    st.sidebar.markdown("### Quick Tips")
    st.sidebar.caption("Paste SMS content, email text, or suspicious short messages into Message Analysis.")
    st.sidebar.caption("Use URL Analysis only for pages where readable text can actually be extracted.")

    st.sidebar.markdown("### Dataset Summary")
    st.sidebar.caption(f"Total samples: {len(dataset)}")
    st.sidebar.caption(f"Scam/Spam samples: {spam_count}")
    st.sidebar.caption(f"Legitimate samples: {legit_count}")


def render_model_picker(label: str, key: str) -> str:
    """
    Render a dropdown model selector.
    """
    return st.selectbox(label, MODEL_NAMES, key=key)


def render_stat(label: str, value: str) -> None:
    """
    Render a compact stat card.
    """
    st.markdown(
        f"""
        <div class="mini-stat">
            <div class="mini-stat-label">{label}</div>
            <div class="mini-stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message_page(models: dict[str, Any]) -> None:
    """
    Render the text message analysis page.
    """
    top_left, top_right = st.columns([1.35, 0.65], gap="large")

    with top_left:
        st.subheader("Analyze a Message")
        st.caption("Paste a suspicious SMS message or email text below.")
        message = st.text_area(
            "Message",
            height=120,
            placeholder="Example: Your account has been suspended. Verify now using this secure link...",
            label_visibility="collapsed",
        )
        model_choice = render_model_picker("Choose a model", "message_model")
        analyze = st.button("Analyze", use_container_width=True, type="primary")

    with top_right:
        st.subheader("Before You Start")
        st.markdown(
            """
            <div class="soft-note">
                Choose one model, paste the message, then click Analyze.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("The app will clean the text, convert it into TF-IDF features, and then classify it.")

    if analyze:
        if not message.strip():
            st.warning("Please enter a message before clicking Analyze.")
            return

        result = explain_prediction(models[model_choice], message)

        result_col, meta_col = st.columns([1.05, 0.95], gap="large")
        with result_col:
            st.markdown("### Result")
            if result["label"] == "Scam/Spam":
                st.error(f"Prediction: {result['label']}")
            else:
                st.success(f"Prediction: {result['label']}")
        with meta_col:
            st.markdown("### Confidence")
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-value">{result["confidence"]:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Model used: {model_choice}")

        with st.expander("See why the app gave this result"):
            explanation_left, explanation_right = st.columns([1.0, 1.0], gap="large")
            with explanation_left:
                st.markdown("**Cleaned text used by the model**")
                st.code(result["cleaned_text"] or "(no usable words after cleaning)", language=None)
            with explanation_right:
                st.markdown("**Top TF-IDF terms noticed in this message**")
                if result["top_terms"]:
                    chips = "".join(f'<span class="token-chip">{term}</span>' for term in result["top_terms"])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("No strong terms were found after preprocessing.")
                st.markdown(
                    """
                    <div class="explanation-list">
                        The result comes from the cleaned words in the message, their TF-IDF weights,
                        and the selected model's learned decision pattern.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_url_page(models: dict[str, Any]) -> None:
    """
    Render the webpage analysis page.
    """
    st.subheader("Analyze Suspicious Web Content")
    st.write("Enter a URL and the app will extract readable page text before classifying it.")
    url = st.text_input("URL", placeholder="https://example.com", label_visibility="collapsed")
    model_choice = render_model_picker("Choose a model for URL analysis", "url_model")
    analyze = st.button("Analyze URL", use_container_width=True, type="primary")

    if analyze:
        if not url.strip():
            st.warning("Please enter a URL before clicking Analyze URL.")
            return

        try:
            extracted_text = extract_text_from_url(url)
            result = explain_prediction(models[model_choice], extracted_text)

            preview_col, result_col = st.columns([1.25, 0.75], gap="large")
            with preview_col:
                st.subheader("Extracted Text Preview")
                preview = extracted_text[:900] + ("..." if len(extracted_text) > 900 else "")
                st.code(preview, language=None)

            with result_col:
                st.subheader("Prediction")
                if result["label"] == "Scam/Spam":
                    st.error(f"Prediction: {result['label']}")
                else:
                    st.success(f"Prediction: {result['label']}")
                st.markdown(
                    f"""
                    <div class="result-card">
                        <h4>Confidence</h4>
                        <div class="result-value">{result["confidence"]:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Model used: {model_choice}")

            with st.expander("See why the app gave this result"):
                st.markdown("**Cleaned text used by the model**")
                st.code(result["cleaned_text"] or "(no usable words after cleaning)", language=None)
                st.markdown("**Top TF-IDF terms noticed on this page**")
                if result["top_terms"]:
                    chips = "".join(f'<span class="token-chip">{term}</span>' for term in result["top_terms"])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.caption("No strong terms were found after preprocessing.")
                st.markdown(
                    """
                    <div class="explanation-list">
                        The result comes from the cleaned webpage text, its TF-IDF weights,
                        and the selected model's learned decision pattern.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as error:
            st.warning("The app could not extract readable content from that URL. Try another page or check the link.")
            st.caption(f"Technical detail: {error}")


def render_comparison_page(metrics_df: pd.DataFrame) -> None:
    """
    Render the model comparison table.
    """
    st.subheader("Model Comparison")
    st.write("These scores come from the current train/test split and help compare the three member-owned models.")

    formatted_df = metrics_df.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        formatted_df[column] = formatted_df[column].map(lambda value: f"{value:.2%}")

    st.dataframe(formatted_df, use_container_width=True, hide_index=True)


def render_dataset_page(dataset: pd.DataFrame) -> None:
    """
    Render a preview of the training data.
    """
    st.subheader("Dataset Preview")
    st.write("Use this page during demo or report writing to show the training examples used by the app.")
    st.dataframe(dataset[["label", "text"]], use_container_width=True, hide_index=True)
