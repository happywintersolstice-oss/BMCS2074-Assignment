"""
Presentation layer for the e-wallet review classification Streamlit app.

What this file does:
- builds the full Streamlit interface
- shows the review analysis page
- shows the model comparison page
- shows the dataset preview page

How it works:
- loads the trained models and cleaned dataset
- draws the page layout and styling
- calls the modeling module when the user clicks Analyze Review
"""

from typing import Any

import pandas as pd
import streamlit as st

from src.ewallet_review_constants import APP_SUBTITLE, APP_TITLE, LABEL_DISPLAY_NAMES, LABEL_NAMES, MODEL_NAMES
from src.ewallet_review_dataset import load_ewallet_review_dataset_with_summary
from src.ewallet_review_modeling import explain_prediction, train_models


def render_app() -> None:
    """
    Render the full Streamlit application.
    """
    apply_theme()

    # Load models and dataset once at startup so the rest of the UI can use them.
    try:
        models, metrics_df = train_models()
        dataset, dataset_summary = load_ewallet_review_dataset_with_summary()
    except Exception as error:
        st.error("The app could not start because dataset loading or model setup failed.")
        st.code(str(error))
        st.stop()

    render_header(dataset, dataset_summary)
    render_sidebar(dataset_summary)

    # Split the app into simple tabs so users can move between the main tasks.
    analyze_tab, comparison_tab, dataset_tab = st.tabs(
        ["Review Analysis", "Model Comparison", "Dataset Preview"]
    )

    with analyze_tab:
        render_review_page(models)
    with comparison_tab:
        render_comparison_page(metrics_df)
    with dataset_tab:
        render_dataset_page(dataset)


def apply_theme() -> None:
    """
    Replace the default Streamlit look with a cleaner visual style.
    """
    # Inject custom CSS so the default Streamlit look matches the project style.
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
            max-width: 1020px;
            padding-top: 3rem;
            padding-bottom: 1.5rem;
        }
        .page-header {
            margin-bottom: 0.8rem;
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
            max-width: 820px;
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
            font-size: 1.2rem;
            font-weight: 650;
        }
        .soft-note {
            background: rgba(225, 241, 255, 0.72);
            border-radius: 14px;
            border: 1px solid rgba(88, 139, 188, 0.12);
            padding: 0.9rem 1rem;
            color: #486276;
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
        .token-score {
            margin-top: 0.75rem;
            color: #486276;
            font-size: 0.92rem;
        }
        .token-score strong {
            color: #1f3d57;
        }
        div[data-testid="stSidebar"] {
            background: #f7fbff;
            border-right: 1px solid rgba(88, 139, 188, 0.08);
        }
        div[data-testid="stTabs"] button {
            color: #627c95;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #2f79b7;
        }
        details[data-testid="stExpander"],
        div[data-testid="stExpander"] {
            border: 1px solid rgba(88, 139, 188, 0.12);
            border-radius: 14px;
            background: rgba(250, 253, 255, 0.92);
            overflow: hidden;
        }
        details[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary {
            background: rgba(244, 250, 255, 0.96) !important;
            color: #2f79b7 !important;
            border-radius: 14px;
        }
        details[data-testid="stExpander"] summary:hover,
        div[data-testid="stExpander"] summary:hover {
            background: rgba(232, 244, 255, 0.98) !important;
            color: #235f94 !important;
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
        .stSelectbox label, .stTextArea label {
            color: #5b7389 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(dataset: pd.DataFrame, dataset_summary: dict[str, int | float | dict[str, int]]) -> None:
    """
    Render the app heading and top-level stats.
    """
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    label_counts = dataset_summary["label_counts"]

    col1, col2, col3 = st.columns(3)
    with col1:
        render_stat("Reviews", str(len(dataset)))
    with col2:
        render_stat("Issue Categories", str(len(LABEL_NAMES)))
    with col3:
        # Show which category currently has the most rows in the dataset.
        largest_label = max(label_counts, key=label_counts.get)
        render_stat("Largest Category", LABEL_DISPLAY_NAMES[largest_label])


def render_sidebar(dataset_summary: dict[str, int | float | dict[str, int]]) -> None:
    """
    Keep the sidebar informational and specific to the current assignment topic.
    """
    st.sidebar.markdown("### About")
    st.sidebar.caption(
        "This app classifies e-wallet app reviews into operational issue categories to help teams sort customer feedback faster."
    )

    st.sidebar.markdown("### Label Guide")
    st.sidebar.caption("Payment Failure: checkout or payment did not go through.")
    st.sidebar.caption("Account Access Issue: login, verification, or OTP problem.")
    st.sidebar.caption("Transfer Issue: money transfer or bank transfer problem.")
    st.sidebar.caption("Security Concern: suspicious activity or privacy worry.")
    st.sidebar.caption("Feature Request: requested improvement or missing function.")

    st.sidebar.markdown("### Model Guide")
    st.sidebar.caption("Naive Bayes: simple text probability baseline.")
    st.sidebar.caption("SVM: often strong at separating harder text categories.")
    st.sidebar.caption("Logistic Regression: clear linear multiclass comparison model.")

    st.sidebar.markdown("### Dataset Summary")
    st.sidebar.caption(f"Original rows: {dataset_summary['original_rows']}")
    st.sidebar.caption(f"Final usable rows: {dataset_summary['final_rows']}")
    st.sidebar.caption(f"Duplicates removed: {dataset_summary['duplicates_removed']}")
    st.sidebar.caption(f"Imbalance ratio: {dataset_summary['imbalance_ratio']}:1")
    for label in LABEL_NAMES:
        st.sidebar.caption(f"{LABEL_DISPLAY_NAMES[label]}: {dataset_summary['label_counts'][label]}")


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


def render_review_page(models: dict[str, Any]) -> None:
    """
    Render the review analysis page.
    """
    top_left, top_right = st.columns([1.35, 0.65], gap="large")

    with top_left:
        st.subheader("Analyze an E-Wallet Review")
        st.caption("Paste a user review, complaint, or feature suggestion below.")
        review_text = st.text_area(
            "Review",
            height=120,
            placeholder="Example: Transfer keeps failing and the money only returns after many hours.",
            label_visibility="collapsed",
        )
        model_choice = render_model_picker("Choose a model", "review_model")
        analyze = st.button("Analyze Review", use_container_width=True, type="primary")

    with top_right:
        st.subheader("Before You Start")
        st.markdown(
            """
            <div class="soft-note">
                Choose one model, paste a review, then click Analyze Review.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "The app cleans the review text, converts it into TF-IDF features, and predicts the main operational issue category."
        )

    if analyze:
        if not review_text.strip():
            st.warning("Please enter a review before clicking Analyze Review.")
            return

        # Run the selected model only after the user provides review text.
        result = explain_prediction(models[model_choice], review_text)

        result_col, meta_col = st.columns([1.05, 0.95], gap="large")
        with result_col:
            st.markdown("### Result")
            st.info(f"Predicted issue category: {result['label']}")
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
                st.markdown("**Top TF-IDF terms noticed in this review**")
                if result["top_terms"]:
                    # Show the most influential cleaned words in a beginner-friendly way.
                    chips = "".join(
                        f'<span class="token-chip">{item["term"]}</span>'
                        for item in result["top_terms"]
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                    for item in result["top_terms"]:
                        st.markdown(
                            f"""
                            <div class="token-score">
                                <strong>{item["term"]}</strong>: TF-IDF weight = {item["score"]:.3f}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No strong terms were found after preprocessing.")
                st.markdown(
                    f"""
                    <div class="explanation-list">
                        {result["label_reason"]}
                        These are the highest-weight words from this review after cleaning, so they had the biggest effect on the selected model's decision.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_comparison_page(metrics_df: pd.DataFrame) -> None:
    """
    Render the model comparison table.
    """
    st.subheader("Model Comparison")
    st.write(
        "These scores come from the shared train/test split and compare how the three member-owned models handle multiclass e-wallet review classification."
    )

    # Format scores as percentages so they are easier to read in the app.
    formatted_df = metrics_df.copy()
    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "CV Accuracy",
        "CV Precision",
        "CV Recall",
        "CV F1 Score",
    ]:
        formatted_df[column] = formatted_df[column].map(lambda value: f"{value:.2%}")

    st.dataframe(formatted_df, use_container_width=True, hide_index=True)


def render_dataset_page(dataset: pd.DataFrame) -> None:
    """
    Render a preview of the training data.
    """
    st.subheader("Dataset Preview")
    st.write(
        "Use this page to inspect the manually labeled e-wallet review dataset used by the models."
    )

    # Convert internal label codes into readable names for the preview table.
    preview = dataset[["label", "text"]].copy()
    preview["label"] = preview["label"].map(lambda value: LABEL_DISPLAY_NAMES.get(value, value))
    st.dataframe(preview, use_container_width=True, hide_index=True)
