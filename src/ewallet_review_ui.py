"""
Presentation layer for the e-wallet review classification Streamlit app.

What this file does:
- builds the full Streamlit interface
- shows the review analysis page
- shows a manual testing page for held-out test rows
- shows the model comparison page
- shows the dataset preview page

How it works:
- loads the trained models plus the training and testing datasets
- draws the page layout and styling
- calls the modeling module when the user clicks Analyze Review
"""

from typing import Any

import pandas as pd
import streamlit as st

from src.ewallet_review_constants import (
    APP_SUBTITLE,
    APP_TITLE,
    BALANCE_CLASSES_FOR_TRAINING,
    LABEL_DISPLAY_NAMES,
    LABEL_NAMES,
    MODEL_NAMES,
)
from src.ewallet_review_dataset import (
    load_ewallet_review_dataset_with_summary,
    load_testing_review_dataset,
    load_uploaded_review_file,
)
from src.ewallet_review_google_play import (
    collect_google_play_reviews,
    extract_google_play_app_id,
    triage_google_play_reviews,
)
from src.ewallet_review_modeling import (
    explain_prediction,
    evaluate_model_on_testing_rows,
    get_saved_model_status,
    load_saved_training_bundle,
    predict_uploaded_review_file,
    train_and_save_models,
)


def get_balance_mode_label(apply_balancing: bool) -> str:
    """
    Convert the internal balancing flag into a short UI label.
    """
    return "Balanced" if apply_balancing else "Unbalanced"


def get_balance_mode_value(mode_label: str) -> bool:
    """
    Convert the selected UI label back into the balancing flag used by training.
    """
    return mode_label == "Balanced"


def render_app() -> None:
    """
    Render the full Streamlit application.
    """
    apply_theme()

    models: dict[str, Any] | None = None
    metrics_df: pd.DataFrame | None = None
    detailed_results: dict[str, dict[str, pd.DataFrame]] | None = None
    bundle_warning: str | None = None

    if "training_bundle" not in st.session_state:
        try:
            st.session_state["training_bundle"] = load_saved_training_bundle()
        except Exception as error:
            st.session_state["training_bundle"] = None
            bundle_warning = str(error)

    training_bundle = st.session_state.get("training_bundle")
    default_balance_mode = get_balance_mode_label(BALANCE_CLASSES_FOR_TRAINING)
    saved_mode = default_balance_mode
    if training_bundle:
        saved_mode = get_balance_mode_label(
            training_bundle["metadata"].get("training_mode", "balanced") == "balanced"
        )

    if "training_balance_mode" not in st.session_state:
        st.session_state["training_balance_mode"] = saved_mode

    apply_balancing = get_balance_mode_value(st.session_state["training_balance_mode"])

    # Load the dataset using the currently selected training mode so the summary matches the next train action.
    try:
        testing_dataset = load_testing_review_dataset()
        dataset, dataset_summary = load_ewallet_review_dataset_with_summary(
            apply_balancing=apply_balancing,
            excluded_clean_texts=set(testing_dataset["clean_text"]),
        )
    except Exception as error:
        st.error("The app could not start because dataset loading or model setup failed.")
        st.code(str(error))
        st.stop()

    if training_bundle:
        models = training_bundle["models"]
        metrics_df = training_bundle["metrics_df"]
        detailed_results = training_bundle["detailed_results"]

    training_status = get_saved_model_status(training_bundle)

    render_header(dataset_summary, len(testing_dataset), apply_balancing)
    training_bundle, selected_page = render_sidebar(dataset_summary, training_bundle, training_status)

    if bundle_warning:
        st.warning(f"Saved models could not be loaded: {bundle_warning}")

    message = st.session_state.pop("training_message", None)
    if training_bundle:
        models = training_bundle["models"]
        metrics_df = training_bundle["metrics_df"]
        detailed_results = training_bundle["detailed_results"]

    render_app_status(training_bundle, training_status, message)

    if selected_page == "Review Analysis":
        render_review_page(models)
    elif selected_page == "Google Play Triage":
        render_google_play_triage_page(models)
    elif selected_page == "Manual Testing":
        render_manual_testing_page(models, testing_dataset)
    elif selected_page == "Model Comparison":
        render_comparison_page(metrics_df, detailed_results)
    else:
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
        .status-strip {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 0 0 1.35rem 0;
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: rgba(248, 252, 255, 0.96);
            border: 1px solid rgba(88, 139, 188, 0.12);
        }
        .status-strip.warning {
            background: rgba(255, 247, 226, 0.96);
            border-color: rgba(227, 168, 70, 0.24);
        }
        .status-strip.info {
            background: rgba(239, 247, 255, 0.96);
            border-color: rgba(88, 139, 188, 0.16);
        }
        .status-dot {
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 999px;
            background: #4f97d1;
            flex: 0 0 auto;
        }
        .status-strip.warning .status-dot {
            background: #d58d2f;
        }
        .status-strip.info .status-dot {
            background: #4f97d1;
        }
        .status-copy {
            color: #355069;
            font-size: 0.95rem;
            line-height: 1.45;
        }
        .status-copy strong {
            color: #173752;
        }
        .header-stats-gap {
            height: 0.95rem;
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
        div[data-testid="stSidebar"] .stCaption,
        div[data-testid="stSidebar"] .stCaption p,
        div[data-testid="stSidebar"] label,
        div[data-testid="stSidebar"] p {
            color: #5b7389 !important;
            -webkit-text-fill-color: #5b7389 !important;
        }
        details[data-testid="stExpander"],
        div[data-testid="stExpander"] {
            border: 1px solid rgba(88, 139, 188, 0.12);
            border-radius: 14px;
            background: rgba(250, 253, 255, 0.92);
            overflow: hidden;
        }
        div[data-testid="stSidebar"] details[data-testid="stExpander"] .stCaption,
        div[data-testid="stSidebar"] details[data-testid="stExpander"] .stCaption p,
        div[data-testid="stSidebar"] details[data-testid="stExpander"] p,
        div[data-testid="stSidebar"] div[data-testid="stExpander"] .stCaption,
        div[data-testid="stSidebar"] div[data-testid="stExpander"] .stCaption p,
        div[data-testid="stSidebar"] div[data-testid="stExpander"] p {
            color: #4d667d !important;
            -webkit-text-fill-color: #4d667d !important;
            opacity: 1 !important;
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
        div[data-testid="stTextArea"] textarea {
            background: #2b2d36 !important;
            color: #f5f9ff !important;
            -webkit-text-fill-color: #f5f9ff !important;
            caret-color: #f5f9ff !important;
        }
        div[data-testid="stTextArea"] textarea::placeholder {
            color: #a6b7c9 !important;
            -webkit-text-fill-color: #a6b7c9 !important;
        }
        div[data-testid="stTextArea"] textarea:focus {
            border-color: #4f97d1 !important;
            box-shadow: 0 0 0 1px rgba(79, 151, 209, 0.35) !important;
        }
        .section-intro {
            margin: 0 0 0.85rem 0;
            color: #647c92;
            font-size: 0.96rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    dataset_summary: dict[str, int | float | dict[str, int]],
    testing_rows: int,
    apply_balancing: bool,
) -> None:
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

    current_mode_rows = (
        dataset_summary["balanced_rows"]
        if dataset_summary["applied_balancing"]
        else dataset_summary["final_rows"]
    )
    current_mode = "Balanced" if apply_balancing else "Unbalanced"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat("Training Rows", str(current_mode_rows))
    with col2:
        render_stat("Testing Rows", str(testing_rows))
    with col3:
        render_stat("Models", str(len(MODEL_NAMES)))
    with col4:
        render_stat("Mode", current_mode)

    st.markdown('<div class="header-stats-gap"></div>', unsafe_allow_html=True)


def render_app_status(
    training_bundle: dict[str, Any] | None,
    training_status: dict[str, Any],
    message: str | None,
) -> None:
    """
    Show one compact top-level status strip instead of multiple heavy alerts.
    """
    if message:
        st.markdown(
            f"""
            <div class="status-strip info">
                <div class="status-dot"></div>
                <div class="status-copy">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if training_bundle:
        metadata = training_status.get("metadata", {})
        trained_at = metadata.get("trained_at", "unknown time")
        training_mode = metadata.get("training_mode", "balanced").replace("_", " ").title()
        training_rows = metadata.get("training_rows", "?")
        testing_rows = metadata.get("testing_rows", "?")
        variant = "warning" if training_status.get("is_stale") else ""
        stale_note = " Saved models are older than the datasets." if training_status.get("is_stale") else ""
        st.markdown(
            f"""
            <div class="status-strip {variant}">
                <div class="status-dot"></div>
                <div class="status-copy">
                    <strong>Models Loaded.</strong>
                    Trained on {trained_at} using {training_mode.lower()} mode.
                    Training rows: {training_rows}. Testing rows: {testing_rows}.{stale_note}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-strip info">
                <div class="status-dot"></div>
                <div class="status-copy">
                    <strong>No Saved Models Yet.</strong>
                    Use the sidebar to choose the training mode and run the training step.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar(
    dataset_summary: dict[str, int | float | dict[str, int]],
    training_bundle: dict[str, Any] | None,
    training_status: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """
    Keep the sidebar minimal and focused on navigation plus training.
    """
    st.sidebar.markdown("### Navigation")
    selected_page = st.sidebar.selectbox(
        "Page",
        ["Review Analysis", "Google Play Triage", "Manual Testing", "Model Comparison", "Dataset Preview"],
        key="page_selector",
    )

    st.sidebar.markdown("### Training")
    selected_balance_mode = st.sidebar.radio(
        "Training data mode",
        ["Balanced", "Unbalanced"],
        key="training_balance_mode",
    )

    if training_status["exists"]:
        metadata = training_status["metadata"]
        st.sidebar.caption("Saved models are available.")
        if metadata.get("trained_at"):
            st.sidebar.caption(f"Last trained: {metadata['trained_at']}")
        if metadata.get("training_rows") and metadata.get("testing_rows"):
            st.sidebar.caption(
                f"Saved bundle used {metadata['training_rows']} training rows and {metadata['testing_rows']} testing rows."
            )
        if metadata.get("training_mode"):
            st.sidebar.caption(
                f"Saved bundle mode: {get_balance_mode_label(metadata['training_mode'] == 'balanced')}"
            )
        if training_status["is_stale"]:
            st.sidebar.warning("The datasets are newer than the saved models. Retrain before presenting new results.")
    else:
        st.sidebar.warning("No saved trained models found yet.")

    if training_status["exists"]:
        saved_mode = training_status["metadata"].get("training_mode")
        if saved_mode in {"balanced", "unbalanced"}:
            selected_mode_internal = "balanced" if selected_balance_mode == "Balanced" else "unbalanced"
            if saved_mode != selected_mode_internal:
                st.sidebar.info("The selected training mode is different from the saved models. Retrain to apply it.")

    if st.sidebar.button("Train / Retrain Models", width="stretch", type="primary"):
        try:
            with st.spinner("Training all models and saving them for later use..."):
                apply_balancing = get_balance_mode_value(selected_balance_mode)
                training_bundle = train_and_save_models(apply_balancing=apply_balancing)
            st.session_state["training_bundle"] = training_bundle
            st.session_state.pop("manual_testing_all_results", None)
            st.session_state.pop("manual_testing_all_summary", None)
            st.session_state.pop("manual_testing_all_model", None)
            trained_at = training_bundle["metadata"]["trained_at"]
            trained_mode = get_balance_mode_label(training_bundle["metadata"]["training_mode"] == "balanced")
            st.session_state["training_message"] = (
                f"Models trained and saved successfully on {trained_at} using {trained_mode.lower()} training."
            )
            st.rerun()
        except Exception as error:
            st.sidebar.error("Training failed.")
            st.sidebar.code(str(error))

    with st.sidebar.expander("Dataset Details"):
        st.caption(f"Original rows: {dataset_summary['original_rows']}")
        st.caption(f"Final usable rows: {dataset_summary['final_rows']}")
        current_mode_rows = (
            dataset_summary["balanced_rows"]
            if dataset_summary["applied_balancing"]
            else dataset_summary["final_rows"]
        )
        st.caption(f"Rows used in selected mode: {current_mode_rows}")
        st.caption(f"Duplicates removed: {dataset_summary['duplicates_removed']}")
        if dataset_summary["overlap_rows_removed"]:
            st.caption(f"Training/test overlaps removed: {dataset_summary['overlap_rows_removed']}")
        st.caption(f"Imbalance ratio: {dataset_summary['imbalance_ratio']}:1")
        for label in LABEL_NAMES:
            before_count = dataset_summary["label_counts_before_balance"][label]
            after_count = dataset_summary["label_counts"][label]
            st.caption(f"{LABEL_DISPLAY_NAMES[label]}: {after_count} rows ({before_count} raw)")

    with st.sidebar.expander("Reference"):
        st.caption("Payment Failure: checkout or payment did not go through.")
        st.caption("Account Access Issue: login, verification, or OTP problem.")
        st.caption("Transfer Issue: money transfer or bank transfer problem.")
        st.caption("Security Concern: suspicious activity or privacy worry.")
        st.caption("Feature Request: requested improvement or missing function.")
        st.caption("Naive Bayes: simple text probability baseline.")
        st.caption("SVM: often strong at separating harder text categories.")
        st.caption("Logistic Regression: clear linear multiclass comparison model.")

    return training_bundle, selected_page


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


def render_review_page(models: dict[str, Any] | None) -> None:
    """
    Render the review analysis page.
    """
    st.subheader("Review Analysis")
    st.markdown(
        '<div class="section-intro">Paste one review and run a trained model.</div>',
        unsafe_allow_html=True,
    )

    review_text = st.text_area(
        "Review",
        height=130,
        placeholder="Example: Transfer keeps failing and the money only returns after many hours.",
        label_visibility="collapsed",
    )
    controls_left, controls_right = st.columns([1.0, 0.45], gap="large")
    with controls_left:
        model_choice = render_model_picker("Choose a model", "review_model")
    with controls_right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        analyze = st.button("Analyze Review", width="stretch", type="primary", disabled=models is None)

    if models is None:
        return

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


def render_google_play_triage_page(models: dict[str, Any] | None) -> None:
    """Collect current Google Play reviews and triage only negative feedback."""
    st.subheader("Google Play Review Triage")
    st.markdown(
        '<div class="section-intro">Collect recent Google Play reviews for operational triage. Positive and neutral reviews remain unchanged; only 1–2 star reviews are classified into issue categories.</div>',
        unsafe_allow_html=True,
    )
    if models is None:
        st.info("Train the models first from the sidebar before collecting reviews.")
        return

    app_reference = st.text_input(
        "Google Play URL or package id",
        placeholder="my.com.tngdigital.ewallet or https://play.google.com/store/apps/details?id=...",
        key="google_play_app_reference",
    )
    input_left, input_middle, input_right = st.columns([0.8, 0.8, 1.0], gap="large")
    with input_left:
        review_count = st.selectbox("Recent reviews to collect", [25, 50, 100, 200], index=1)
    with input_middle:
        google_play_model = render_model_picker("Model for negative reviews", "google_play_model")
    with input_right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        collect_and_triage = st.button("Collect and Triage Reviews", width="stretch", type="primary")

    if collect_and_triage:
        try:
            app_id = extract_google_play_app_id(app_reference)
            with st.spinner("Collecting recent Google Play reviews and triaging negative feedback..."):
                collected_reviews = collect_google_play_reviews(app_id, review_count, "en", "my")
                triage_results, triage_summary = triage_google_play_reviews(
                    models[google_play_model],
                    collected_reviews,
                )
            st.session_state["google_play_results"] = triage_results
            st.session_state["google_play_summary"] = triage_summary
            st.session_state["google_play_app_id"] = app_id
            st.session_state["google_play_model_used"] = google_play_model
        except Exception as error:
            st.error("Google Play reviews could not be collected.")
            st.code(str(error))

    triage_results = st.session_state.get("google_play_results")
    triage_summary = st.session_state.get("google_play_summary")
    if triage_results is not None and triage_summary is not None:
        st.caption(
            f"Live results for {st.session_state.get('google_play_app_id')} using "
            f"{st.session_state.get('google_play_model_used')}. These results do not modify the training dataset."
        )
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            render_stat("Collected", str(triage_summary["total"]))
        with stat_col2:
            render_stat("Negative Classified", str(triage_summary["negative"]))
        with stat_col3:
            render_stat("Positive Untouched", str(triage_summary["positive"]))
        with stat_col4:
            render_stat("Neutral Untouched", str(triage_summary["neutral"]))

        display_results = triage_results.copy()
        display_results["Confidence"] = display_results["Confidence"].map(
            lambda value: f"{value:.2%}" if pd.notna(value) else ""
        )
        st.dataframe(display_results, width="stretch", hide_index=True)
        st.download_button(
            "Download Google Play Triage CSV",
            data=triage_results.to_csv(index=False).encode("utf-8"),
            file_name="google_play_review_triage.csv",
            mime="text/csv",
            width="stretch",
        )


def render_manual_testing_page(models: dict[str, Any] | None, testing_dataset: pd.DataFrame) -> None:
    """
    Render a row-by-row manual testing page using the held-out testing dataset.
    """
    st.subheader("Manual Testing")
    st.markdown(
        '<div class="section-intro">Use held-out testing rows to inspect one prediction at a time or run the full testing file.</div>',
        unsafe_allow_html=True,
    )

    testing_rows = testing_dataset[["label", "text"]].copy().reset_index(drop=True)
    testing_rows.insert(0, "test_row", testing_rows.index + 1)

    category_options = ["All Categories"] + [LABEL_DISPLAY_NAMES[label] for label in LABEL_NAMES]
    selected_category = st.selectbox(
        "Filter testing rows by actual category",
        category_options,
        key="manual_testing_category",
    )

    filtered_rows = testing_rows.copy()
    if selected_category != "All Categories":
        filtered_rows = filtered_rows[
            filtered_rows["label"].map(lambda value: LABEL_DISPLAY_NAMES[value]) == selected_category
        ].reset_index(drop=True)

    st.caption(f"Rows available in this view: {len(filtered_rows)}")

    selected_row_id = st.selectbox(
        "Choose a testing row",
        filtered_rows["test_row"].tolist(),
        key="manual_testing_row",
        format_func=lambda row_id: format_testing_row_option(filtered_rows, row_id),
    )
    selected_row = filtered_rows.loc[filtered_rows["test_row"] == selected_row_id].iloc[0]
    actual_label = LABEL_DISPLAY_NAMES[selected_row["label"]]

    review_col, control_col = st.columns([1.25, 0.75], gap="large")
    with review_col:
        st.markdown("**Selected testing review**")
        st.code(selected_row["text"], language=None)
    with control_col:
        st.markdown("**Ground truth**")
        st.info(actual_label)
        model_choice = render_model_picker("Choose a model for manual testing", "manual_testing_model")
        run_manual_test = st.button(
            "Test Selected Review",
            width="stretch",
            type="primary",
            disabled=models is None,
            key="manual_testing_button",
        )

    if models is None:
        st.info("Train the models first from the sidebar, then come back here to test the held-out rows.")
        return

    if run_manual_test:
        result = explain_prediction(models[model_choice], selected_row["text"])
        predicted_label = result["label"]
        is_correct = predicted_label == actual_label

        result_left, result_right = st.columns([1.0, 1.0], gap="large")
        with result_left:
            st.markdown("### Manual Test Result")
            if is_correct:
                st.success(f"Correct prediction: {predicted_label}")
            else:
                st.error(f"Predicted: {predicted_label}")
                st.caption(f"Actual: {actual_label}")
        with result_right:
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

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            render_stat("Testing Row", str(int(selected_row["test_row"])))
        with summary_col2:
            render_stat("Actual Label", actual_label)
        with summary_col3:
            render_stat("Match", "Yes" if is_correct else "No")

        with st.expander("See model explanation for this testing row"):
            explanation_left, explanation_right = st.columns([1.0, 1.0], gap="large")
            with explanation_left:
                st.markdown("**Cleaned text used by the model**")
                st.code(result["cleaned_text"] or "(no usable words after cleaning)", language=None)
            with explanation_right:
                st.markdown("**Top TF-IDF terms noticed in this review**")
                if result["top_terms"]:
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
                        These are the highest-weight words from this testing review after cleaning, so they had the biggest effect on the selected model's decision.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.subheader("Full Testing Run")
    st.markdown(
        '<div class="section-intro">Run one trained model across the entire held-out testing dataset and inspect the full output table.</div>',
        unsafe_allow_html=True,
    )

    batch_left, batch_right = st.columns([1.0, 1.0], gap="large")
    with batch_left:
        batch_model_choice = render_model_picker("Choose a model for full testing", "batch_testing_model")
    with batch_right:
        run_full_testing = st.button(
            "Run Full Testing Dataset",
            width="stretch",
            type="primary",
            disabled=models is None,
            key="run_full_testing_button",
        )

    if models is None:
        return

    if run_full_testing:
        with st.spinner("Running the selected model across all testing rows..."):
            results_df, summary = evaluate_model_on_testing_rows(models[batch_model_choice], testing_dataset)
        st.session_state["manual_testing_all_results"] = results_df
        st.session_state["manual_testing_all_summary"] = summary
        st.session_state["manual_testing_all_model"] = batch_model_choice

    stored_results = st.session_state.get("manual_testing_all_results")
    stored_summary = st.session_state.get("manual_testing_all_summary")
    stored_model = st.session_state.get("manual_testing_all_model")

    if stored_results is not None and stored_summary is not None and stored_model is not None:
        st.caption(f"Showing full testing output for: {stored_model}")

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            render_stat("Testing Rows", str(stored_summary["rows"]))
        with summary_col2:
            render_stat("Correct Rows", str(stored_summary["correct"]))
        with summary_col3:
            render_stat("Accuracy", f"{stored_summary['accuracy']:.2%}")

        summary_col4, summary_col5, summary_col6 = st.columns(3)
        with summary_col4:
            render_stat("Precision", f"{stored_summary['precision']:.2%}")
        with summary_col5:
            render_stat("Recall", f"{stored_summary['recall']:.2%}")
        with summary_col6:
            render_stat("F1 Score", f"{stored_summary['f1_score']:.2%}")

        show_only_wrong = st.checkbox("Show incorrect predictions only", key="batch_show_only_wrong")
        display_results = stored_results.copy()
        if show_only_wrong:
            display_results = display_results[~display_results["Correct"]].reset_index(drop=True)

        display_results["Confidence"] = display_results["Confidence"].map(lambda value: f"{value:.2%}")
        st.dataframe(display_results, width="stretch", hide_index=True)
        csv_bytes = stored_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Testing Results CSV",
            data=csv_bytes,
            file_name="manual_testing_results.csv",
            mime="text/csv",
            width="stretch",
        )

    st.markdown("---")
    st.subheader("Test Your Own CSV File")
    st.markdown(
        '<div class="section-intro">Upload a CSV with a review-text column (`text`, `content`, `review`, `comment`, `feedback`, or `body`) to classify every usable row.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload review CSV", type=["csv"], key="uploaded_review_csv")
    upload_left, upload_right = st.columns([1.0, 1.0], gap="large")
    with upload_left:
        upload_model_choice = render_model_picker("Choose a model for uploaded file", "uploaded_file_model")
    with upload_right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        run_uploaded_file = st.button(
            "Classify Uploaded File",
            width="stretch",
            type="primary",
            disabled=uploaded_file is None,
            key="classify_uploaded_file_button",
        )

    if run_uploaded_file and uploaded_file is not None:
        try:
            with st.spinner("Classifying every usable row in the uploaded file..."):
                uploaded_dataset = load_uploaded_review_file(uploaded_file.getvalue())
                uploaded_results, uploaded_summary = predict_uploaded_review_file(
                    models[upload_model_choice],
                    uploaded_dataset,
                )
            st.session_state["uploaded_file_results"] = uploaded_results
            st.session_state["uploaded_file_summary"] = uploaded_summary
            st.session_state["uploaded_file_model"] = upload_model_choice
            st.session_state["uploaded_file_name"] = uploaded_file.name
        except Exception as error:
            st.error("The uploaded file could not be classified.")
            st.code(str(error))

    uploaded_results = st.session_state.get("uploaded_file_results")
    uploaded_summary = st.session_state.get("uploaded_file_summary")
    if uploaded_results is not None and uploaded_summary is not None:
        st.caption(
            f"Showing {uploaded_summary['rows']} classified rows from "
            f"{st.session_state.get('uploaded_file_name', 'uploaded file')} using "
            f"{st.session_state.get('uploaded_file_model', 'the selected model')}."
        )
        st.dataframe(
            uploaded_results.assign(Confidence=uploaded_results["Confidence"].map(lambda value: f"{value:.2%}")),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download Uploaded File Predictions CSV",
            data=uploaded_results.to_csv(index=False).encode("utf-8"),
            file_name="uploaded_review_predictions.csv",
            mime="text/csv",
            width="stretch",
        )


def format_testing_row_option(testing_rows: pd.DataFrame, row_id: int) -> str:
    """
    Build a readable selectbox label for one testing row.
    """
    selected_row = testing_rows.loc[testing_rows["test_row"] == row_id].iloc[0]
    label = LABEL_DISPLAY_NAMES[selected_row["label"]]
    text_preview = str(selected_row["text"]).strip().replace("\n", " ")
    if len(text_preview) > 70:
        text_preview = f"{text_preview[:70].rstrip()}..."
    return f"Row {int(row_id)} | {label} | {text_preview}"


def render_comparison_page(
    metrics_df: pd.DataFrame | None,
    detailed_results: dict[str, dict[str, pd.DataFrame]] | None,
) -> None:
    """
    Render the model comparison table and deeper held-out test results.
    """
    st.subheader("Model Comparison")
    st.markdown(
        '<div class="section-intro">Compare the three trained models using the same held-out testing dataset.</div>',
        unsafe_allow_html=True,
    )

    if metrics_df is None or detailed_results is None:
        return

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

    st.dataframe(formatted_df, width="stretch", hide_index=True)

    st.markdown("### Held-Out Test Breakdown")
    selected_model = st.selectbox(
        "Choose a model to inspect in detail",
        MODEL_NAMES,
        key="comparison_model_picker",
    )
    selected_details = detailed_results[selected_model]

    per_class_df = selected_details["per_class_metrics"].copy()
    for column in ["Precision", "Recall", "F1 Score"]:
        per_class_df[column] = per_class_df[column].map(lambda value: f"{value:.2%}")

    detail_left, detail_right = st.columns([1.0, 1.1], gap="large")
    with detail_left:
        st.markdown("**Per-class results on the testing dataset**")
        st.dataframe(per_class_df, width="stretch", hide_index=True)
    with detail_right:
        st.markdown("**Confusion matrix on the testing dataset**")
        st.dataframe(selected_details["confusion_matrix"], width="stretch")


def render_dataset_page(dataset: pd.DataFrame) -> None:
    """
    Render a preview of the training data.
    """
    st.subheader("Dataset Preview")
    st.markdown(
        '<div class="section-intro">Preview the labeled training rows currently used by the app.</div>',
        unsafe_allow_html=True,
    )

    # Convert internal label codes into readable names for the preview table.
    preview = dataset[["label", "text"]].copy()
    preview["label"] = preview["label"].map(lambda value: LABEL_DISPLAY_NAMES.get(value, value))
    st.dataframe(preview, width="stretch", hide_index=True)
