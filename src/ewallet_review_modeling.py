"""
Model training, persistence, evaluation, and prediction helpers for the e-wallet review classifier.

What this file does:
- trains all three models
- saves trained models and evaluation results to disk
- loads saved models back into the app
- predicts the issue category for a new review
- explains the prediction using top TF-IDF terms

How it works:
- the cleaned training dataset is used only for fitting
- the held-out testing dataset is used only for evaluation
- each model trains on the same training/testing split files for fair comparison
- the trained pipelines and result tables are saved as one artifact bundle
- the prediction flow cleans new text, transforms it with TF-IDF, and asks the selected model for a label
"""

from datetime import datetime
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.ewallet_review_constants import (
    LABEL_DISPLAY_NAMES,
    LABEL_NAMES,
    MODEL_ARTIFACT_PATH,
    MODEL_NAMES,
    TESTING_DATA_PATH,
    TRAINING_DATA_PATH,
)
from src.ewallet_review_dataset import load_ewallet_review_dataset, load_testing_review_dataset
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


def prepare_training_inputs(
    apply_balancing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str], list[str]]:
    """
    Load and validate the dedicated training and testing datasets.
    """
    # Load testing first so any shared text can be excluded from model fitting.
    testing_dataframe = load_testing_review_dataset()
    training_dataframe = load_ewallet_review_dataset(
        apply_balancing=apply_balancing,
        excluded_clean_texts=set(testing_dataframe["clean_text"]),
    )

    x_train = training_dataframe["clean_text"].tolist()
    y_train = training_dataframe["label"].tolist()
    x_test = testing_dataframe["clean_text"].tolist()
    y_test = testing_dataframe["label"].tolist()

    class_counts = training_dataframe["label"].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Training requires at least two classes in the dataset.")

    if (class_counts < 2).any():
        raise ValueError("Each training class must have at least 2 examples.")

    if len(training_dataframe) < 10:
        raise ValueError("Dataset is too small for stable model training.")

    return training_dataframe, testing_dataframe, x_train, y_train, x_test, y_test


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
    # Use the model-name lookup to keep all algorithms on the same input contract.
    if model_name not in MODEL_TRAINERS:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_TRAINERS[model_name](x_train, x_test, y_train, y_test)


def cross_validate_model(model_name: str, x: list[str], y: list[str]) -> dict[str, float]:
    """
    Run stratified cross-validation so evaluation is less sensitive to one small split.
    """
    # Preserve label proportions in each fold, then average each validation metric.
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


def build_detailed_evaluation(y_true: list[str], y_pred: list[str]) -> dict[str, pd.DataFrame]:
    """
    Build per-class metrics and a confusion matrix from the held-out testing results.
    """
    # Convert machine-readable labels into tables that users can interpret in the UI.
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_rows: list[dict[str, float | int | str]] = []
    for label in LABEL_NAMES:
        label_report = report[label]
        report_rows.append(
            {
                "Category": LABEL_DISPLAY_NAMES[label],
                "Precision": float(label_report["precision"]),
                "Recall": float(label_report["recall"]),
                "F1 Score": float(label_report["f1-score"]),
                "Support": int(label_report["support"]),
            }
        )

    confusion = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES)
    confusion_df = pd.DataFrame(
        confusion,
        index=[f"Actual: {LABEL_DISPLAY_NAMES[label]}" for label in LABEL_NAMES],
        columns=[f"Predicted: {LABEL_DISPLAY_NAMES[label]}" for label in LABEL_NAMES],
    )
    return {
        "per_class_metrics": pd.DataFrame(report_rows),
        "confusion_matrix": confusion_df,
    }


def train_models(
    apply_balancing: bool,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """
    Train all supported models and return them with summary and detailed evaluation tables.
    """
    # Give every algorithm identical inputs so its scores are directly comparable.
    # Load separate training and testing datasets so evaluation uses the real held-out test file.
    _, _, x_train, y_train, x_test, y_test = prepare_training_inputs(apply_balancing=apply_balancing)

    models: dict[str, Any] = {}
    metrics_rows: list[dict[str, float | str]] = []
    detailed_results: dict[str, dict[str, pd.DataFrame]] = {}

    # Train each model on the same data and collect its evaluation scores.
    for model_name in MODEL_NAMES:
        trained_name, pipeline, metrics = train_single_model(
            model_name=model_name,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )
        metrics.update(cross_validate_model(model_name=model_name, x=x_train, y=y_train))
        predictions = pipeline.predict(x_test)
        models[trained_name] = pipeline
        metrics_rows.append(metrics)
        detailed_results[trained_name] = build_detailed_evaluation(y_test, predictions)

    return models, pd.DataFrame(metrics_rows), detailed_results


def build_training_bundle(apply_balancing: bool) -> dict[str, Any]:
    """
    Train all models and package the results with simple metadata for later loading.
    """
    # Store models, evaluation tables, and dataset details in one reusable artifact.
    training_dataframe, testing_dataframe, _, _, _, _ = prepare_training_inputs(
        apply_balancing=apply_balancing
    )
    models, metrics_df, detailed_results = train_models(apply_balancing=apply_balancing)
    trained_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training_mode = "balanced" if apply_balancing else "unbalanced"

    return {
        "models": models,
        "metrics_df": metrics_df,
        "detailed_results": detailed_results,
        "metadata": {
            "trained_at": trained_at,
            "training_rows": int(len(training_dataframe)),
            "testing_rows": int(len(testing_dataframe)),
            "training_dataset": TRAINING_DATA_PATH.name,
            "testing_dataset": TESTING_DATA_PATH.name,
            "training_mode": training_mode,
        },
    }


def save_training_bundle(bundle: dict[str, Any]) -> None:
    """
    Save the trained model bundle so the app can load it on later runs.
    """
    # Ensure the artifact folder exists before writing the joblib bundle.
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_ARTIFACT_PATH)


def load_saved_training_bundle() -> dict[str, Any] | None:
    """
    Load the most recent saved model bundle if one exists.
    """
    # Validate the loaded structure early so the UI can report a useful error.
    if not MODEL_ARTIFACT_PATH.exists():
        return None

    bundle = joblib.load(MODEL_ARTIFACT_PATH)
    required_keys = {"models", "metrics_df", "detailed_results", "metadata"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError("Saved model artifact is missing required data.")
    return bundle


def train_and_save_models(apply_balancing: bool) -> dict[str, Any]:
    """
    Train the models once and save the full bundle to disk.
    """
    # Keep the button-triggered training flow as one simple reusable operation.
    bundle = build_training_bundle(apply_balancing=apply_balancing)
    save_training_bundle(bundle)
    return bundle


def get_saved_model_status(bundle: dict[str, Any] | None) -> dict[str, Any]:
    """
    Describe whether saved models exist and whether the datasets are newer than the artifact.
    """
    # Compare modification times to warn when saved predictions may be outdated.
    if not MODEL_ARTIFACT_PATH.exists():
        return {
            "exists": False,
            "artifact_modified": None,
            "is_stale": False,
            "metadata": {},
        }

    artifact_modified = datetime.fromtimestamp(MODEL_ARTIFACT_PATH.stat().st_mtime)
    newest_dataset_time = max(TRAINING_DATA_PATH.stat().st_mtime, TESTING_DATA_PATH.stat().st_mtime)
    is_stale = MODEL_ARTIFACT_PATH.stat().st_mtime < newest_dataset_time

    return {
        "exists": True,
        "artifact_modified": artifact_modified.strftime("%Y-%m-%d %H:%M:%S"),
        "is_stale": is_stale,
        "metadata": bundle.get("metadata", {}) if bundle else {},
    }


def explain_prediction(model: Any, raw_text: str) -> dict[str, Any]:
    """
    Predict the label for a review text and explain it.
    """
    # Reuse training preprocessing, then expose the strongest TF-IDF input terms.
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


def evaluate_model_on_testing_rows(model: Any, testing_dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """
    Run one trained model across the full held-out testing dataset and return row-level results.
    """
    # Predict every held-out row and calculate both row-level output and summary metrics.
    evaluation_rows = testing_dataset[["label", "text", "clean_text"]].copy().reset_index(drop=True)
    evaluation_rows.insert(0, "test_row", evaluation_rows.index + 1)

    cleaned_texts = evaluation_rows["clean_text"].tolist()
    actual_labels = evaluation_rows["label"].tolist()
    predicted_codes = model.predict(cleaned_texts).tolist()

    confidence_scores = [0.0] * len(evaluation_rows)
    if hasattr(model, "predict_proba"):
        probability_rows = model.predict_proba(cleaned_texts)
        confidence_scores = [float(max(probabilities)) for probabilities in probability_rows]

    results_df = pd.DataFrame(
        {
            "Test Row": evaluation_rows["test_row"],
            "Text": evaluation_rows["text"],
            "Actual Label": [LABEL_DISPLAY_NAMES[label] for label in actual_labels],
            "Predicted Label": [
                LABEL_DISPLAY_NAMES.get(label, str(label).replace("_", " ").title())
                for label in predicted_codes
            ],
            "Confidence": confidence_scores,
        }
    )
    results_df["Correct"] = results_df["Actual Label"] == results_df["Predicted Label"]

    summary = {
        "rows": int(len(results_df)),
        "correct": int(results_df["Correct"].sum()),
        "accuracy": float(accuracy_score(actual_labels, predicted_codes)),
        "precision": float(precision_score(actual_labels, predicted_codes, average="macro", zero_division=0)),
        "recall": float(recall_score(actual_labels, predicted_codes, average="macro", zero_division=0)),
        "f1_score": float(f1_score(actual_labels, predicted_codes, average="macro", zero_division=0)),
    }
    return results_df, summary


def predict_uploaded_review_file(model: Any, uploaded_dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict every usable review in a user-uploaded CSV file."""
    # This prediction-only path deliberately never changes the training dataset.
    evaluation_rows = uploaded_dataset[["text", "clean_text"]].copy().reset_index(drop=True)
    predicted_codes = model.predict(evaluation_rows["clean_text"].tolist()).tolist()

    confidence_scores = [0.0] * len(evaluation_rows)
    if hasattr(model, "predict_proba"):
        probability_rows = model.predict_proba(evaluation_rows["clean_text"].tolist())
        confidence_scores = [float(max(probabilities)) for probabilities in probability_rows]

    results_df = pd.DataFrame(
        {
            "File Row": evaluation_rows.index + 1,
            "Text": evaluation_rows["text"],
            "Predicted Label": [
                LABEL_DISPLAY_NAMES.get(label, str(label).replace("_", " ").title())
                for label in predicted_codes
            ],
            "Confidence": confidence_scores,
        }
    )
    summary = {
        "rows": int(len(results_df)),
        "label_counts": results_df["Predicted Label"].value_counts().to_dict(),
    }
    return results_df, summary
