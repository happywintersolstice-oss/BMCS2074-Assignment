"""
Train the Logistic Regression model without using Streamlit.

What this file does:
- loads the current training dataset
- splits the data into train and test sets
- trains the Logistic Regression pipeline
- prints the evaluation metrics in the terminal

How it works:
- it reuses the same dataset loader used by the app
- it reuses the same Logistic Regression training function used by the app
- this keeps manual training consistent with the main project
"""

from pathlib import Path
import sys

# Add the project root so this script can import from `src` when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ewallet_review_dataset import load_ewallet_review_dataset_with_summary, load_testing_review_dataset
from src.models.logistic_regression_model import train_logistic_regression


def main() -> None:
    """
    Run a manual Logistic Regression training flow and print the results.
    """
    # Load the same cleaned training and testing datasets that the Streamlit app uses.
    training_dataframe, summary = load_ewallet_review_dataset_with_summary()
    testing_dataframe = load_testing_review_dataset()

    x_train = training_dataframe["clean_text"].tolist()
    y_train = training_dataframe["label"].tolist()
    x_test = testing_dataframe["clean_text"].tolist()
    y_test = testing_dataframe["label"].tolist()

    model_name, _pipeline, metrics = train_logistic_regression(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("Manual training complete.")
    print(f"Model: {model_name}")
    print(f"Training dataset rows used after cleanup and balancing: {len(training_dataframe)}")
    print(f"Training split rows: {len(x_train)}")
    print(f"Testing split rows: {len(x_test)}")
    print()
    print("Dataset summary:")
    print(f"- Original rows in CSV: {summary['original_rows']}")
    print(f"- Final usable rows before balancing: {summary['final_rows']}")
    print(f"- Balanced rows used for training: {summary['balanced_rows']}")
    print()
    print("Metrics:")
    print(f"- Accuracy: {metrics['Accuracy']:.2%}")
    print(f"- Precision: {metrics['Precision']:.2%}")
    print(f"- Recall: {metrics['Recall']:.2%}")
    print(f"- F1 Score: {metrics['F1 Score']:.2%}")


if __name__ == "__main__":
    main()
