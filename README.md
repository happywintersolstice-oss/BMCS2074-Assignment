# E-Wallet Review Issue Classification Using NLP

A Streamlit application for operational triage of e-wallet reviews. It classifies negative customer feedback into actionable issue categories while keeping training, evaluation, and live-review analysis separate.

## Issue Categories

- `payment_failure` — failed payments, declined transactions, or checkout problems
- `account_access_issue` — login, verification, account-lock, or OTP problems
- `transfer_issue` — delayed or failed money and bank transfers
- `security_concern` — suspicious activity, privacy, or unauthorized-access concerns
- `feature_request` — requested improvements or missing functionality

## Features

### Review Analysis

Paste one review, choose a model, and view its predicted issue category, confidence, cleaned text, and strongest TF-IDF terms.

### Google Play Review Triage

Enter a Google Play URL or Android package ID to collect recent reviews for an app. This is intended for live operational use:

- 4–5 star reviews are marked **Positive** and left unchanged.
- 1–3 star reviews are marked **Needs Review** and classified into an issue category.
- Results can be downloaded as a CSV for support or product teams.

Collected reviews are **never added automatically to the training dataset**.

### Manual Testing and CSV Upload

- Test a single review from the held-out test dataset.
- Evaluate a model across the full held-out testing dataset.
- Upload a CSV and classify every usable review in the file.
- Download prediction results as CSV.

An uploaded CSV needs one of these text columns: `text`, `content`, `review`, `comment`, `feedback`, or `body`.

### Model Comparison

Compare Naive Bayes, SVM, and Logistic Regression using accuracy, macro precision, macro recall, macro F1 score, cross-validation, per-class metrics, and confusion matrices.

## Models and Training Method

All models use the same preprocessing and TF-IDF text features:

- lowercase review text
- remove punctuation and special characters
- normalize repeated spaces
- use unigrams and bigrams

The models are:

- Multinomial Naive Bayes
- Calibrated linear SVM
- Logistic Regression

Each model is tuned with `GridSearchCV` and stratified cross-validation. The project currently recommends **SVM** as the strongest model based on the held-out evaluation.

## Data Safety and Evaluation

The app keeps model development and evaluation separate:

- `data/splits/ewallet_reviews_training.csv` is used for fitting and cross-validation.
- `data/splits/ewallet_reviews_testing_manual.csv` is used only for held-out evaluation.
- Exact cleaned-text overlap between the training and testing datasets is removed before fitting to prevent data leakage.
- Class balancing occurs only within the training flow; the held-out testing file remains unchanged.

## Run the App

Create and activate a virtual environment if needed, then install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal, typically `http://localhost:8501`.

## How to Use the UI

1. In the sidebar, select **Balanced** mode and click **Train / Retrain Models**.
2. Use **Review Analysis** for one review at a time.
3. Use **Google Play Triage** for current public-review monitoring. Enter a package ID such as `my.com.tngdigital.ewallet`, choose a review count and model, then click **Collect and Triage Reviews**.
4. Use **Manual Testing** to demonstrate held-out testing or to upload a separate CSV for prediction.
5. Use **Model Comparison** for report figures and model-selection discussion.

## Project Structure

- `app.py` — Streamlit entry point
- `src/ewallet_review_ui.py` — application interface
- `src/ewallet_review_modeling.py` — training, evaluation, prediction, and model persistence
- `src/ewallet_review_dataset.py` — dataset loading, validation, and overlap protection
- `src/ewallet_review_google_play.py` — live Google Play collection and rating-based triage
- `src/models/` — individual Naive Bayes, SVM, and Logistic Regression pipelines
- `data/splits/` — training and held-out testing files
- `tests/` — preprocessing, upload, and Google Play triage checks

## Development Checks

Run the automated checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Notes for the Report

Position the system as a decision-support tool that helps e-wallet support and product teams organize negative reviews faster. The live Google Play workflow is for triage only; retraining should use manually reviewed and verified labels.
