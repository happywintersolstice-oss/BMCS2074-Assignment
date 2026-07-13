# E-Wallet App Review Issue Classification Using NLP

This project is a Python + Streamlit NLP assignment app that classifies **e-wallet app reviews** into operational issue categories.

The current label set is:
- `payment_failure`
- `account_access_issue`
- `transfer_issue`
- `security_concern`
- `feature_request`

The project is framed around review triage for e-wallet apps such as:
- Touch 'n Go eWallet
- GrabPay
- Boost
- ShopeePay

## Why This Project

E-wallet providers receive large numbers of customer reviews. Manually reading and sorting them is slow and inconsistent. This project uses NLP text classification to organize those reviews into actionable categories that can help product, support, and operations teams respond faster.

## Current Scope

The app uses:
- TF-IDF for feature extraction
- Naive Bayes
- SVM
- Logistic Regression

The current preprocessing flow:
- lowercase text
- remove punctuation
- remove special characters
- remove extra spaces

## App Features

- `Review Analysis`
  - paste an e-wallet review, complaint, or feature suggestion
  - choose a model
  - click `Analyze Review`
  - view predicted issue category and confidence
  - view top TF-IDF terms and a short explanation

- `Model Comparison`
  - Accuracy
  - macro Precision
  - macro Recall
  - macro F1 Score

- `Dataset Preview`
  - inspect the current labeled review rows used for training

## Project Structure

- [`app.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\app.py)
  Streamlit entry point

- `data/raw/ewallet_reviews_demo.csv`
  Current 10,000-row weak-labeled e-wallet review dataset used by the app

- [`src/ewallet_review_constants.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\ewallet_review_constants.py)
  Shared paths, labels, app title, and model names

- [`src/ewallet_review_text_processing.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\ewallet_review_text_processing.py)
  Review text cleaning logic

- [`src/ewallet_review_dataset.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\ewallet_review_dataset.py)
  Dataset loading, label normalization, and dataset cleanup summary

- [`src/ewallet_review_modeling.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\ewallet_review_modeling.py)
  Shared training coordinator, prediction flow, and explanation logic

- [`src/ewallet_review_ui.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\ewallet_review_ui.py)
  Streamlit layout, styling, and page rendering

## Model Ownership

Each model has its own file so each group member can clearly own one model:

- [`src/models/naive_bayes_model.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\models\naive_bayes_model.py)
  Naive Bayes pipeline and training

- [`src/models/svm_model.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\models\svm_model.py)
  SVM pipeline and training

- [`src/models/logistic_regression_model.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\src\models\logistic_regression_model.py)
  Logistic Regression pipeline and training

## Dataset Notes

The app now uses a 10,000-row weak-labeled GCash review dataset stored directly in `ewallet_reviews_demo.csv`. It contains no normalized duplicate reviews and is ASCII-only so emoji cannot remain in the training text.

Because its labels were assigned using conservative keyword rules rather than manual annotation, create a separate human-labeled test set before reporting final model accuracy.

Recommended minimum class target:
- `payment_failure`: 60 reviews
- `account_access_issue`: 60 reviews
- `transfer_issue`: 60 reviews
- `security_concern`: 60 reviews
- `feature_request`: 60 reviews

If your team can collect more, aim for `80 to 100` reviews per class.

Recommended final dataset columns:
- `text`
- `label`
- optional: `app_name`
- optional: `store_source`

## Labeling Policy

Use one shared label taxonomy across the whole group:

- `payment_failure`
  - failed payment, declined transaction, checkout not going through
- `account_access_issue`
  - login failure, verification issue, locked account, OTP problem
- `transfer_issue`
  - transfer delay, bank transfer failure, wrong transfer behavior
- `security_concern`
  - suspicious activity, privacy concern, unauthorized access worry
- `feature_request`
  - requests for new features or product improvements

If a review matches more than one category, label it by the **main actionable issue mentioned first or most strongly**.

## How To Run

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the app:

```powershell
python -m streamlit run app.py
```

## Collecting Real Reviews

You can start collecting real Google Play reviews with the helper script:

```powershell
python scripts/collect_google_play_reviews.py --count-per-app 120 --lang en --country my
```

This creates:
- `data/raw/ewallet_reviews_collected_raw.csv`
- `data/raw/ewallet_reviews_for_labeling.csv`

The second file is meant for manual labeling by the team using:
- `payment_failure`
- `account_access_issue`
- `transfer_issue`
- `security_concern`
- `feature_request`

Use the guide in:
- `data/raw/ewallet_review_labeling_guide.md`

## Notes

- The current 10,000-row dataset uses weak labels; use a manually labeled holdout set for trustworthy final evaluation.
- All three models should use the **same cleaned dataset** and **same train/test split** for fair comparison.
- The app now uses multiclass evaluation metrics to fit the review-classification task.
- SVM uses a calibrated linear classifier so the app can still show confidence scores.

## Suggested Final Framing

For the report, position the system as helping e-wallet teams:
- organize review complaints automatically
- identify recurring payment or transfer issues faster
- detect account access and security concerns
- group feature suggestions for product improvement
