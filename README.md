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
  - cross-validation summary on the training split
  - per-class held-out test results
  - confusion matrix for each model

- `Dataset Preview`
  - inspect the current labeled review rows used for training

## Project Structure

- [`app.py`](C:\Users\User\OneDrive\Documents\BMCS2074 Assignment\app.py)
  Streamlit entry point

- `data/raw/ewallet_reviews_final.csv`
  Current master labeled dataset used as the source for splitting

- `data/splits/ewallet_reviews_training.csv`
  Training dataset for model development

- `data/splits/ewallet_reviews_testing_manual.csv`
  Held-out testing dataset for final evaluation

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

The project now has one master dataset and two role-based split datasets:

- `data/raw/ewallet_reviews_final.csv`
  - the current master labeled source file
- `data/splits/ewallet_reviews_training.csv`
  - for model training and development
  - this is now the dataset the app trains from
- `data/splits/ewallet_reviews_testing_manual.csv`
  - for held-out testing and report results

The master file was created by filtering and relabeling a noisier weak-labeled source with stricter rules so the labels are more reliable.

The current app training flow works like this:

- `data/raw/ewallet_reviews_final.csv`
  - master source file
- `data/splits/ewallet_reviews_training.csv`
  - the only file used for model fitting and cross-validation
- `data/splits/ewallet_reviews_testing_manual.csv`
  - the only file used for final held-out evaluation in the app and report

The app balances classes only inside the training flow before fitting so the classifier does not overlearn the larger classes.
The held-out testing file is not balanced or mixed back into training.

Because the current training split is still imbalanced, balancing will temporarily downsample the larger classes to match the smallest class. This keeps evaluation fair, but it also means the current effective training size is smaller than the raw training CSV until more `security_concern` rows are collected.

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

The review collection script can create raw review files for manual labeling when needed.

The labeling file is meant for manual annotation by the team using:
- `payment_failure`
- `account_access_issue`
- `transfer_issue`
- `security_concern`
- `feature_request`

Use the guide in:
- `data/raw/ewallet_review_labeling_guide.md`

## Notes

- The active dataset is `ewallet_reviews_final.csv`, not the earlier weak-labeled source.
- All three models use the **same training split** for fitting and the **same held-out testing split** for final comparison.
- The app now uses multiclass evaluation metrics to fit the review-classification task.
- SVM uses a calibrated linear classifier so the app can still show confidence scores.
- The comparison page now includes per-class metrics and a confusion matrix, which you can reuse in the report discussion section.

## Suggested Final Framing

For the report, position the system as helping e-wallet teams:
- organize review complaints automatically
- identify recurring payment or transfer issues faster
- detect account access and security concerns
- group feature suggestions for product improvement
