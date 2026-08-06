# Strong Dataset Acquisition Plan

This plan is for building a stronger final dataset for:

- model training
- formal testing

## Final Targets

- training dataset: `10,000` labeled rows
- testing dataset: `800` labeled rows
Total final labeled target:
- `10,800` rows

Recommended raw collection target before labeling:
- `14,000 to 18,000` raw rows

Reason:
- some rows will be duplicates
- some rows will be too vague
- some rows will not fit your 5 labels cleanly

## Label Targets

Try to keep the final labeled dataset as balanced as possible:

- `payment_failure`: `2,320`
- `account_access_issue`: `2,320`
- `transfer_issue`: `2,320`
- `security_concern`: `2,320`
- `feature_request`: `2,320`

If exact balance is too hard, at least protect `security_concern` because it is usually the smallest class.

## Pipeline

### 1. Collect raw reviews

Use:
- `scripts/collect_google_play_reviews.py`
- `data/raw/google_play_targets.csv`

Suggested first collection command:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts/collect_google_play_reviews.py --count-per-app 1500 --lang en --country my --sort newest
```

Then repeat with different countries or languages if needed:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts/collect_google_play_reviews.py --count-per-app 1500 --lang en --country sg --sort newest --raw-output data/raw/ewallet_reviews_raw_sg.csv --labeling-output data/raw/ewallet_reviews_for_labeling_sg.csv
```

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts/collect_google_play_reviews.py --count-per-app 1500 --lang en --country ph --sort newest --raw-output data/raw/ewallet_reviews_raw_ph.csv --labeling-output data/raw/ewallet_reviews_for_labeling_ph.csv
```

### 2. Split the labeling work manually

Recommended approach:

- open the collected labeling CSV
- divide rows across team members in Google Sheets or Excel
- keep the same shared label taxonomy for all members

### 3. Manual labeling

Rules:

- label only one main issue per review
- skip praise-only rows
- skip vague rows with no clear operational issue
- use the shared taxonomy only

Shared labels:

- `payment_failure`
- `account_access_issue`
- `transfer_issue`
- `security_concern`
- `feature_request`

### 4. Merge approved labeled rows into the project dataset

Recommended approach:

- keep one clean master labeled CSV as the project source
- append only reviewed and approved rows
- remove duplicate `text` + `label` pairs before saving

### 5. Create final splits

Use:
- `scripts/create_dataset_splits.py`

When your master labeled dataset becomes the new official dataset, replace `data/raw/ewallet_reviews_final.csv` with the stronger cleaned master file, then rerun:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts/create_dataset_splits.py
```

## Recommended Team Workflow

Member responsibilities:

- Member 1: collection monitoring + Naive Bayes
- Member 2: labeling quality control + SVM
- Member 3: dataset merging/report evidence + Logistic Regression

Weekly workflow:

1. collect raw reviews
2. split into member batches
3. label rows
4. combine labeled rows
5. rebuild training/testing datasets
6. retrain models
7. record metric changes

## Current Status

Current available split from existing data:

- training: `2597`
- testing: `800`
This is usable now, but it is still below the stronger final target of `10,000 + 800`.
