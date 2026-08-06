# Dataset Splits

This folder stores the separate datasets used for:

- training
- held-out testing

How to regenerate the split files:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python313\python.exe scripts/create_dataset_splits.py
```

Expected outputs:

- `ewallet_reviews_training.csv`
- `ewallet_reviews_testing_manual.csv`
- `dataset_split_summary.md`
