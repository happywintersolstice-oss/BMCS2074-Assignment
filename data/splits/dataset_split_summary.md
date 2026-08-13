# Dataset Split Summary

This file was generated from `data/raw/ewallet_reviews_final.csv`.

## Split Sizes

- Source rows: 5967
- Training rows: 5167
- Testing rows: 800

## Training Label Counts

| Label | Rows |
|---|---:|
| `account_access_issue` | 1218 |
| `feature_request` | 1143 |
| `payment_failure` | 1141 |
| `security_concern` | 187 |
| `transfer_issue` | 1478 |

## Testing Label Counts

| Label | Rows |
|---|---:|
| `account_access_issue` | 189 |
| `feature_request` | 177 |
| `payment_failure` | 176 |
| `security_concern` | 29 |
| `transfer_issue` | 229 |

## Intended Usage

- `ewallet_reviews_training.csv`: use for model development and training.
- `ewallet_reviews_testing_manual.csv`: keep as a held-out evaluation set for report results.