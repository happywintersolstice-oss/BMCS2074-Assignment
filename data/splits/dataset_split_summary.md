# Dataset Split Summary

This file was generated from `data/raw/ewallet_reviews_final.csv`.

## Split Sizes

- Source rows: 5967
- Training rows: 4367
- Testing rows: 800
- Presentation rows: 800

## Training Label Counts

| Label | Rows |
|---|---:|
| `account_access_issue` | 1030 |
| `feature_request` | 966 |
| `payment_failure` | 964 |
| `security_concern` | 158 |
| `transfer_issue` | 1249 |

## Testing Label Counts

| Label | Rows |
|---|---:|
| `account_access_issue` | 188 |
| `feature_request` | 177 |
| `payment_failure` | 177 |
| `security_concern` | 29 |
| `transfer_issue` | 229 |

## Presentation Label Counts

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
- `ewallet_reviews_presentation.csv`: keep as a separate demo set for presentation examples.