# Dataset Split Summary

This file was generated from `data/raw/ewallet_reviews_final.csv`.

## Split Sizes

- Source rows: 4197
- Training rows: 2597
- Testing rows: 800
- Presentation rows: 800

## Training Label Counts

| Label | Rows |
|---|---:|
| `account_access_issue` | 598 |
| `feature_request` | 665 |
| `payment_failure` | 410 |
| `security_concern` | 69 |
| `transfer_issue` | 855 |

## Testing Label Counts

| Label | Rows |
|---|---:|
| `account_access_issue` | 185 |
| `feature_request` | 205 |
| `payment_failure` | 126 |
| `security_concern` | 21 |
| `transfer_issue` | 263 |

## Presentation Label Counts

| Label | Rows |
|---|---:|
| `account_access_issue` | 184 |
| `feature_request` | 205 |
| `payment_failure` | 127 |
| `security_concern` | 21 |
| `transfer_issue` | 263 |

## Intended Usage

- `ewallet_reviews_training.csv`: use for model development and training.
- `ewallet_reviews_testing_manual.csv`: keep as a held-out evaluation set for report results.
- `ewallet_reviews_presentation.csv`: keep as a separate demo set for presentation examples.