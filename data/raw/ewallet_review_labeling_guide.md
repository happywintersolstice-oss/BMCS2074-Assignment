# E-Wallet Review Labeling Guide

Use this guide when filling the `label` column in `ewallet_reviews_for_labeling.csv`.

## Recommended Collection Target

Try to collect at least:
- `60` reviews for `payment_failure`
- `60` reviews for `account_access_issue`
- `60` reviews for `transfer_issue`
- `60` reviews for `security_concern`
- `60` reviews for `feature_request`

This gives you a practical starting target of around `300` labeled reviews.
If possible, aim for `80 to 100` reviews per class for stronger training stability.

## Allowed Labels

### `payment_failure`
Use this when the main issue is:
- payment declined
- scan and pay failed
- top up failed
- bill payment unsuccessful
- checkout or merchant payment not going through

### `account_access_issue`
Use this when the main issue is:
- cannot log in
- OTP not received
- account locked
- identity verification failed
- password reset still does not allow access

### `transfer_issue`
Use this when the main issue is:
- bank transfer failed
- wallet transfer delayed
- recipient did not receive money
- transfer still pending
- cash out or withdrawal problem

### `security_concern`
Use this when the main issue is:
- suspicious transaction
- unauthorized access concern
- privacy concern
- account safety complaint
- weak authentication complaint

### `feature_request`
Use this when the main issue is:
- asking for a new feature
- requesting an improvement
- wanting a better dashboard or report
- asking for new login methods, budgeting tools, or export tools

## Team Rules

- Assign **one main label only**.
- If a review mentions multiple issues, choose the strongest or earliest major issue.
- If a review is too vague, skip it or leave a note in `label_notes`.
- If a review is only praise with no clear issue or request, skip it for now because the current model uses issue categories only.
