"""
Review and relabel the weakly labeled e-wallet review dataset using stricter rules.

What this script does:
- reads the large weak-label dataset
- applies stricter keyword and phrase rules for each issue category
- keeps only high-confidence rows for training
- sends unclear rows to a manual review queue

How it works:
- review text is normalized into lowercase plain text
- each label gets a score based on matched phrases and tokens
- rows are only kept when one label clearly wins
- positive or resolved-sounding reviews are filtered out where possible
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd


SOURCE_DATASET_PATH = Path("data/raw/ewallet_reviews_for_labeling.csv")
RELABELED_OUTPUT_PATH = Path("data/raw/ewallet_reviews_relabeled_strict.csv")
REVIEW_QUEUE_OUTPUT_PATH = Path("data/raw/ewallet_reviews_manual_review_queue.csv")

LABEL_RULES = {
    "payment_failure": {
        "phrases": [
            "payment failed",
            "payment unsuccessful",
            "unable to process transaction",
            "transaction unsuccessful",
            "charged twice",
            "double charge",
            "bill payment",
            "scan qr",
            "qr code",
            "qr payment",
            "purchase failed",
            "pay bills",
            "bills payment",
            "transaction failure",
            "cannot pay",
            "cant pay",
            "pay online",
            "checkout",
            "refund",
            "receipt",
        ],
        "tokens": {"payment", "transaction", "charged", "bill", "bills", "qr", "merchant", "checkout", "pay"},
        "exclude_phrases": [
            "fixed the login issue",
            "good enough",
            "security comes first",
        ],
    },
    "account_access_issue": {
        "phrases": [
            "cannot log in",
            "cant log in",
            "can t log in",
            "login issue",
            "log in problem",
            "unable to login",
            "cannot login",
            "cant login",
            "not letting me login",
            "cannot access my gcash account",
            "account locked",
            "locked account",
            "otp not received",
            "verification failed",
            "reset my",
            "password",
            "mpin",
            "authenticate me",
        ],
        "tokens": {"login", "otp", "mpin", "password", "verify", "verification", "authenticate", "locked"},
        "exclude_phrases": [
            "fixed the login issue",
            "thanks for the fast update",
            "excellent support",
            "resolved",
        ],
    },
    "transfer_issue": {
        "phrases": [
            "send money",
            "cash in",
            "cash out",
            "bank transfer",
            "fund transfer",
            "transfer failed",
            "transfer delay",
            "transfer pending",
            "load to gcash",
            "paypal to gcash",
            "recipient never received",
            "recipient did not receive",
            "didnt receive",
            "withdrawal",
            "cashout",
            "cashin",
        ],
        "tokens": {"transfer", "send", "cash", "withdraw", "withdrawal", "bank", "recipient", "pending", "cashout", "cashin"},
        "exclude_phrases": [
            "would be nice",
            "please add",
            "option or feature to transfer",
        ],
    },
    "security_concern": {
        "phrases": [
            "security concern",
            "security issue",
            "privacy concern",
            "unauthorized access",
            "suspicious activity",
            "fraud claim",
            "fraudulent",
            "details of scammer",
            "mastercard is scam",
            "not safe",
            "weak authentication",
            "no protection",
        ],
        "tokens": {"scam", "fraud", "security", "privacy", "unsafe", "hacker", "hack", "paranoid", "protected", "protection", "suspicious"},
        "exclude_phrases": [
            "i just hope",
            "works as stated",
            "easy to use",
            "good enough",
            "hope no scam",
        ],
    },
    "feature_request": {
        "phrases": [
            "please add",
            "i hope",
            "would be nice",
            "feature request",
            "suggestion",
            "kindly include",
            "kindly enhance",
            "bring back",
            "allow us to",
            "allow the",
            "can you add",
            "please include",
            "please allow",
            "dark mode",
            "student id",
            "material design",
            "option to",
        ],
        "tokens": {"add", "feature", "suggestion", "suggest", "include", "option", "dark", "mode", "allow"},
        "exclude_phrases": [
            "cannot",
            "cant",
            "failed",
            "error",
            "unable",
            "not working",
        ],
    },
}

POSITIVE_ONLY_PHRASES = [
    "easy to use",
    "user friendly",
    "works as stated",
    "good app",
    "reliable app",
    "very useful",
    "thank you",
]


def normalize_text(text: object) -> str:
    """
    Convert raw text into a simple searchable version for rule matching.
    """
    normalized = str(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def score_label(normalized_text: str, rule: dict[str, list[str] | set[str]]) -> int:
    """
    Give one label a score based on phrase and token matches.
    """
    score = 0
    for phrase in rule["phrases"]:
        if phrase in normalized_text:
            score += 2

    tokens = set(normalized_text.split())
    for token in rule["tokens"]:
        if token in tokens:
            score += 1

    return score


def should_exclude_positive_or_resolved_text(normalized_text: str, predicted_label: str) -> bool:
    """
    Remove rows that sound like praise, resolved issues, or non-actionable mentions.
    """
    if any(phrase in normalized_text for phrase in POSITIVE_ONLY_PHRASES):
        if predicted_label != "feature_request":
            return True

    exclude_phrases = LABEL_RULES[predicted_label]["exclude_phrases"]
    return any(phrase in normalized_text for phrase in exclude_phrases)


def relabel_row(text: object) -> tuple[str | None, int, str]:
    """
    Suggest one strict label for a review or send it to manual review.
    """
    normalized_text = normalize_text(text)
    scores = Counter()

    for label, rule in LABEL_RULES.items():
        scores[label] = score_label(normalized_text, rule)

    strongest = scores.most_common(2)
    if not strongest or strongest[0][1] < 2:
        return None, strongest[0][1] if strongest else 0, "too_weak"

    if len(strongest) > 1 and strongest[0][1] < strongest[1][1] + 2:
        return None, strongest[0][1], "ambiguous_overlap"

    predicted_label = strongest[0][0]
    if should_exclude_positive_or_resolved_text(normalized_text, predicted_label):
        return None, strongest[0][1], "positive_or_resolved"

    return predicted_label, strongest[0][1], "high_confidence_rule_match"


def main() -> None:
    """
    Create a stricter relabeled dataset and a manual review queue.
    """
    if not SOURCE_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_DATASET_PATH}. "
            "Run the review collection script first or update the source path."
        )

    dataframe = pd.read_csv(SOURCE_DATASET_PATH)
    if "text" not in dataframe.columns:
        raise ValueError("Source dataset must contain a text column.")

    relabel_results = dataframe["text"].apply(relabel_row)

    dataframe["suggested_label"] = relabel_results.map(lambda item: item[0])
    dataframe["rule_score"] = relabel_results.map(lambda item: item[1])
    dataframe["review_status"] = relabel_results.map(lambda item: item[2])

    reviewed_dataframe = dataframe[dataframe["suggested_label"].notna()].copy()
    reviewed_dataframe["label"] = reviewed_dataframe["suggested_label"]
    reviewed_dataframe["label_method"] = "strict_rule_relabel"

    manual_review_queue = dataframe[dataframe["suggested_label"].isna()].copy()

    RELABELED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reviewed_dataframe.to_csv(RELABELED_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    manual_review_queue.to_csv(REVIEW_QUEUE_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Strict relabeling complete.")
    print(f"Relabeled dataset saved to: {RELABELED_OUTPUT_PATH}")
    print(f"Manual review queue saved to: {REVIEW_QUEUE_OUTPUT_PATH}")
    print(f"High-confidence rows kept: {len(reviewed_dataframe)}")
    print(f"Rows sent to manual review: {len(manual_review_queue)}")
    print("Kept rows by label:")
    print(reviewed_dataframe["label"].value_counts().to_string())
    print("Manual review reasons:")
    print(manual_review_queue["review_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
