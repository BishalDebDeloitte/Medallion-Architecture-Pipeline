"""Detect sensitive columns and redact their values before anything is shown to an AI model.

Everything the agents see about the data (sample values in profiles, previews and query
results) passes through this module, so cards, bank details, ids, credentials and contact
details never leave the machine as examples. Detection uses the column name and, when
values are available, their shape (Luhn-valid card numbers, email addresses).
Redaction only affects what the model sees; the pipeline still processes the real values.
"""

import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

KINDS = ("financial", "identity", "credential", "contact")
_NAME_PATTERNS: dict[str, re.Pattern[str]] = {
    "financial": re.compile(
        r"(^|_)(credit_?card|debit_?card|card_?(no|num|number)|cvv|cvc|iban|swift|ifsc|routing|upi|vpa|"
        r"account_?(no|num|number)|bank_?acc(ount)?|acct_?(no|num|number))(_|$)"
    ),
    "identity": re.compile(
        r"(^|_)(ssn|social_?security|aadhaa?r|passport|national_?id|pan_?(no|number|card)|tax_?id|"
        r"driver_?licen[cs]e)(_|$)"
    ),
    "credential": re.compile(r"(^|_)(password|passwd|pwd|secret|api_?key|access_?token|auth_?token|otp)(_|$)"),
    "contact": re.compile(r"(^|_)(e_?mail|phone|mobile|msisdn|contact_?(no|num|number))(_|$)"),
}
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
SAMPLE_SIZE = 200
VALUE_MATCH_SHARE = 0.8


def normalise_name(name: str) -> str:
    """Lower-case a column name and join its words with underscores (``Phone 1`` -> ``phone_1``)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return re.sub(r"[^a-z0-9]+", "_", spaced.lower()).strip("_")


def _luhn_valid(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        number = int(char)
        if index % 2 == parity:
            number *= 2
            number -= 9 if number > 9 else 0
        total += number
    return total % 10 == 0


def _looks_like_card(value: str) -> bool:
    digits = re.sub(r"[ -]", "", value)
    return digits.isdigit() and 13 <= len(digits) <= 19 and _luhn_valid(digits)


def sensitivity_by_name(name: str) -> str | None:
    """The sensitivity kind suggested by a column name alone, or None."""
    normalised = normalise_name(name)
    for kind, pattern in _NAME_PATTERNS.items():
        if pattern.search(normalised):
            return kind
    return None


def classify_column(name: str, values: pd.Series | Iterable[Any] | None = None) -> str | None:
    """Return 'financial', 'identity', 'credential', 'contact' or None for a column.

    The name decides first; otherwise the first non-null values are checked for card
    numbers and email addresses.
    """
    by_name = sensitivity_by_name(name)
    if by_name:
        return by_name
    if values is None:
        return None
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    sample = [str(v).strip() for v in series.dropna().head(SAMPLE_SIZE).tolist() if str(v).strip()]
    if len(sample) < 5:
        return None
    if sum(_looks_like_card(v) for v in sample) / len(sample) >= VALUE_MATCH_SHARE:
        return "financial"
    if sum(bool(_EMAIL.match(v)) for v in sample) / len(sample) >= VALUE_MATCH_SHARE:
        return "contact"
    return None


def redaction_marker(kind: str) -> str:
    """The placeholder shown to the model instead of a sensitive value."""
    return f"[redacted: {kind}]"


def redact_samples(samples: list[str], kind: str | None) -> list[str]:
    """Replace example values of a sensitive column with a single placeholder."""
    return [redaction_marker(kind)] if kind and samples else samples


def sensitive_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map each sensitive column of a frame to its kind (name- and value-based)."""
    found = {}
    for column in frame.columns:
        kind = classify_column(str(column), frame[column])
        if kind:
            found[str(column)] = kind
    return found


def redacted_records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    """The first rows of a frame as JSON-ready records with sensitive columns masked."""
    kinds = sensitive_columns(frame.head(max(limit, SAMPLE_SIZE)))
    head = frame.head(limit).copy()
    for column, kind in kinds.items():
        head[column] = redaction_marker(kind)
    import json

    return json.loads(head.to_json(orient="records", date_format="iso"))
