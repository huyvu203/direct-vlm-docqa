"""Accuracy and latency metrics for document question answering."""

import math
import re
import unicodedata
from collections import Counter
from typing import Sequence


def normalize_text(text: str) -> str:
    """Normalize case, Unicode, punctuation, and whitespace for comparison."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"[^\w\s%.,+-]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """Return 1 when normalized answers match exactly, otherwise 0."""
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Calculate F1 from normalized token overlap."""
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def best_exact_match(prediction: str, references: Sequence[str]) -> float:
    """Return the best Exact Match score across accepted answers."""
    if not references:
        raise ValueError("At least one reference answer is required")
    return max(exact_match(prediction, reference) for reference in references)


def best_token_f1(prediction: str, references: Sequence[str]) -> float:
    """Return the best token F1 score across accepted answers."""
    if not references:
        raise ValueError("At least one reference answer is required")
    return max(token_f1(prediction, reference) for reference in references)


def percentile(values: Sequence[float], percent: float) -> float:
    """Calculate a nearest-rank percentile."""
    if not 0 <= percent <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    if not values:
        return 0.0

    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return float(ordered[rank - 1])
