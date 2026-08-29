"""Validation for structured VLM answers."""

import json
from typing import Mapping, Tuple


class ResponseValidationError(ValueError):
    """Raised when a model response is not a usable answer."""


def parse_answer(response_text: str) -> Tuple[str, float]:
    """Parse and validate an ``answer`` and ``confidence`` JSON object."""
    try:
        response = json.loads(response_text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ResponseValidationError("Response is not valid JSON") from error

    if not isinstance(response, Mapping):
        raise ResponseValidationError("Response must be a JSON object")

    answer = response.get("answer")
    confidence = response.get("confidence")
    if not isinstance(answer, str) or not answer.strip():
        raise ResponseValidationError("Response answer must be a non-empty string")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ResponseValidationError("Response confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ResponseValidationError("Response confidence must be between 0 and 1")

    return answer.strip(), float(confidence)
