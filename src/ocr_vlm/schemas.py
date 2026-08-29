"""Shared record contracts for prepared questions and model predictions."""

from typing import List, Optional, TypedDict


class QuestionRecord(TypedDict):
    """One prepared DocumentVQA question and its document image."""

    page_id: str
    question_id: str
    question: str
    answers: List[str]
    question_types: List[str]
    image_path: str


class ModelAnswer(TypedDict):
    """Answer and request metadata returned by a VLM client."""

    answer: str
    confidence: float
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    attempt_count: int
    error: Optional[str]


class PredictionRecord(ModelAnswer):
    """A model answer associated with its source question."""

    page_id: str
    question_id: str
