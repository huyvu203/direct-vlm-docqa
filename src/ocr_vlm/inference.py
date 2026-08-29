"""Run direct VLM inference over prepared document questions."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, cast

import typer
from tqdm import tqdm

from .jsonl import append_jsonl, read_jsonl, write_jsonl
from .prompts import build_user_prompt
from .retry import ResponseValidationError, parse_answer
from .schemas import PredictionRecord, QuestionRecord
from .vlm_client import VLMClient

app = typer.Typer()

QUESTION_FIELDS = {"page_id", "question_id", "question", "image_path"}


def validate_questions(records: List[Dict[str, Any]]) -> List[QuestionRecord]:
    """Check the fields inference needs and return typed question records."""
    for index, record in enumerate(records, start=1):
        missing = QUESTION_FIELDS.difference(record)
        if missing:
            raise ValueError(f"Question record {index} is missing fields: {sorted(missing)}")
    return cast(List[QuestionRecord], records)


def error_prediction(
    question: QuestionRecord,
    client: VLMClient,
    error: str,
    attempt_count: int = 0,
    latency_ms: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
    response_status: Optional[str] = None,
    incomplete_reason: Optional[str] = None,
) -> PredictionRecord:
    """Create a prediction record for a question that could not be answered."""
    return {
        "page_id": question["page_id"],
        "question_id": question["question_id"],
        "answer": "unknown",
        "confidence": 0.0,
        "model": client.model,
        "reasoning_effort": client.reasoning_effort,
        "max_output_tokens": client.max_output_tokens,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "attempt_count": attempt_count,
        "response_status": response_status,
        "incomplete_reason": incomplete_reason,
        "error": error,
    }


def infer_question(question: QuestionRecord, client: VLMClient) -> PredictionRecord:
    """Answer one question, retrying once when response validation fails."""
    image_path = Path(question["image_path"])
    if not image_path.is_file():
        return error_prediction(question, client, f"Image not found: {image_path}")

    latency_ms = 0
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    validation_error = ""
    response_status: Optional[str] = None
    incomplete_reason: Optional[str] = None

    for attempt in (1, 2):
        prompt = build_user_prompt(question["question"], corrective=attempt == 2)
        try:
            response = client.generate(image_path, prompt)
        except Exception as error:
            return error_prediction(
                question,
                client,
                f"{type(error).__name__}: {error}",
                attempt,
                latency_ms,
                input_tokens,
                output_tokens,
                reasoning_tokens,
                response_status,
                incomplete_reason,
            )

        latency_ms += response["latency_ms"]
        input_tokens += response["input_tokens"]
        output_tokens += response["output_tokens"]
        reasoning_tokens += response["reasoning_tokens"]
        response_status = response["status"]
        incomplete_reason = response["incomplete_reason"]

        if response_status != "completed":
            validation_error = incomplete_reason or response_status
            continue

        try:
            answer, confidence = parse_answer(response["text"])
        except ResponseValidationError as error:
            validation_error = str(error)
            continue

        return {
            "page_id": question["page_id"],
            "question_id": question["question_id"],
            "answer": answer,
            "confidence": confidence,
            "model": client.model,
            "reasoning_effort": client.reasoning_effort,
            "max_output_tokens": client.max_output_tokens,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "attempt_count": attempt,
            "response_status": response_status,
            "incomplete_reason": incomplete_reason,
            "error": None,
        }

    if response_status == "incomplete":
        error_message = f"Model response incomplete after 2 attempts: {validation_error}"
    else:
        error_message = f"Invalid model response after 2 attempts: {validation_error}"

    return error_prediction(
        question,
        client,
        error_message,
        2,
        latency_ms,
        input_tokens,
        output_tokens,
        reasoning_tokens,
        response_status,
        incomplete_reason,
    )


def run_inference(
    questions_path: Path,
    output_path: Path,
    model: str = "gpt-5",
    max_output_tokens: int = 1000,
    reasoning_effort: str = "minimal",
    max_questions: Optional[int] = None,
    resume: bool = True,
    client: Optional[VLMClient] = None,
) -> None:
    """Generate predictions and save each result immediately as JSONL."""
    if max_questions is not None and max_questions <= 0:
        raise ValueError("max_questions must be greater than zero")

    questions = validate_questions(read_jsonl(questions_path))
    if max_questions is not None:
        questions = questions[:max_questions]

    completed_ids: Set[str] = set()
    if resume and output_path.exists():
        selected_ids = {question["question_id"] for question in questions}
        existing_predictions = read_jsonl(output_path)
        kept_predictions = []
        retried_error_ids: Set[str] = set()

        for record in existing_predictions:
            question_id = str(record["question_id"])
            if question_id in selected_ids and record.get("error") is not None:
                retried_error_ids.add(question_id)
                continue
            kept_predictions.append(record)
            if question_id in selected_ids:
                completed_ids.add(question_id)

        if retried_error_ids:
            write_jsonl(output_path, kept_predictions)
            typer.echo(f"Retrying {len(retried_error_ids)} previous error records")
    else:
        write_jsonl(output_path, [])

    pending = [
        question
        for question in questions
        if question["question_id"] not in completed_ids
    ]
    if not pending:
        typer.echo(f"No pending questions; {len(completed_ids)} already completed")
        return

    inference_client = client or VLMClient(
        model,
        max_output_tokens,
        reasoning_effort,
    )
    retries = 0
    errors = 0
    for question in tqdm(pending, desc="Answering questions"):
        prediction = infer_question(question, inference_client)
        append_jsonl(output_path, prediction)
        retries += int(prediction["attempt_count"] > 1)
        errors += int(prediction["error"] is not None)

    typer.echo(f"Processed {len(pending)} questions with {inference_client.model}")
    typer.echo(f"Retries: {retries}; errors: {errors}")
    typer.echo(f"Predictions saved to {output_path}")


@app.command()
def main(
    questions: Path = typer.Option(
        Path("data/processed/questions/dev.jsonl"),
        help="Prepared questions JSONL file",
    ),
    output: Path = typer.Option(
        Path("outputs/predictions/dev.jsonl"),
        help="Prediction JSONL file",
    ),
    model: str = typer.Option("gpt-5", help="OpenAI vision-capable model ID"),
    max_output_tokens: int = typer.Option(1000, help="Maximum tokens per response"),
    reasoning_effort: str = typer.Option(
        "minimal", help="GPT-5 reasoning effort"
    ),
    max_questions: Optional[int] = typer.Option(
        None, help="Limit questions for a smoke test"
    ),
    resume: bool = typer.Option(
        True, "--resume/--overwrite", help="Resume an existing output file"
    ),
) -> None:
    """Answer prepared DocumentVQA questions directly from their images."""
    run_inference(
        questions,
        output,
        model,
        max_output_tokens,
        reasoning_effort,
        max_questions,
        resume,
    )


if __name__ == "__main__":
    app()
