"""Evaluate VLM predictions against prepared DocumentVQA answers."""

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import typer

from .jsonl import read_jsonl
from .metrics import best_exact_match, best_token_f1, percentile

app = typer.Typer()


def index_by_question_id(
    records: Sequence[Mapping[str, Any]],
    label: str,
) -> Dict[str, Mapping[str, Any]]:
    """Index records and reject missing or duplicate question IDs."""
    indexed: Dict[str, Mapping[str, Any]] = {}
    duplicates = set()

    for position, record in enumerate(records, start=1):
        if "question_id" not in record:
            raise ValueError(f"{label} record {position} is missing question_id")
        question_id = str(record["question_id"])
        if question_id in indexed:
            duplicates.add(question_id)
        indexed[question_id] = record

    if duplicates:
        raise ValueError(f"Duplicate {label} question IDs: {sorted(duplicates)}")
    return indexed


def evaluate_records(
    questions: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Calculate accuracy, coverage, latency, token, and reliability metrics."""
    question_index = index_by_question_id(questions, "question")
    prediction_index = index_by_question_id(predictions, "prediction")

    exact_scores: List[float] = []
    f1_scores: List[float] = []
    latencies: List[float] = []
    matched_count = 0
    error_count = 0
    incomplete_count = 0
    retry_questions = 0
    retry_attempts = 0
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0

    for question_id, question in question_index.items():
        references = question.get("answers")
        if not isinstance(references, list) or not references:
            raise ValueError(f"Question {question_id} has no reference answers")

        prediction = prediction_index.get(question_id)
        if prediction is None:
            exact_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        matched_count += 1
        latency = float(prediction.get("latency_ms", 0) or 0)
        if latency > 0:
            latencies.append(latency)

        input_tokens += int(prediction.get("input_tokens", 0) or 0)
        output_tokens += int(prediction.get("output_tokens", 0) or 0)
        reasoning_tokens += int(prediction.get("reasoning_tokens", 0) or 0)

        attempt_count = int(prediction.get("attempt_count", 0) or 0)
        if attempt_count > 1:
            retry_questions += 1
            retry_attempts += attempt_count - 1

        if prediction.get("response_status") == "incomplete":
            incomplete_count += 1

        if prediction.get("error") is not None:
            error_count += 1
            exact_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        answer = prediction.get("answer")
        if not isinstance(answer, str):
            raise ValueError(f"Prediction {question_id} has no string answer")

        exact_scores.append(best_exact_match(answer, references))
        f1_scores.append(best_token_f1(answer, references))

    total_questions = len(question_index)
    missing_count = total_questions - matched_count
    extra_count = len(set(prediction_index).difference(question_index))
    successful_count = matched_count - error_count
    failure_count = missing_count + error_count
    token_denominator = matched_count or 1

    return {
        "dataset": {
            "questions": total_questions,
            "predictions": len(prediction_index),
            "matched": matched_count,
            "successful": successful_count,
            "missing": missing_count,
            "extra": extra_count,
        },
        "accuracy": {
            "exact_match": (
                sum(exact_scores) / total_questions if total_questions else 0.0
            ),
            "token_f1": (
                sum(f1_scores) / total_questions if total_questions else 0.0
            ),
        },
        "latency": {
            "mean_ms": statistics.mean(latencies) if latencies else 0.0,
            "median_ms": statistics.median(latencies) if latencies else 0.0,
            "p95_ms": percentile(latencies, 95),
        },
        "tokens": {
            "total_input": input_tokens,
            "total_output": output_tokens,
            "total_reasoning": reasoning_tokens,
            "average_input": input_tokens / token_denominator,
            "average_output": output_tokens / token_denominator,
            "average_reasoning": reasoning_tokens / token_denominator,
        },
        "reliability": {
            "retry_questions": retry_questions,
            "retry_attempts": retry_attempts,
            "retry_rate": (
                retry_questions / total_questions if total_questions else 0.0
            ),
            "error_count": error_count,
            "error_rate": error_count / total_questions if total_questions else 0.0,
            "incomplete_count": incomplete_count,
            "missing_predictions": missing_count,
            "failure_count": failure_count,
            "failure_rate": (
                failure_count / total_questions if total_questions else 0.0
            ),
        },
    }


def evaluate_files(
    questions_path: Path,
    predictions_path: Path,
    output_path: Path,
    max_questions: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate JSONL files and write aggregate metrics as JSON."""
    if max_questions is not None and max_questions <= 0:
        raise ValueError("max_questions must be greater than zero")

    questions = read_jsonl(questions_path)
    predictions = read_jsonl(predictions_path)
    if max_questions is not None:
        questions = questions[:max_questions]
        selected_ids = {str(question["question_id"]) for question in questions}
        predictions = [
            prediction
            for prediction in predictions
            if str(prediction.get("question_id")) in selected_ids
        ]

    metrics = evaluate_records(questions, predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def print_summary(metrics: Mapping[str, Any], output_path: Path) -> None:
    """Print the primary evaluation results."""
    dataset = metrics["dataset"]
    accuracy = metrics["accuracy"]
    reliability = metrics["reliability"]

    typer.echo(f"Questions: {dataset['questions']}; matched: {dataset['matched']}")
    typer.echo(f"Exact Match: {accuracy['exact_match']:.2%}")
    typer.echo(f"Token F1: {accuracy['token_f1']:.2%}")
    typer.echo(
        f"Retries: {reliability['retry_questions']}; "
        f"errors: {reliability['error_count']}; "
        f"missing: {reliability['missing_predictions']}"
    )
    typer.echo(f"Metrics saved to {output_path}")


@app.command()
def main(
    questions: Path = typer.Option(
        Path("data/processed/questions/dev.jsonl"),
        help="Prepared questions JSONL file",
    ),
    predictions: Path = typer.Option(
        Path("outputs/predictions/dev.jsonl"),
        help="Prediction JSONL file",
    ),
    output: Path = typer.Option(
        Path("results/dev_metrics.json"),
        help="Metrics JSON file",
    ),
    max_questions: Optional[int] = typer.Option(
        None, help="Evaluate only the first questions"
    ),
) -> None:
    """Evaluate DocumentVQA predictions and operational measurements."""
    metrics = evaluate_files(questions, predictions, output, max_questions)
    print_summary(metrics, output)


if __name__ == "__main__":
    app()
