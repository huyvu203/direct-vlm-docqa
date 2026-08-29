"""Offline tests for DocQA metrics and evaluation."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ocr_vlm.evaluation import evaluate_files, evaluate_records
from ocr_vlm.jsonl import write_jsonl
from ocr_vlm.metrics import (
    best_exact_match,
    best_token_f1,
    exact_match,
    normalize_text,
    percentile,
    token_f1,
)


def make_question(question_id, answers):
    return {
        "page_id": f"documentvqa_{question_id}",
        "question_id": str(question_id),
        "question": "What is shown?",
        "answers": answers,
        "question_types": ["extractive"],
        "image_path": f"images/{question_id}.png",
    }


def make_prediction(
    question_id,
    answer,
    *,
    error=None,
    latency_ms=100,
    input_tokens=10,
    output_tokens=5,
    reasoning_tokens=1,
    attempt_count=1,
    response_status="completed",
    incomplete_reason=None,
):
    return {
        "page_id": f"documentvqa_{question_id}",
        "question_id": str(question_id),
        "answer": answer,
        "confidence": 0.9,
        "model": "test-model",
        "reasoning_effort": "minimal",
        "max_output_tokens": 1000,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "attempt_count": attempt_count,
        "response_status": response_status,
        "incomplete_reason": incomplete_reason,
        "error": error,
    }


class MetricTests(unittest.TestCase):
    def test_normalization_exact_match_and_token_f1(self):
        self.assertEqual(normalize_text("  Hello!   WORLD  "), "hello world")
        self.assertEqual(exact_match("Let Yourself Grow!", "let yourself grow"), 1.0)
        self.assertAlmostEqual(token_f1("blue car", "car"), 2 / 3)

    def test_best_scores_use_all_reference_answers(self):
        references = ["twenty-five dollars", "$25.00", "25"]

        self.assertEqual(best_exact_match("$25.00", references), 1.0)
        self.assertEqual(best_token_f1("$25.00", references), 1.0)

    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(percentile([100, 200, 300, 400], 95), 400.0)
        self.assertEqual(percentile([], 95), 0.0)


class EvaluationTests(unittest.TestCase):
    def test_errors_and_missing_predictions_score_zero(self):
        questions = [
            make_question("1", ["correct"]),
            make_question("2", ["answer"]),
            make_question("3", ["missing"]),
        ]
        predictions = [
            make_prediction("1", "correct", latency_ms=100),
            make_prediction(
                "2",
                "unknown",
                error="max output tokens",
                latency_ms=300,
                input_tokens=20,
                output_tokens=15,
                reasoning_tokens=10,
                attempt_count=2,
                response_status="incomplete",
                incomplete_reason="max_output_tokens",
            ),
        ]

        metrics = evaluate_records(questions, predictions)

        self.assertEqual(metrics["dataset"], {
            "questions": 3,
            "predictions": 2,
            "matched": 2,
            "successful": 1,
            "missing": 1,
            "extra": 0,
        })
        self.assertAlmostEqual(metrics["accuracy"]["exact_match"], 1 / 3)
        self.assertAlmostEqual(metrics["accuracy"]["token_f1"], 1 / 3)
        self.assertEqual(metrics["latency"]["mean_ms"], 200)
        self.assertEqual(metrics["latency"]["median_ms"], 200)
        self.assertEqual(metrics["latency"]["p95_ms"], 300)
        self.assertEqual(metrics["tokens"]["total_input"], 30)
        self.assertEqual(metrics["tokens"]["total_output"], 20)
        self.assertEqual(metrics["tokens"]["total_reasoning"], 11)
        self.assertEqual(metrics["tokens"]["average_input"], 15)
        self.assertEqual(metrics["reliability"]["retry_questions"], 1)
        self.assertEqual(metrics["reliability"]["retry_attempts"], 1)
        self.assertAlmostEqual(metrics["reliability"]["retry_rate"], 1 / 3)
        self.assertEqual(metrics["reliability"]["error_count"], 1)
        self.assertEqual(metrics["reliability"]["incomplete_count"], 1)
        self.assertEqual(metrics["reliability"]["missing_predictions"], 1)
        self.assertEqual(metrics["reliability"]["failure_count"], 2)

    def test_duplicate_predictions_are_rejected(self):
        questions = [make_question("1", ["answer"])]
        predictions = [
            make_prediction("1", "answer"),
            make_prediction("1", "answer"),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate prediction question IDs"):
            evaluate_records(questions, predictions)

    def test_evaluate_files_limits_questions_and_writes_json(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            questions_path = root / "questions.jsonl"
            predictions_path = root / "predictions.jsonl"
            output_path = root / "metrics.json"
            write_jsonl(
                questions_path,
                [make_question("1", ["yes"]), make_question("2", ["no"])],
            )
            write_jsonl(
                predictions_path,
                [make_prediction("1", "yes"), make_prediction("2", "wrong")],
            )

            metrics = evaluate_files(
                questions_path,
                predictions_path,
                output_path,
                max_questions=1,
            )
            saved_metrics = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(metrics["dataset"]["questions"], 1)
        self.assertEqual(metrics["dataset"]["predictions"], 1)
        self.assertEqual(metrics["dataset"]["extra"], 0)
        self.assertEqual(metrics["accuracy"]["exact_match"], 1.0)
        self.assertEqual(saved_metrics, metrics)


if __name__ == "__main__":
    unittest.main()
