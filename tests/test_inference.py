"""Offline tests for the direct VLM inference pipeline."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from ocr_vlm.inference import infer_question, run_inference
from ocr_vlm.jsonl import read_jsonl, write_jsonl
from ocr_vlm.prompts import build_user_prompt
from ocr_vlm.retry import ResponseValidationError, parse_answer
from ocr_vlm.vlm_client import ANSWER_FORMAT, VLMClient


def make_question(image_path: Path, question_id: str = "10"):
    return {
        "page_id": "documentvqa_1",
        "question_id": question_id,
        "question": "What is the total?",
        "answers": ["42"],
        "question_types": ["extractive"],
        "image_path": str(image_path),
    }


def make_response(
    text: str,
    latency: int = 10,
    input_tokens: int = 20,
    reasoning_tokens: int = 3,
    status: str = "completed",
    incomplete_reason=None,
):
    return {
        "text": text,
        "latency_ms": latency,
        "input_tokens": input_tokens,
        "output_tokens": 5,
        "reasoning_tokens": reasoning_tokens,
        "status": status,
        "incomplete_reason": incomplete_reason,
    }


class FakeInferenceClient:
    def __init__(
        self,
        responses,
        model="test-model",
        reasoning_effort="minimal",
        max_output_tokens=1000,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.responses = list(responses)
        self.prompts = []

    def generate(self, image_path, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class PromptAndValidationTests(unittest.TestCase):
    def test_prompt_contains_the_trimmed_question(self):
        prompt = build_user_prompt("  What is shown?  ")

        self.assertIn("Question: What is shown?", prompt)
        self.assertNotIn("required format", prompt)
        self.assertIn("required format", build_user_prompt("Question", corrective=True))

    def test_parse_answer_validates_the_schema(self):
        self.assertEqual(
            parse_answer('{"answer": " 42 ", "confidence": 0.9}'),
            ("42", 0.9),
        )

        invalid_responses = [
            "not json",
            "[]",
            '{"answer": "", "confidence": 0.5}',
            '{"answer": "42", "confidence": true}',
            '{"answer": "42", "confidence": 1.5}',
        ]
        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(
                ResponseValidationError
            ):
                parse_answer(response)


class VLMClientTests(unittest.TestCase):
    def test_generate_builds_a_structured_multimodal_request(self):
        calls = []

        class FakeResponses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    output_text='{"answer": "42", "confidence": 0.9}',
                    status="completed",
                    incomplete_details=None,
                    usage=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=12,
                        output_tokens_details=SimpleNamespace(reasoning_tokens=7),
                    ),
                )

        sdk_client = SimpleNamespace(responses=FakeResponses())

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "document.png"
            image_path.write_bytes(b"image bytes")
            client = VLMClient(
                model="test-model",
                max_output_tokens=200,
                client=sdk_client,
            )

            result = client.generate(image_path, "Question: What is the total?")

        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 12)
        self.assertEqual(result["reasoning_tokens"], 7)
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["incomplete_reason"])
        self.assertEqual(len(calls), 1)
        request = calls[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["max_output_tokens"], 200)
        self.assertEqual(request["reasoning"], {"effort": "minimal"})
        self.assertFalse(request["store"])
        self.assertEqual(request["text"]["format"], ANSWER_FORMAT)
        content = request["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_image")
        self.assertTrue(content[0]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[0]["detail"], "high")
        self.assertEqual(content[1]["type"], "input_text")


class QuestionInferenceTests(unittest.TestCase):
    def test_invalid_response_is_retried_once_and_usage_is_accumulated(self):
        client = FakeInferenceClient(
            [
                make_response("not json", latency=10, input_tokens=20),
                make_response(
                    '{"answer": "42", "confidence": 0.8}',
                    latency=15,
                    input_tokens=25,
                ),
            ]
        )

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "document.png"
            image_path.write_bytes(b"image bytes")
            prediction = infer_question(make_question(image_path), client)

        self.assertEqual(prediction["answer"], "42")
        self.assertEqual(prediction["attempt_count"], 2)
        self.assertEqual(prediction["latency_ms"], 25)
        self.assertEqual(prediction["input_tokens"], 45)
        self.assertEqual(prediction["output_tokens"], 10)
        self.assertEqual(prediction["reasoning_tokens"], 6)
        self.assertEqual(prediction["response_status"], "completed")
        self.assertIsNone(prediction["error"])
        self.assertIn("required format", client.prompts[1])

    def test_missing_image_becomes_an_error_record_without_an_api_call(self):
        client = FakeInferenceClient([])
        prediction = infer_question(make_question(Path("missing.png")), client)

        self.assertEqual(prediction["answer"], "unknown")
        self.assertEqual(prediction["attempt_count"], 0)
        self.assertIn("Image not found", prediction["error"])
        self.assertEqual(client.prompts, [])

    def test_incomplete_response_reports_the_api_reason_after_one_retry(self):
        incomplete = make_response(
            "",
            reasoning_tokens=5,
            status="incomplete",
            incomplete_reason="max_output_tokens",
        )
        client = FakeInferenceClient([incomplete, incomplete])

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "document.png"
            image_path.write_bytes(b"image bytes")
            prediction = infer_question(make_question(image_path), client)

        self.assertEqual(prediction["attempt_count"], 2)
        self.assertEqual(prediction["reasoning_tokens"], 10)
        self.assertEqual(prediction["response_status"], "incomplete")
        self.assertEqual(prediction["incomplete_reason"], "max_output_tokens")
        self.assertIn("incomplete after 2 attempts: max_output_tokens", prediction["error"])


class PipelineTests(unittest.TestCase):
    def test_run_inference_keeps_successes_and_retries_previous_errors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "document.png"
            image_path.write_bytes(b"image bytes")
            questions_path = root / "questions.jsonl"
            output_path = root / "predictions.jsonl"
            questions = [
                make_question(image_path, "10"),
                make_question(image_path, "11"),
            ]
            write_jsonl(questions_path, questions)
            write_jsonl(
                output_path,
                [
                    {
                        "page_id": "documentvqa_1",
                        "question_id": "10",
                        "answer": "old answer",
                        "confidence": 1.0,
                        "model": "test-model",
                        "reasoning_effort": "minimal",
                        "max_output_tokens": 1000,
                        "latency_ms": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "reasoning_tokens": 0,
                        "attempt_count": 1,
                        "response_status": "completed",
                        "incomplete_reason": None,
                        "error": None,
                    },
                    {
                        "page_id": "documentvqa_1",
                        "question_id": "11",
                        "answer": "unknown",
                        "confidence": 0.0,
                        "model": "test-model",
                        "reasoning_effort": "minimal",
                        "max_output_tokens": 1000,
                        "latency_ms": 1,
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "reasoning_tokens": 1,
                        "attempt_count": 2,
                        "response_status": "incomplete",
                        "incomplete_reason": "max_output_tokens",
                        "error": "previous failure",
                    },
                ],
            )
            client = FakeInferenceClient(
                [make_response(json.dumps({"answer": "new", "confidence": 0.7}))]
            )

            run_inference(
                questions_path,
                output_path,
                resume=True,
                client=client,
            )

            predictions = read_jsonl(output_path)

        self.assertEqual([item["question_id"] for item in predictions], ["10", "11"])
        self.assertEqual(predictions[0]["answer"], "old answer")
        self.assertEqual(predictions[1]["answer"], "new")
        self.assertEqual(len(client.prompts), 1)


if __name__ == "__main__":
    unittest.main()
