"""Small OpenAI Responses API client for document-image QA."""

import base64
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import SYSTEM_PROMPT


class VLMResponse(TypedDict):
    """Raw response text and request measurements."""

    text: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    status: str
    incomplete_reason: Optional[str]


ANSWER_FORMAT = {
    "type": "json_schema",
    "name": "document_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    },
}

IMAGE_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class VLMClient:
    """Send one document image and question to an OpenAI vision model."""

    def __init__(
        self,
        model: str = "gpt-5",
        max_output_tokens: int = 1000,
        reasoning_effort: str = "minimal",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        if client is None:
            load_dotenv()
            client = OpenAI(api_key=api_key, max_retries=0)
        self.client = client

    @staticmethod
    def image_data_url(image_path: Path) -> str:
        """Encode a supported local image as a data URL."""
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        media_type = IMAGE_TYPES.get(image_path.suffix.lower())
        if media_type is None:
            raise ValueError(f"Unsupported image type: {image_path.suffix}")

        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def generate(self, image_path: Path, prompt: str) -> VLMResponse:
        """Generate one structured answer and collect latency and token usage."""
        image_url = self.image_data_url(image_path)
        start = time.perf_counter()
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_url, "detail": "high"},
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            text={"format": ANSWER_FORMAT},
            max_output_tokens=self.max_output_tokens,
            reasoning={"effort": self.reasoning_effort},
            store=False,
        )
        latency_ms = round((time.perf_counter() - start) * 1000)
        usage = response.usage
        output_details = usage.output_tokens_details if usage else None
        incomplete_details = response.incomplete_details

        return {
            "text": response.output_text or "",
            "latency_ms": latency_ms,
            "input_tokens": usage.input_tokens if usage else 0,
            "output_tokens": usage.output_tokens if usage else 0,
            "reasoning_tokens": (
                output_details.reasoning_tokens if output_details else 0
            ),
            "status": response.status or "unknown",
            "incomplete_reason": (
                incomplete_details.reason if incomplete_details else None
            ),
        }
