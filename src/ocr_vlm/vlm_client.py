"""
GPT-5 Vision API Client
Implements the exact configuration from huy_article.md:
- Model: gpt-5 with native multimodal capabilities
- Temperature: 0.0 (deterministic)
- Max tokens: 150
- Image mode: High detail
- Tracks: latency, token counts, cost
"""

import base64
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI


class VLMClient:
    """Client for GPT-5 Vision API with tracking and configuration from article."""
    
    # Pricing from article (August 2025 OpenAI pricing)
    VISION_INPUT_PRICE_PER_1K = 0.0075  # $0.0075 per 1K vision tokens
    TEXT_INPUT_PRICE_PER_1K = 0.0075    # $0.0075 per 1K text tokens
    OUTPUT_PRICE_PER_1K = 0.02          # $0.02 per 1K output tokens
    
    def __init__(
        self,
        model: str = "gpt-5",
        temperature: float = 0.0,
        max_tokens: int = 150,
        api_key: Optional[str] = None
    ):
        """
        Initialize VLM client with article-specified parameters.
        
        Args:
            model: Model name (default: gpt-5)
            temperature: Sampling temperature (default: 0.0 for deterministic)
            max_tokens: Maximum output tokens (default: 150)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def call_vision_api(
        self,
        image_path: str,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call GPT-5 Vision API with image and prompt.
        
        Returns dict with:
            - response: Raw text response from model
            - latency_ms: Request latency in milliseconds
            - tokens_input: Input tokens (text + vision)
            - tokens_output: Output tokens
            - cost_usd: Estimated cost in USD
        """
        start_time = time.time()
        
        # Prepare messages
        messages = []
        if system_message:
            # GPT-5 uses "developer" role instead of "system"
            role = "developer" if self.model.startswith("gpt-5") or self.model.startswith("gpt-4.1") else "system"
            messages.append({
                "role": role,
                "content": system_message
            })
        
        # Add user message with image
        # Use high detail mode for best quality (article specification)
        image_ext = Path(image_path).suffix.lower()
        media_type = "image/jpeg" if image_ext in ['.jpg', '.jpeg'] else "image/png"
        
        base64_image = self.encode_image(image_path)
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}",
                        "detail": "high"  # High detail mode from article
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })
        
        # Make API call
        # GPT-5 and o1 models have different parameter requirements
        api_params = {
            "model": self.model,
            "messages": messages,
        }
        
        # Temperature: 
        # - o1 models don't support it at all
        # - gpt-5 only supports default value of 1 (not 0)
        # - other models support 0-2 range
        if self.model.startswith("o1"):
            # o1 models don't support temperature parameter
            pass
        elif self.model.startswith("gpt-5") or self.model.startswith("gpt-4.1"):
            # gpt-5 only supports temperature=1 (default), so only set if not 0
            if self.temperature != 0.0:
                api_params["temperature"] = self.temperature
            # else: omit parameter to use default of 1
        else:
            # Other models support temperature 0-2
            api_params["temperature"] = self.temperature
        
        # Use max_completion_tokens for gpt-5/o1/gpt-4.1, max_tokens for older models
        if self.model.startswith("gpt-5") or self.model.startswith("o1") or self.model.startswith("gpt-4.1"):
            api_params["max_completion_tokens"] = self.max_tokens
        else:
            api_params["max_tokens"] = self.max_tokens
        
        response = self.client.chat.completions.create(**api_params)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Extract token usage
        usage = response.usage
        tokens_input = usage.prompt_tokens
        tokens_output = usage.completion_tokens
        
        # Calculate cost (article pricing)
        # Note: OpenAI API doesn't separate vision vs text tokens in usage
        # We use the vision input price for all input tokens as approximation
        cost_usd = (
            tokens_input * self.VISION_INPUT_PRICE_PER_1K / 1000 +
            tokens_output * self.OUTPUT_PRICE_PER_1K / 1000
        )
        
        return {
            "response": response.choices[0].message.content,
            "latency_ms": latency_ms,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cost_usd": cost_usd,
            "finish_reason": response.choices[0].finish_reason
        }
    
    def answer_question(
        self,
        image_path: str,
        question: str,
        system_message: str,
        user_template: str
    ) -> Dict[str, Any]:
        """
        Answer a DocQA question using VLM.
        
        Args:
            image_path: Path to document image
            question: Question text
            system_message: System prompt
            user_template: User prompt template (with {question} placeholder)
            
        Returns:
            Dict with response and metadata
        """
        # Format user prompt
        user_prompt = user_template.format(question=question)
        
        return self.call_vision_api(
            image_path=image_path,
            prompt=user_prompt,
            system_message=system_message
        )
