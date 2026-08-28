"""
Retry handler with JSON validation.
Implements the retry mechanism from huy_article.md:
- Validate JSON schema on responses
- Single retry with corrective prompt on failure
- Target: <1.3% validation failure rate
"""

import json
import re
from typing import Dict, Any, Optional, Tuple


class JSONValidationError(Exception):
    """Raised when response fails JSON schema validation."""
    pass


class RetryHandler:
    """Handle response validation and retries for VLM outputs."""
    
    # Corrective prompt from article
    CORRECTIVE_PROMPT = (
        'Your previous response was not valid JSON. Respond with exactly this format: '
        '{"answer": "text", "confidence": 0.8} and nothing else.'
    )
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text that may contain surrounding content.
        
        Tries multiple strategies:
        1. Direct JSON parse
        2. Extract JSON block from markdown code fence
        3. Find JSON object with regex
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from markdown code fence
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON object anywhere in text
        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    @staticmethod
    def validate_response_schema(response_json: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate response matches expected schema.
        
        Expected schema from article:
        {
            "answer": "exact text or number",
            "confidence": 0.0-1.0
        }
        
        Returns:
            (is_valid, error_message)
        """
        # Check required fields
        if "answer" not in response_json:
            return False, "Missing 'answer' field"
        
        if "confidence" not in response_json:
            return False, "Missing 'confidence' field"
        
        # Validate types
        if not isinstance(response_json["answer"], str):
            return False, "'answer' must be a string"
        
        # Confidence should be numeric
        try:
            confidence = float(response_json["confidence"])
            if not (0.0 <= confidence <= 1.0):
                return False, "'confidence' must be between 0.0 and 1.0"
        except (TypeError, ValueError):
            return False, "'confidence' must be a number between 0.0 and 1.0"
        
        return True, None
    
    @classmethod
    def parse_and_validate(cls, response_text: str) -> Dict[str, Any]:
        """
        Parse response text and validate schema.
        
        Returns:
            Validated JSON dict
            
        Raises:
            JSONValidationError: If validation fails
        """
        # Extract JSON
        response_json = cls.extract_json_from_text(response_text)
        if response_json is None:
            raise JSONValidationError(f"Could not extract valid JSON from response: {response_text[:200]}")
        
        # Validate schema
        is_valid, error_msg = cls.validate_response_schema(response_json)
        if not is_valid:
            raise JSONValidationError(f"Schema validation failed: {error_msg}")
        
        return response_json
    
    @classmethod
    def get_corrective_prompt(cls, original_question: str) -> str:
        """
        Get corrective prompt for retry.
        
        Args:
            original_question: The original question text
            
        Returns:
            Corrective prompt combining original question and format reminder
        """
        return f"{original_question}\n\n{cls.CORRECTIVE_PROMPT}"
