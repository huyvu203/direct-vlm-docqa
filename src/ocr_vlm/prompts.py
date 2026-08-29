"""Prompts used for direct document-image question answering."""

SYSTEM_PROMPT = (
    "Answer questions using only information visible in the document image. "
    "Return the shortest exact answer supported by the image. "
    "If the answer cannot be determined, use 'unknown' with confidence 0."
)


def build_user_prompt(question: str, corrective: bool = False) -> str:
    """Build the question prompt, optionally emphasizing the required output."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")

    prompt = f"Question: {question}\n\nDo not explain your answer."
    if corrective:
        prompt += " Return both the answer and confidence in the required format."
    return prompt
