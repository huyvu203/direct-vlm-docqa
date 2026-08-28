#!/usr/bin/env python3
"""
VLM-based Question Answering Script
Implements direct vision-language inference from huy_article.md.

Usage:
    python scripts/05_answer_questions.py \
        --mode vlm_image \
        --model gpt-5 \
        --questions data/processed/questions/dev.jsonl \
        --images-root data/raw \
        --output outputs/answers/vlm_dev.jsonl
"""

import json
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from .prompts import PromptTemplate
from .retry import JSONValidationError, RetryHandler
from .vlm_client import VLMClient

app = typer.Typer()


@app.command()
def answer_questions(
    mode: str = typer.Option(
        "vlm_image",
        help="Mode: 'vlm_image' for direct VLM, 'textract_llm' for OCR+LLM"
    ),
    questions: Path = typer.Option(
        ...,
        help="Path to questions JSONL file"
    ),
    images_root: Path = typer.Option(
        Path("data/raw"),
        help="Root directory for images"
    ),
    output: Path = typer.Option(
        ...,
        help="Output path for answers JSONL"
    ),
    model: str = typer.Option(
        "gpt-5",
        help="Model name (e.g., gpt-5, gpt-4o)"
    ),
    prompt_template: Optional[Path] = typer.Option(
        None,
        help="Optional custom prompt template file"
    ),
    max_questions: Optional[int] = typer.Option(
        None,
        help="Maximum number of questions to process (for testing)"
    )
):
    """
    Answer DocQA questions using VLM.
    
    Implements the pipeline from huy_article.md:
    - Single-step: Image + Question → GPT-5 → Answer
    - Temperature: 0.0 (deterministic)
    - Max tokens: 150
    - High detail vision mode
    - Single retry on JSON validation failure
    """
    if mode != "vlm_image":
        typer.echo(f"Error: Mode '{mode}' not implemented. Use 'vlm_image'.", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Loading questions from {questions}...")
    question_data = []
    with open(questions, 'r') as f:
        for line in f:
            question_data.append(json.loads(line))
    
    if max_questions:
        question_data = question_data[:max_questions]
        typer.echo(f"Limited to {max_questions} questions for testing")
    
    typer.echo(f"Loaded {len(question_data)} questions")
    
    # Initialize VLM client
    typer.echo(f"Initializing VLM client with model: {model}")
    client = VLMClient(model=model, temperature=0.0, max_tokens=150)
    
    # Load prompts
    if prompt_template:
        typer.echo(f"Loading custom prompt template from {prompt_template}")
        prompts = PromptTemplate.load_from_file(str(prompt_template))
        system_message = prompts['system']
        user_template = prompts['user']
    else:
        typer.echo("Using default prompt templates from article")
        system_message = PromptTemplate.get_system_message()
        user_template = PromptTemplate.USER_TEMPLATE
    
    # Process questions
    typer.echo(f"\nProcessing {len(question_data)} questions with {model}...")
    
    output.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    retry_count = 0
    validation_failures = 0
    
    with open(output, 'w') as out_f:
        for q_data in tqdm(question_data, desc="Answering questions"):
            question_id = q_data['question_id']
            page_id = q_data['page_id']
            question_text = q_data['question']
            ground_truth = q_data.get('answer', '')
            
            # Find image path
            # Try multiple possible locations
            possible_paths = [
                images_root / page_id / f"{page_id}.png",
                images_root / "infographicvqa" / f"{page_id}.png",
                images_root / f"{page_id}.png",
            ]
            
            image_path = None
            for path in possible_paths:
                if path.exists():
                    image_path = path
                    break
            
            if not image_path:
                # Log error but continue
                result = {
                    "question_id": question_id,
                    "page_id": page_id,
                    "question": question_text,
                    "answer": "ERROR: Image not found",
                    "confidence": 0.0,
                    "latency_ms": 0,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "cost_usd": 0.0,
                    "retry_needed": False,
                    "error": f"Image not found at {possible_paths[0]}"
                }
                out_f.write(json.dumps(result) + '\n')
                validation_failures += 1
                continue
            
            # Call VLM
            retry_needed = False
            try:
                response = client.answer_question(
                    image_path=str(image_path),
                    question=question_text,
                    system_message=system_message,
                    user_template=user_template
                )
                
                # Parse and validate response
                try:
                    parsed = RetryHandler.parse_and_validate(response['response'])
                    answer = parsed['answer']
                    confidence = float(parsed['confidence'])
                
                except JSONValidationError as e:
                    # Retry with corrective prompt
                    retry_needed = True
                    retry_count += 1
                    
                    # Create corrective prompt
                    corrective_user_prompt = RetryHandler.get_corrective_prompt(
                        user_template.format(question=question_text)
                    )
                    
                    # Retry
                    response = client.call_vision_api(
                        image_path=str(image_path),
                        prompt=corrective_user_prompt,
                        system_message=system_message
                    )
                    
                    try:
                        parsed = RetryHandler.parse_and_validate(response['response'])
                        answer = parsed['answer']
                        confidence = float(parsed['confidence'])
                    except JSONValidationError:
                        # Still failed after retry
                        validation_failures += 1
                        answer = "unknown"
                        confidence = 0.0
                
                # Build result
                result = {
                    "question_id": question_id,
                    "page_id": page_id,
                    "question": question_text,
                    "ground_truth": ground_truth,
                    "answer": answer,
                    "confidence": confidence,
                    "latency_ms": response['latency_ms'],
                    "tokens_input": response['tokens_input'],
                    "tokens_output": response['tokens_output'],
                    "cost_usd": response['cost_usd'],
                    "retry_needed": retry_needed
                }
                
            except Exception as e:
                # API error or other failure
                result = {
                    "question_id": question_id,
                    "page_id": page_id,
                    "question": question_text,
                    "ground_truth": ground_truth,
                    "answer": "ERROR",
                    "confidence": 0.0,
                    "latency_ms": 0,
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "cost_usd": 0.0,
                    "retry_needed": False,
                    "error": str(e)
                }
                validation_failures += 1
            
            # Write result
            out_f.write(json.dumps(result) + '\n')
            results.append(result)
    
    # Summary statistics
    total_questions = len(question_data)
    retry_rate = (retry_count / total_questions * 100) if total_questions > 0 else 0
    failure_rate = (validation_failures / total_questions * 100) if total_questions > 0 else 0
    
    typer.echo(f"\n✓ Processing complete!")
    typer.echo(f"  Total questions: {total_questions}")
    typer.echo(f"  Retries needed: {retry_count} ({retry_rate:.1f}%)")
    typer.echo(f"  Validation failures: {validation_failures} ({failure_rate:.1f}%)")
    typer.echo(f"  Results written to: {output}")
    
    # Calculate aggregate stats if we have valid results
    valid_results = [r for r in results if 'error' not in r and r['answer'] != 'ERROR']
    if valid_results:
        avg_latency = sum(r['latency_ms'] for r in valid_results) / len(valid_results)
        total_cost = sum(r['cost_usd'] for r in valid_results)
        avg_tokens_in = sum(r['tokens_input'] for r in valid_results) / len(valid_results)
        avg_tokens_out = sum(r['tokens_output'] for r in valid_results) / len(valid_results)
        
        typer.echo(f"\nOperational Metrics:")
        typer.echo(f"  Avg latency: {avg_latency:.0f}ms")
        typer.echo(f"  Avg tokens (input): {avg_tokens_in:.0f}")
        typer.echo(f"  Avg tokens (output): {avg_tokens_out:.0f}")
        typer.echo(f"  Total cost: ${total_cost:.4f}")
        typer.echo(f"  Cost per 100 questions: ${total_cost / len(valid_results) * 100:.2f}")


if __name__ == "__main__":
    app()
