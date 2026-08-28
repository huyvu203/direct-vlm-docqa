#!/usr/bin/env python3
"""
Evaluation Script for DocQA
Computes metrics from huy_article.md:
- Exact Match (EM)
- Token F1
- Numeric Tolerance (±0.5%)
- Operational metrics (latency, cost)
- Per-question-type breakdown

Usage:
    python scripts/06_evaluate_qa.py \
        --pred vlm=outputs/answers/vlm_test.jsonl \
        --questions data/processed/questions/test.jsonl \
        --output results/vlm_metrics.json
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import typer

from .metrics import (
    calculate_exact_match,
    calculate_token_f1,
    calculate_numeric_tolerance,
    classify_question_type
)

app = typer.Typer()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]


@app.command()
def evaluate(
    pred: List[str] = typer.Option(
        ...,
        help="Prediction files in format 'name=path.jsonl' (can specify multiple)"
    ),
    questions: Path = typer.Option(
        ...,
        help="Ground truth questions JSONL file"
    ),
    output: Path = typer.Option(
        ...,
        help="Output path for metrics JSON"
    ),
    verbose: bool = typer.Option(
        False,
        help="Print detailed per-question results"
    )
):
    """
    Evaluate DocQA predictions against ground truth.
    
    Computes all metrics from huy_article.md:
    - Accuracy: EM, Token F1, Numeric Tolerance
    - Operational: Latency (median, p95), Cost, Tokens
    - Per-question-type breakdown
    """
    # Parse prediction file arguments
    pred_files = {}
    for p in pred:
        if '=' not in p:
            typer.echo(f"Error: Prediction must be in format 'name=path', got: {p}", err=True)
            raise typer.Exit(1)
        name, path = p.split('=', 1)
        pred_files[name] = Path(path)
    
    # Load ground truth
    typer.echo(f"Loading ground truth from {questions}...")
    gt_data = load_jsonl(questions)
    gt_dict = {item['question_id']: item for item in gt_data}
    typer.echo(f"Loaded {len(gt_dict)} ground truth questions")
    
    # Evaluate each prediction file
    all_metrics = {}
    
    for name, pred_path in pred_files.items():
        typer.echo(f"\nEvaluating predictions: {name} from {pred_path}")
        
        pred_data = load_jsonl(pred_path)
        typer.echo(f"Loaded {len(pred_data)} predictions")
        
        # Calculate metrics
        metrics = evaluate_predictions(pred_data, gt_dict, verbose)
        all_metrics[name] = metrics
        
        # Print summary
        print_metrics_summary(name, metrics)
    
    # Save results
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    typer.echo(f"\n✓ Metrics saved to {output}")


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth: Dict[str, Dict[str, Any]],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth.
    
    Returns dict with all metrics from article.
    """
    # Overall metrics
    em_scores = []
    f1_scores = []
    numeric_scores = []
    
    # Per-question-type metrics
    type_metrics = defaultdict(lambda: {
        'em_scores': [],
        'f1_scores': [],
        'count': 0
    })
    
    # Operational metrics
    latencies = []
    tokens_input = []
    tokens_output = []
    costs = []
    retry_counts = 0
    error_counts = 0
    
    # Error analysis
    errors = []
    
    for pred in predictions:
        question_id = pred['question_id']
        
        # Skip if ground truth not found
        if question_id not in ground_truth:
            if verbose:
                typer.echo(f"Warning: No ground truth for {question_id}", err=True)
            continue
        
        gt = ground_truth[question_id]
        pred_answer = pred.get('answer', 'unknown')
        gt_answer = gt.get('answer', '')
        question_text = gt.get('question', '')
        
        # Skip errors
        if 'error' in pred or pred_answer in ['ERROR', 'ERROR: Image not found']:
            error_counts += 1
            continue
        
        # Calculate accuracy metrics
        em = calculate_exact_match(pred_answer, gt_answer)
        f1 = calculate_token_f1(pred_answer, gt_answer)
        numeric = calculate_numeric_tolerance(pred_answer, gt_answer)
        
        em_scores.append(em)
        f1_scores.append(f1)
        numeric_scores.append(numeric)
        
        # Classify question type and track
        q_type = classify_question_type(question_text)
        type_metrics[q_type]['em_scores'].append(em)
        type_metrics[q_type]['f1_scores'].append(f1)
        type_metrics[q_type]['count'] += 1
        
        # Operational metrics
        if 'latency_ms' in pred and pred['latency_ms'] > 0:
            latencies.append(pred['latency_ms'])
        if 'tokens_input' in pred:
            tokens_input.append(pred['tokens_input'])
        if 'tokens_output' in pred:
            tokens_output.append(pred['tokens_output'])
        if 'cost_usd' in pred:
            costs.append(pred['cost_usd'])
        if pred.get('retry_needed', False):
            retry_counts += 1
        
        # Track errors for analysis
        if em == 0.0:
            errors.append({
                'question_id': question_id,
                'question': question_text,
                'predicted': pred_answer,
                'ground_truth': gt_answer,
                'f1': f1,
                'type': q_type
            })
        
        if verbose and em == 0.0:
            typer.echo(f"  ✗ {question_id}: '{pred_answer}' vs '{gt_answer}' (F1: {f1:.2f})")
    
    # Compile metrics
    total_questions = len(em_scores)
    
    metrics = {
        'accuracy': {
            'exact_match': sum(em_scores) / total_questions if total_questions > 0 else 0.0,
            'token_f1': sum(f1_scores) / total_questions if total_questions > 0 else 0.0,
            'numeric_tolerance': sum(numeric_scores) / total_questions if total_questions > 0 else 0.0,
            'total_questions': total_questions
        },
        'by_question_type': {},
        'operational': {},
        'errors': {
            'error_count': error_counts,
            'retry_count': retry_counts,
            'retry_rate_percent': retry_counts / total_questions * 100 if total_questions > 0 else 0.0
        }
    }
    
    # Per-question-type breakdown
    for q_type, type_data in type_metrics.items():
        count = type_data['count']
        if count > 0:
            metrics['by_question_type'][q_type] = {
                'exact_match': sum(type_data['em_scores']) / count,
                'token_f1': sum(type_data['f1_scores']) / count,
                'count': count
            }
    
    # Operational metrics
    if latencies:
        metrics['operational']['latency'] = {
            'median_ms': statistics.median(latencies),
            'mean_ms': statistics.mean(latencies),
            'p95_ms': calculate_percentile(latencies, 95),
            'min_ms': min(latencies),
            'max_ms': max(latencies)
        }
    
    if tokens_input:
        metrics['operational']['tokens'] = {
            'avg_input': statistics.mean(tokens_input),
            'avg_output': statistics.mean(tokens_output) if tokens_output else 0,
            'total_input': sum(tokens_input),
            'total_output': sum(tokens_output) if tokens_output else 0
        }
    
    if costs:
        total_cost = sum(costs)
        metrics['operational']['cost'] = {
            'total_usd': total_cost,
            'avg_per_question': total_cost / len(costs),
            'per_100_questions': total_cost / len(costs) * 100
        }
    
    # Add sample errors for analysis
    metrics['sample_errors'] = errors[:10]  # First 10 errors
    
    return metrics


def print_metrics_summary(name: str, metrics: Dict[str, Any]):
    """Print formatted metrics summary."""
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Metrics for: {name}")
    typer.echo(f"{'='*60}")
    
    # Accuracy
    acc = metrics['accuracy']
    typer.echo(f"\nAccuracy Metrics:")
    typer.echo(f"  Exact Match (EM):      {acc['exact_match']*100:6.1f}%")
    typer.echo(f"  Token F1:              {acc['token_f1']*100:6.1f}%")
    typer.echo(f"  Numeric Tolerance:     {acc['numeric_tolerance']*100:6.1f}%")
    typer.echo(f"  Total Questions:       {acc['total_questions']:6d}")
    
    # By question type
    if 'by_question_type' in metrics and metrics['by_question_type']:
        typer.echo(f"\nBy Question Type:")
        for q_type, type_metrics in sorted(metrics['by_question_type'].items()):
            typer.echo(f"  {q_type:20s}: EM={type_metrics['exact_match']*100:5.1f}%, "
                      f"F1={type_metrics['token_f1']*100:5.1f}%, n={type_metrics['count']}")
    
    # Operational
    if 'operational' in metrics:
        op = metrics['operational']
        
        if 'latency' in op:
            typer.echo(f"\nLatency:")
            typer.echo(f"  Median:   {op['latency']['median_ms']:7.0f}ms")
            typer.echo(f"  Mean:     {op['latency']['mean_ms']:7.0f}ms")
            typer.echo(f"  P95:      {op['latency']['p95_ms']:7.0f}ms")
        
        if 'tokens' in op:
            typer.echo(f"\nTokens:")
            typer.echo(f"  Avg Input:  {op['tokens']['avg_input']:7.0f}")
            typer.echo(f"  Avg Output: {op['tokens']['avg_output']:7.0f}")
        
        if 'cost' in op:
            typer.echo(f"\nCost:")
            typer.echo(f"  Total:            ${op['cost']['total_usd']:7.4f}")
            typer.echo(f"  Per question:     ${op['cost']['avg_per_question']:7.4f}")
            typer.echo(f"  Per 100 questions: ${op['cost']['per_100_questions']:6.2f}")
    
    # Errors
    if 'errors' in metrics:
        err = metrics['errors']
        typer.echo(f"\nErrors & Retries:")
        typer.echo(f"  Errors:       {err['error_count']:6d}")
        typer.echo(f"  Retries:      {err['retry_count']:6d} ({err['retry_rate_percent']:.1f}%)")


if __name__ == "__main__":
    app()
