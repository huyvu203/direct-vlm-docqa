"""
Evaluation metrics for DocQA.
Implements metrics from huy_article.md:
- Exact Match (EM)
- Token-level F1
- Numeric Tolerance (±0.5%)
"""

import re
from typing import List, Tuple
from collections import Counter


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    
    - Convert to lowercase
    - Remove punctuation except % and decimal points
    - Strip whitespace
    """
    text = text.lower().strip()
    # Keep % and decimal points for numeric answers
    text = re.sub(r'[^\w\s%.,-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Calculate Exact Match score.
    
    Returns 1.0 if normalized strings match exactly, 0.0 otherwise.
    """
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    return 1.0 if pred_norm == gt_norm else 0.0


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return normalize_text(text).split()


def calculate_token_f1(prediction: str, ground_truth: str) -> float:
    """
    Calculate token-level F1 score.
    
    Computes F1 based on word overlap between prediction and ground truth.
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    # Count token frequencies
    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)
    
    # Calculate overlap
    common_tokens = pred_counter & gt_counter
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def extract_number(text: str) -> Tuple[float, str]:
    """
    Extract numeric value and unit from text.
    
    Returns:
        (numeric_value, unit_string)
        Returns (None, "") if no number found
    """
    text = normalize_text(text)
    
    # Pattern to match numbers with optional units
    # Handles: 12.4%, $2.4 million, 1,234.56, etc.
    pattern = r'([-+]?[\d,]+\.?\d*)\s*(%|million|billion|thousand|k|m|b)?'
    match = re.search(pattern, text)
    
    if not match:
        return None, ""
    
    # Extract number and remove commas
    number_str = match.group(1).replace(',', '')
    try:
        number = float(number_str)
    except ValueError:
        return None, ""
    
    unit = match.group(2) or ""
    return number, unit


def calculate_numeric_tolerance(
    prediction: str,
    ground_truth: str,
    tolerance_percent: float = 0.5
) -> float:
    """
    Calculate match with numeric tolerance.
    
    For numeric answers, allows ±0.5% deviation from ground truth.
    For non-numeric answers, falls back to exact match.
    
    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        tolerance_percent: Tolerance percentage (default: 0.5%)
        
    Returns:
        1.0 if match within tolerance, 0.0 otherwise
    """
    pred_num, pred_unit = extract_number(prediction)
    gt_num, gt_unit = extract_number(ground_truth)
    
    # If either is not numeric, use exact match
    if pred_num is None or gt_num is None:
        return calculate_exact_match(prediction, ground_truth)
    
    # Check units match (case-insensitive)
    if pred_unit.lower() != gt_unit.lower():
        return 0.0
    
    # Check if within tolerance
    if gt_num == 0:
        # Avoid division by zero
        return 1.0 if pred_num == 0 else 0.0
    
    percent_diff = abs(pred_num - gt_num) / abs(gt_num) * 100
    return 1.0 if percent_diff <= tolerance_percent else 0.0


def classify_question_type(question: str) -> str:
    """
    Classify question into types from article:
    - Factual lookup
    - Numeric extraction
    - Spatial reasoning
    - Multi-hop logic
    """
    question_lower = question.lower()
    
    # Spatial reasoning keywords
    spatial_keywords = [
        'which is larger', 'which is smaller', 'compare', 'tallest', 'shortest',
        'highest', 'lowest', 'trend', 'increase', 'decrease', 'chart', 'graph',
        'bar', 'line', 'pie', 'above', 'below', 'left', 'right', 'top', 'bottom'
    ]
    if any(kw in question_lower for kw in spatial_keywords):
        return 'spatial_reasoning'
    
    # Multi-hop logic keywords
    multihop_keywords = [
        'difference between', 'total of', 'average of', 'sum of',
        'how many more', 'how many less', 'combined', 'altogether'
    ]
    if any(kw in question_lower for kw in multihop_keywords):
        return 'multi_hop_logic'
    
    # Numeric extraction keywords
    numeric_keywords = [
        'how much', 'how many', 'what is the number', 'what is the value',
        'percentage', 'percent', 'revenue', 'cost', 'price', 'count'
    ]
    if any(kw in question_lower for kw in numeric_keywords):
        return 'numeric_extraction'
    
    # Default to factual lookup
    return 'factual_lookup'
