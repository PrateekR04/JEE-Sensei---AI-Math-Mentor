"""
Confidence calculation utilities for OCR and ASR results.
"""

import re
from typing import List, Tuple


def calculate_average_confidence(confidences: List[float]) -> float:
    """
    Calculate average confidence score from a list of confidence values.
    
    Args:
        confidences: List of confidence scores (0.0 to 1.0)
        
    Returns:
        Average confidence score (0.0 to 1.0)
    """
    if not confidences:
        return 0.0
    
    return sum(confidences) / len(confidences)


def calculate_weighted_confidence(results: List[Tuple[str, float]]) -> float:
    """
    Calculate weighted confidence based on text length.
    Longer text segments contribute more to the overall confidence.
    
    Args:
        results: List of (text, confidence) tuples
        
    Returns:
        Weighted average confidence score (0.0 to 1.0)
    """
    if not results:
        return 0.0
    
    total_weight = 0
    weighted_sum = 0
    
    for text, confidence in results:
        weight = len(text.strip())
        if weight > 0:
            weighted_sum += confidence * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return weighted_sum / total_weight


def calculate_fragmentation_penalty(num_detections: int, total_chars: int) -> float:
    """
    Calculate penalty for highly fragmented OCR output.
    Many small detections indicate poor quality or fragmented text.
    
    Args:
        num_detections: Number of separate text detections
        total_chars: Total number of characters detected
        
    Returns:
        Penalty factor (0.0 to 1.0), where 1.0 means no penalty
    """
    if num_detections == 0 or total_chars == 0:
        return 1.0
    
    # Calculate average characters per detection
    avg_chars_per_detection = total_chars / num_detections
    
    # Ideal: 10+ characters per detection
    # Fragmented: <3 characters per detection
    if avg_chars_per_detection >= 10:
        return 1.0  # No penalty
    elif avg_chars_per_detection >= 5:
        return 0.95  # Small penalty
    elif avg_chars_per_detection >= 3:
        return 0.85  # Medium penalty
    else:
        return 0.7  # High penalty for very fragmented output


def detect_math_patterns(text: str) -> float:
    """
    Detect common math patterns in text to boost confidence.
    Recognizable math structures indicate successful OCR.
    
    Args:
        text: Extracted text to analyze
        
    Returns:
        Confidence boost factor (1.0 to 1.2)
    """
    if not text:
        return 1.0
    
    boost = 1.0
    
    # Check for common math patterns
    patterns = [
        r'\d+\s*[+\-×x÷/]\s*\d+',  # Basic arithmetic (2 + 3, 5 x 4)
        r'[a-z]\s*[+\-×x÷/]\s*\d+',  # Variable + number (x + 5)
        r'\d+\s*[=]\s*\d+',  # Equations (2 = 2)
        r'[a-z]\s*[=]\s*',  # Variable equations (x = )
        r'\d+[a-z]',  # Coefficient notation (2x, 3y)
        r'\([^)]+\)',  # Parentheses
        r'\d+\^\d+',  # Exponents (2^3)
    ]
    
    matches = 0
    for pattern in patterns:
        if re.search(pattern, text.lower()):
            matches += 1
    
    # Boost confidence based on number of math patterns found
    if matches >= 3:
        boost = 1.15
    elif matches >= 2:
        boost = 1.10
    elif matches >= 1:
        boost = 1.05
    
    return boost


def calculate_ocr_confidence(detections: List[Tuple[str, float]], final_text: str) -> float:
    """
    Calculate comprehensive OCR confidence score.
    Combines weighted average, fragmentation penalty, and pattern detection.
    
    Args:
        detections: List of (text, confidence) tuples from OCR
        final_text: Final processed text
        
    Returns:
        Final confidence score (0.0 to 1.0)
    """
    if not detections:
        return 0.0
    
    # 1. Calculate weighted average confidence
    weighted_conf = calculate_weighted_confidence(detections)
    
    # 2. Calculate fragmentation penalty
    num_detections = len(detections)
    total_chars = sum(len(text.strip()) for text, _ in detections)
    fragmentation_penalty = calculate_fragmentation_penalty(num_detections, total_chars)
    
    # 3. Detect math patterns for confidence boost
    pattern_boost = detect_math_patterns(final_text)
    
    # 4. Combine all factors
    final_confidence = weighted_conf * fragmentation_penalty * pattern_boost
    
    # 5. Ensure confidence stays within [0.0, 1.0]
    final_confidence = min(1.0, max(0.0, final_confidence))
    
    return final_confidence


def format_confidence_percentage(confidence: float) -> str:
    """
    Format confidence score as percentage string.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        Formatted percentage string (e.g., "95.5%")
    """
    return f"{confidence * 100:.1f}%"


def get_confidence_color(confidence: float) -> str:
    """
    Get color code based on confidence level for UI display.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        Color string for Streamlit markdown
    """
    if confidence >= 0.9:
        return "green"
    elif confidence >= 0.7:
        return "orange"
    else:
        return "red"
