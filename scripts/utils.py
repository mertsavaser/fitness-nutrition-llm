"""
Utility functions for dataset generation and cleaning.
"""

import re
import json
from typing import Dict, Any
from langdetect import detect, LangDetectException

# Unsafe keywords related to medical conditions, diseases, or extreme diets
UNSAFE_KEYWORDS = [
    'diyabet', 'şeker hastalığı', 'hipertansiyon', 'tansiyon',
    'kalp hastalığı', 'böbrek hastalığı', 'karaciğer hastalığı',
    'tiroid', 'hipotiroid', 'hipertiroid', 'kanser', 'tümör',
    'anoreksiya', 'bulimia', 'obezite hastalığı', 'metabolik sendrom',
    'kolesterol ilacı', 'tansiyon ilacı', 'insülin', 'glukoz',
    'hastalık', 'tedavi', 'ilaç', 'reçete', 'doktor tavsiyesi',
    'tıbbi müdahale', 'ameliyat', 'cerrahi', 'klinik', 'hastane'
]


def is_valid_json(json_str: str) -> bool:
    """Check if a string is valid JSON."""
    try:
        json.loads(json_str)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def contains_numbers(text: str) -> bool:
    """Check if text contains numeric values."""
    return bool(re.search(r'\d+', text))


def is_turkish(text: str) -> bool:
    """Detect if text is in Turkish language."""
    try:
        detected_lang = detect(text)
        return detected_lang == 'tr'
    except LangDetectException:
        return False


def contains_unsafe_keywords(text: str) -> bool:
    """Check if text contains unsafe medical or disease-related keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in UNSAFE_KEYWORDS)


def has_minimum_length(text: str, min_length: int = 100) -> bool:
    """Check if text meets minimum length requirement."""
    return len(text) >= min_length


def validate_sample(sample: Dict[str, Any], min_output_length: int = 100) -> bool:
    """
    Validate a single sample according to all cleaning criteria.
    
    Args:
        sample: Dictionary with 'instruction', 'input', 'output' keys
        min_output_length: Minimum length for output field
        
    Returns:
        True if sample passes all validation checks, False otherwise
    """
    # Check if sample has required fields
    if not all(key in sample for key in ['instruction', 'input', 'output']):
        return False
    
    output = sample['output']
    
    # Check if output contains numbers
    if not contains_numbers(output):
        return False
    
    # Check minimum length
    if not has_minimum_length(output, min_output_length):
        return False
    
    # Check for unsafe keywords
    if contains_unsafe_keywords(output):
        return False
    
    # Check if output is Turkish
    if not is_turkish(output):
        return False
    
    return True

