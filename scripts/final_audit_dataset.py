"""
Final dataset audit script for training readiness.
Read-only analysis of train_TRAINING_READY.jsonl.
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / 'data' / 'cleaned' / 'train_TRAINING_READY.jsonl'

# Turkish characters for language detection
TURKISH_CHARS = set('çğışüÇĞIŞÜ')
TURKISH_KEYWORDS = ['kalori', 'gram', 'protein', 'karbonhidrat', 'yağ', 'egzersiz', 'antrenman', 'beslenme', 'diyet', 'kilo', 'tıbbi', 'hocam']
ENGLISH_KEYWORDS = ['calorie', 'gram', 'protein', 'carbohydrate', 'fat', 'exercise', 'workout', 'nutrition', 'diet', 'weight', 'medical', 'coach']

# Conditional logic patterns
CONDITIONAL_PATTERNS_TR = [
    r'eğer',
    r'ancak',
    r'ama eğer',
    r'ise',
    r'olursa',
    r'yaparsan',
]
CONDITIONAL_PATTERNS_EN = [
    r'\bif\b',
    r'\bwhen\b',
    r'\bunless\b',
]

# Disclaimer patterns
DISCLAIMER_PATTERNS = [
    r'tıbbi\s+(tavsiye|öneri|müdahale)',
    r'not\s+medical\s+advice',
    r'does\s+not\s+(constitute|replace)\s+medical',
    r'genel\s+(rehberlik|tavsiye)',
    r'general\s+(guidance|advice)',
]


def load_samples(file_path: Path) -> List[Dict[str, Any]]:
    """Load all samples from JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def detect_language(text: str) -> str:
    """Detect language of text. Returns 'tr', 'en', 'mixed', or 'unknown'."""
    if not text:
        return 'unknown'
    
    text_lower = text.lower()
    has_turkish_chars = any(char in TURKISH_CHARS for char in text)
    turkish_keyword_count = sum(1 for kw in TURKISH_KEYWORDS if kw in text_lower)
    english_keyword_count = sum(1 for kw in ENGLISH_KEYWORDS if kw in text_lower)
    
    if has_turkish_chars:
        if english_keyword_count > turkish_keyword_count * 0.5:
            return 'mixed'
        return 'tr'
    elif turkish_keyword_count > english_keyword_count:
        return 'tr'
    elif english_keyword_count > 0:
        return 'en'
    else:
        return 'unknown'


def has_numbers(text: str) -> bool:
    """Check if text contains numeric values."""
    return bool(re.search(r'\d+', text))


def has_conditional_logic(text: str) -> bool:
    """Check if text contains conditional logic (eğer/if/ancak)."""
    text_lower = text.lower()
    patterns = CONDITIONAL_PATTERNS_TR + CONDITIONAL_PATTERNS_EN
    return any(re.search(pattern, text_lower) for pattern in patterns)


def has_disclaimer(text: str) -> bool:
    """Check if text contains safety disclaimer."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in DISCLAIMER_PATTERNS)


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity between two texts (word overlap)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def is_structured(instruction: str, input_text: str) -> bool:
    """Detect if sample is structured prompt."""
    return bool(instruction and input_text and len(instruction) > 10 and len(input_text) > 20)


def is_low_effort(instruction: str, input_text: str) -> bool:
    """Detect if sample is low-effort user prompt."""
    combined = (instruction + " " + input_text).lower()
    indicators = [
        len(input_text) < 50,
        not input_text or not instruction,
        any(casual in combined for casual in ['kanka', 'hocam', 'abi', 'hey', 'selam']),
        combined.count('?') > 2,
    ]
    return sum(indicators) >= 2


def is_reassurance(instruction: str, input_text: str) -> bool:
    """Detect if sample is reassurance/myth-busting type."""
    combined = (instruction + " " + input_text).lower()
    reassurance_keywords = [
        'şart mı', 'zararlı mı', 'faydalı mı', 'gerekli mi',
        'kesilmeli mi', 'yapmalı mı', 'yapmamalı mı',
        'doğru mu', 'yanlış mı', 'mit mi',
        'necessary', 'harmful', 'beneficial', 'required',
        'should i', 'do i need', 'is it', 'myth',
    ]
    return any(keyword in combined for keyword in reassurance_keywords)


def audit_dataset():
    """Perform final dataset audit."""
    print("="*70)
    print("FINAL DATASET AUDIT: train_TRAINING_READY.jsonl")
    print("="*70)
    
    if not INPUT_PATH.exists():
        print(f"\nERROR: Dataset file not found at {INPUT_PATH}")
        return
    
    # Load samples
    print("\nLoading dataset...")
    samples = load_samples(INPUT_PATH)
    total_count = len(samples)
    print(f"Loaded {total_count} samples.\n")
    
    if not samples:
        print("ERROR: No samples found.")
        return
    
    # 1. BASIC STATS
    print("1. BASIC STATS")
    print("-" * 70)
    
    instruction_lengths = [len(s.get('instruction', '')) for s in samples]
    input_lengths = [len(s.get('input', '')) for s in samples]
    output_lengths = [len(s.get('output', '')) for s in samples]
    
    def print_stats(name: str, lengths: List[int]):
        if lengths:
            avg = sum(lengths) / len(lengths)
            min_val = min(lengths)
            max_val = max(lengths)
            print(f"{name:20s}: avg={avg:6.1f}, min={min_val:4d}, max={max_val:5d}")
    
    print(f"Total samples: {total_count}")
    print_stats("instruction", instruction_lengths)
    print_stats("input", input_lengths)
    print_stats("output", output_lengths)
    
    # 2. LANGUAGE DISTRIBUTION
    print("\n2. LANGUAGE DISTRIBUTION")
    print("-" * 70)
    
    language_counts = Counter()
    mixed_count = 0
    
    for sample in samples:
        output = sample.get('output', '')
        lang = detect_language(output)
        
        if lang == 'mixed' or (detect_language(sample.get('instruction', '')) != detect_language(output) and 
                               detect_language(sample.get('input', '')) != detect_language(output)):
            mixed_count += 1
            language_counts['mixed'] += 1
        else:
            language_counts[lang] += 1
    
    for lang in ['tr', 'en', 'mixed']:
        count = language_counts.get(lang, 0)
        if count > 0:
            pct = count / total_count * 100
            print(f"{lang.upper():15s}: {count:5d} ({pct:5.1f}%)")
    
    # 3. SAMPLE TYPE DISTRIBUTION
    print("\n3. SAMPLE TYPE DISTRIBUTION")
    print("-" * 70)
    
    structured_count = 0
    low_effort_count = 0
    reassurance_count = 0
    other_count = 0
    
    for sample in samples:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        
        if is_reassurance(instruction, input_text):
            reassurance_count += 1
        elif is_low_effort(instruction, input_text):
            low_effort_count += 1
        elif is_structured(instruction, input_text):
            structured_count += 1
        else:
            other_count += 1
    
    def print_type(name: str, count: int):
        pct = count / total_count * 100
        print(f"{name:25s}: {count:5d} ({pct:5.1f}%)")
    
    print_type("Structured prompts", structured_count)
    print_type("Low-effort user prompts", low_effort_count)
    print_type("Reassurance/myth-busting", reassurance_count)
    print_type("Other", other_count)
    
    # 4. DUPLICATION CHECK
    print("\n4. DUPLICATION CHECK")
    print("-" * 70)
    
    # Exact duplicates
    sample_signatures = []
    for sample in samples:
        sig = (
            sample.get('instruction', '').strip(),
            sample.get('input', '').strip(),
            sample.get('output', '').strip()
        )
        sample_signatures.append(sig)
    
    signature_counts = Counter(sample_signatures)
    exact_duplicates = sum(count - 1 for count in signature_counts.values() if count > 1)
    unique_samples = len(signature_counts)
    duplicate_rate = exact_duplicates / total_count * 100 if total_count > 0 else 0
    
    print(f"Unique samples: {unique_samples}")
    print(f"Exact duplicates: {exact_duplicates}")
    print(f"Exact duplicate rate: {duplicate_rate:.2f}%")
    
    # Near-duplicates (sample-based)
    print("\nChecking near-duplicates (input similarity > 0.9)...")
    near_duplicate_pairs = []
    checked = set()
    
    # Sample first 1000 for performance
    sample_size = min(1000, len(samples))
    for i in range(sample_size):
        if i in checked:
            continue
        input1 = (samples[i].get('instruction', '') + ' ' + samples[i].get('input', '')).strip()
        
        for j in range(i + 1, min(i + 100, len(samples))):
            if j in checked:
                continue
            input2 = (samples[j].get('instruction', '') + ' ' + samples[j].get('input', '')).strip()
            
            similarity = calculate_similarity(input1, input2)
            if similarity > 0.9:
                near_duplicate_pairs.append((i, j, similarity))
                checked.add(j)
    
    near_dup_count = len(near_duplicate_pairs)
    near_dup_rate = (near_dup_count / sample_size * 100) if sample_size > 0 else 0
    
    print(f"Near-duplicate pairs found (sampled): {near_dup_count}")
    print(f"Near-duplicate rate estimate: {near_dup_rate:.2f}%")
    
    # 5. OUTPUT QUALITY METRICS
    print("\n5. OUTPUT QUALITY METRICS")
    print("-" * 70)
    
    has_numbers_count = sum(1 for s in samples if has_numbers(s.get('output', '')))
    has_conditional_count = sum(1 for s in samples if has_conditional_logic(s.get('output', '')))
    has_disclaimer_count = sum(1 for s in samples if has_disclaimer(s.get('output', '')))
    
    numeric_pct = has_numbers_count / total_count * 100
    conditional_pct = has_conditional_count / total_count * 100
    disclaimer_pct = has_disclaimer_count / total_count * 100
    
    print(f"Outputs with numeric values: {has_numbers_count:5d} ({numeric_pct:5.1f}%)")
    print(f"Outputs with conditional logic: {has_conditional_count:5d} ({conditional_pct:5.1f}%)")
    print(f"Outputs with disclaimers: {has_disclaimer_count:5d} ({disclaimer_pct:5.1f}%)")
    
    # 6. FINAL RISK FLAGS
    print("\n6. FINAL RISK FLAGS")
    print("-" * 70)
    
    risks = []
    
    if duplicate_rate > 5:
        risks.append(f"Exact duplicates > 5%: {duplicate_rate:.2f}%")
    
    if near_dup_rate > 10:
        risks.append(f"Near-duplicates > 10%: {near_dup_rate:.2f}%")
    
    if conditional_pct < 40:
        risks.append(f"Conditional logic < 40%: {conditional_pct:.1f}%")
    
    if disclaimer_pct < 85:
        risks.append(f"Disclaimers < 85%: {disclaimer_pct:.1f}%")
    
    if risks:
        print("⚠ WARNINGS:")
        for risk in risks:
            print(f"  - {risk}")
        print("\n" + "="*70)
        print("FINAL DATASET HAS TRAINING RISKS")
        print("="*70)
    else:
        print("No risk flags detected.")
        print("\n" + "="*70)
        print("FINAL DATASET PASSES TRAINING QUALITY CHECK")
        print("="*70)


def main():
    """Main execution function."""
    audit_dataset()


if __name__ == '__main__':
    main()

