"""
Intelligent deduplication script for final training-ready dataset.
Removes exact and near-duplicates while preserving diversity.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / 'data' / 'cleaned' / 'train_TRAINING_READY.jsonl'
OUTPUT_PATH = BASE_DIR / 'data' / 'cleaned' / 'train_TRAINING_READY.jsonl'
REPORT_PATH = BASE_DIR / 'reports' / 'final_dedup_report.txt'

# Similarity threshold for near-duplicates
NEAR_DUPLICATE_THRESHOLD = 0.9

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


def count_conditionals(text: str) -> int:
    """Count conditional logic occurrences in text."""
    text_lower = text.lower()
    patterns = CONDITIONAL_PATTERNS_TR + CONDITIONAL_PATTERNS_EN
    return sum(len(re.findall(pattern, text_lower)) for pattern in patterns)


def count_numbers(text: str) -> int:
    """Count numeric values in text."""
    return len(re.findall(r'\d+', text))


def is_user_like(instruction: str, input_text: str) -> bool:
    """Detect if sample is user-like (short, messy, incomplete)."""
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


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts (word overlap)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def sample_quality_score(sample: Dict[str, Any]) -> float:
    """Calculate quality score for a sample (higher = better to keep)."""
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')
    
    score = 0.0
    
    # Bonus for user-like prompts (preserve diversity)
    if is_user_like(instruction, input_text):
        score += 10.0
    
    # Bonus for reassurance/myth-busting
    if is_reassurance(instruction, input_text):
        score += 10.0
    
    # Bonus for conditional logic
    if has_conditional_logic(output):
        score += count_conditionals(output) * 2.0
    
    # Bonus for numeric specificity
    if has_numbers(output):
        score += count_numbers(output) * 0.5
    
    # Bonus for disclaimers
    if has_disclaimer(output):
        score += 5.0
    
    # Bonus for longer outputs (more detailed)
    score += len(output) / 100.0
    
    return score


def remove_exact_duplicates(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Remove exact duplicates, keeping the first occurrence."""
    seen = set()
    deduplicated = []
    removed_count = 0
    
    for sample in samples:
        # Create signature from all fields
        sig = (
            sample.get('instruction', '').strip(),
            sample.get('input', '').strip(),
            sample.get('output', '').strip()
        )
        
        if sig not in seen:
            seen.add(sig)
            deduplicated.append(sample)
        else:
            removed_count += 1
    
    return deduplicated, removed_count


def remove_near_duplicates(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Remove near-duplicates, keeping samples with higher quality scores."""
    if not samples:
        return [], 0
    
    # Calculate quality scores for all samples
    scored_samples = [(sample_quality_score(s), i, s) for i, s in enumerate(samples)]
    
    # Sort by quality score (best first)
    scored_samples.sort(reverse=True, key=lambda x: x[0])
    
    kept_samples = []
    removed_indices = set()
    removed_count = 0
    
    # Process samples in order of quality (best first)
    for quality_score, idx, sample in scored_samples:
        if idx in removed_indices:
            continue
        
        input_text = (sample.get('instruction', '') + ' ' + sample.get('input', '')).strip()
        output_text = sample.get('output', '').strip()
        
        # Check if this sample is a near-duplicate of any kept sample
        is_near_duplicate = False
        
        for kept_sample in kept_samples:
            kept_input = (kept_sample.get('instruction', '') + ' ' + kept_sample.get('input', '')).strip()
            kept_output = kept_sample.get('output', '').strip()
            
            input_sim = calculate_similarity(input_text, kept_input)
            output_sim = calculate_similarity(output_text, kept_output)
            
            # If both input and output are very similar, it's a near-duplicate
            if input_sim > NEAR_DUPLICATE_THRESHOLD and output_sim > NEAR_DUPLICATE_THRESHOLD:
                is_near_duplicate = True
                break
        
        if is_near_duplicate:
            removed_indices.add(idx)
            removed_count += 1
        else:
            kept_samples.append(sample)
    
    return kept_samples, removed_count


def validate_quality(samples: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, float]]:
    """Validate quality metrics after deduplication."""
    if not samples:
        return False, {}
    
    total = len(samples)
    
    has_numbers_count = sum(1 for s in samples if has_numbers(s.get('output', '')))
    has_conditional_count = sum(1 for s in samples if has_conditional_logic(s.get('output', '')))
    has_disclaimer_count = sum(1 for s in samples if has_disclaimer(s.get('output', '')))
    
    metrics = {
        'numeric_pct': has_numbers_count / total * 100,
        'conditional_pct': has_conditional_count / total * 100,
        'disclaimer_pct': has_disclaimer_count / total * 100,
    }
    
    # Check thresholds
    passes = (
        metrics['numeric_pct'] >= 95.0 and
        metrics['conditional_pct'] >= 90.0 and
        metrics['disclaimer_pct'] >= 90.0
    )
    
    return passes, metrics


def deduplicate_dataset():
    """Main deduplication process."""
    print("="*70)
    print("FINAL DATASET DEDUPLICATION")
    print("="*70)
    
    if not INPUT_PATH.exists():
        print(f"\nERROR: Input file not found at {INPUT_PATH}")
        return
    
    # Load samples
    print(f"\nLoading dataset from {INPUT_PATH}...")
    samples = load_samples(INPUT_PATH)
    original_count = len(samples)
    print(f"Loaded {original_count} samples.")
    
    if not samples:
        print("ERROR: No samples found.")
        return
    
    # Step 1: Remove exact duplicates
    print("\n" + "="*70)
    print("STEP 1: Removing exact duplicates...")
    print("="*70)
    samples, exact_removed = remove_exact_duplicates(samples)
    exact_removed_pct = exact_removed / original_count * 100 if original_count > 0 else 0
    print(f"Exact duplicates removed: {exact_removed} ({exact_removed_pct:.2f}%)")
    print(f"Remaining samples: {len(samples)}")
    
    # Step 2: Remove near-duplicates
    print("\n" + "="*70)
    print("STEP 2: Removing near-duplicates...")
    print("="*70)
    samples, near_removed = remove_near_duplicates(samples)
    near_removed_pct = near_removed / original_count * 100 if original_count > 0 else 0
    print(f"Near-duplicates removed: {near_removed} ({near_removed_pct:.2f}%)")
    print(f"Remaining samples: {len(samples)}")
    
    # Step 3: Validate quality
    print("\n" + "="*70)
    print("STEP 3: Validating quality metrics...")
    print("="*70)
    passes, metrics = validate_quality(samples)
    
    print(f"Numeric values: {metrics['numeric_pct']:.1f}% (target: ≥95%)")
    print(f"Conditional logic: {metrics['conditional_pct']:.1f}% (target: ≥90%)")
    print(f"Disclaimers: {metrics['disclaimer_pct']:.1f}% (target: ≥90%)")
    
    if not passes:
        print("\n⚠ WARNING: Quality metrics below thresholds!")
        print("Aborting to preserve dataset quality.")
        return
    
    # Step 4: Save output
    print("\n" + "="*70)
    print("STEP 4: Saving final dataset...")
    print("="*70)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(samples)} samples to {OUTPUT_PATH}")
    
    # Step 5: Generate report
    print("\n" + "="*70)
    print("STEP 5: Generating report...")
    print("="*70)
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    total_removed = exact_removed + near_removed
    final_removed_pct = total_removed / original_count * 100 if original_count > 0 else 0
    final_duplicate_rate = exact_removed / original_count * 100 if original_count > 0 else 0
    final_near_duplicate_rate = near_removed / original_count * 100 if original_count > 0 else 0
    
    report_lines = [
        "="*70,
        "FINAL DATASET DEDUPLICATION REPORT",
        "="*70,
        "",
        f"Input file: {INPUT_PATH}",
        f"Output file: {OUTPUT_PATH}",
        "",
        "SUMMARY",
        "-" * 70,
        f"Original samples: {original_count}",
        f"Final samples: {len(samples)}",
        f"Total removed: {total_removed} ({final_removed_pct:.2f}%)",
        "",
        "DEDUPLICATION DETAILS",
        "-" * 70,
        f"Exact duplicates removed: {exact_removed} ({final_duplicate_rate:.2f}%)",
        f"Near-duplicates removed: {near_removed} ({final_near_duplicate_rate:.2f}%)",
        "",
        "QUALITY METRICS",
        "-" * 70,
        f"Numeric values: {metrics['numeric_pct']:.1f}%",
        f"Conditional logic: {metrics['conditional_pct']:.1f}%",
        f"Disclaimers: {metrics['disclaimer_pct']:.1f}%",
        "",
        "TARGETS",
        "-" * 70,
        f"Exact duplicate rate: {final_duplicate_rate:.2f}% (target: ≤5%)",
        f"Near-duplicate rate: {final_near_duplicate_rate:.2f}% (target: ≤10%)",
        f"Final size: {len(samples)} (target: 2200-2400)",
        "",
        "="*70,
    ]
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to {REPORT_PATH}")
    
    # Final summary
    print("\n" + "="*70)
    print("DEDUPLICATION COMPLETE")
    print("="*70)
    print(f"Original: {original_count} samples")
    print(f"Final: {len(samples)} samples")
    print(f"Removed: {total_removed} samples ({final_removed_pct:.2f}%)")
    print(f"Exact duplicate rate: {final_duplicate_rate:.2f}% (target: ≤5%)")
    print(f"Near-duplicate rate: {final_near_duplicate_rate:.2f}% (target: ≤10%)")
    print(f"\nQuality metrics:")
    print(f"  Numeric values: {metrics['numeric_pct']:.1f}%")
    print(f"  Conditional logic: {metrics['conditional_pct']:.1f}%")
    print(f"  Disclaimers: {metrics['disclaimer_pct']:.1f}%")
    print("="*70)


def main():
    """Main execution function."""
    deduplicate_dataset()


if __name__ == '__main__':
    main()

