"""
Comprehensive dataset audit script.
Read-only analysis of train_final.jsonl before model training.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / 'data' / 'cleaned' / 'train_final.jsonl'
OUTPUT_REPORT_PATH = BASE_DIR / 'reports' / 'dataset_audit.txt'

# Turkish characters for language detection
TURKISH_CHARS = set('çğışüÇĞIŞÜ')
TURKISH_KEYWORDS = ['kalori', 'gram', 'protein', 'karbonhidrat', 'yağ', 'egzersiz', 'antrenman', 'beslenme', 'diyet', 'kilo', 'tıbbi', 'hocam', 'kanka']
ENGLISH_KEYWORDS = ['calorie', 'gram', 'protein', 'carbohydrate', 'fat', 'exercise', 'workout', 'nutrition', 'diet', 'weight', 'medical', 'coach']

# Patterns for quality checks
IF_THEN_PATTERNS_TR = [
    r'Eğer\s+[^.]*\s+ise',
    r'Eğer\s+[^.]*\s+olursa',
    r'Eğer\s+[^.]*\s+yaparsan',
]
IF_THEN_PATTERNS_EN = [
    r'If\s+you\s+[^.]*',
    r'If\s+[^.]*,\s+you',
    r'When\s+[^.]*,\s+you',
]
DISCLAIMER_PATTERNS = [
    r'tıbbi\s+(tavsiye|öneri|müdahale)',
    r'not\s+medical\s+advice',
    r'does\s+not\s+(constitute|replace)\s+medical',
    r'general\s+(fitness|guidance)',
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
    """
    Detect language of text.
    Returns: 'tr', 'en', 'mixed', or 'unknown'
    """
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
    elif has_turkish_chars:
        return 'tr'
    else:
        return 'unknown'


def has_numbers(text: str) -> bool:
    """Check if text contains numeric values."""
    return bool(re.search(r'\d+', text))


def has_conditional_logic(text: str) -> bool:
    """Check if text contains if/then conditional logic."""
    patterns = IF_THEN_PATTERNS_TR + IF_THEN_PATTERNS_EN
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def has_disclaimer(text: str) -> bool:
    """Check if text contains safety disclaimer."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in DISCLAIMER_PATTERNS)


def is_low_effort(input_text: str, instruction: str) -> bool:
    """Detect if sample looks like low-effort user prompt."""
    combined = (instruction + " " + input_text).lower()
    
    # Indicators of low-effort prompts
    indicators = [
        len(input_text) < 50,  # Very short
        not input_text or not instruction,  # One field empty
        any(casual in combined for casual in ['kanka', 'hocam', 'abi', 'hey', 'selam']),  # Casual greetings
        combined.count('?') > 2,  # Multiple questions
        not re.search(r'\d+', combined),  # No numbers (lazy)
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
    """Calculate simple similarity between two texts (word overlap)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def audit_dataset():
    """Perform comprehensive dataset audit."""
    print("="*70)
    print("DATASET AUDIT: train_final.jsonl")
    print("="*70)
    
    # Load samples
    print("\nLoading dataset...")
    samples = load_samples(INPUT_PATH)
    total_count = len(samples)
    print(f"Loaded {total_count} samples.\n")
    
    if not samples:
        print("ERROR: No samples found.")
        return
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("DATASET AUDIT REPORT: train_final.jsonl")
    report_lines.append("="*70)
    report_lines.append(f"\nAnalysis Date: {Path(__file__).stat().st_mtime}")
    report_lines.append(f"Total Samples: {total_count}\n")
    
    # 1. BASIC STATS
    print("1. BASIC STATISTICS")
    print("-" * 70)
    report_lines.append("\n1. BASIC STATISTICS")
    report_lines.append("-" * 70)
    
    instruction_lengths = [len(s.get('instruction', '')) for s in samples]
    input_lengths = [len(s.get('input', '')) for s in samples]
    output_lengths = [len(s.get('output', '')) for s in samples]
    
    def stats_summary(name: str, lengths: List[int]):
        if lengths:
            avg = sum(lengths) / len(lengths)
            min_val = min(lengths)
            max_val = max(lengths)
            print(f"{name:20s}: avg={avg:6.1f}, min={min_val:4d}, max={max_val:5d}")
            report_lines.append(f"{name:20s}: avg={avg:6.1f}, min={min_val:4d}, max={max_val:5d}")
    
    stats_summary("Instruction length", instruction_lengths)
    stats_summary("Input length", input_lengths)
    stats_summary("Output length", output_lengths)
    
    # 2. LANGUAGE DISTRIBUTION
    print("\n2. LANGUAGE DISTRIBUTION")
    print("-" * 70)
    report_lines.append("\n2. LANGUAGE DISTRIBUTION")
    report_lines.append("-" * 70)
    
    language_counts = Counter()
    mixed_count = 0
    
    for sample in samples:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        # Detect language per field
        inst_lang = detect_language(instruction)
        input_lang = detect_language(input_text)
        output_lang = detect_language(output)
        
        # Overall language (use output as primary)
        overall_lang = output_lang
        if overall_lang == 'unknown':
            overall_lang = input_lang
        if overall_lang == 'unknown':
            overall_lang = inst_lang
        
        language_counts[overall_lang] += 1
        
        # Check for mixed
        langs = {inst_lang, input_lang, output_lang}
        if 'mixed' in langs or (langs & {'tr', 'en'} == {'tr', 'en'}):
            mixed_count += 1
    
    for lang, count in language_counts.most_common():
        pct = count / total_count * 100
        print(f"{lang.upper():15s}: {count:5d} ({pct:5.1f}%)")
        report_lines.append(f"{lang.upper():15s}: {count:5d} ({pct:5.1f}%)")
    
    if mixed_count > 0:
        pct = mixed_count / total_count * 100
        print(f"MIXED samples: {mixed_count:5d} ({pct:5.1f}%)")
        report_lines.append(f"MIXED samples: {mixed_count:5d} ({pct:5.1f}%)")
    
    # 3. DUPLICATION ANALYSIS
    print("\n3. DUPLICATION ANALYSIS")
    print("-" * 70)
    report_lines.append("\n3. DUPLICATION ANALYSIS")
    report_lines.append("-" * 70)
    
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
    print(f"Duplicate rate: {duplicate_rate:.2f}%")
    report_lines.append(f"Unique samples: {unique_samples}")
    report_lines.append(f"Exact duplicates: {exact_duplicates}")
    report_lines.append(f"Duplicate rate: {duplicate_rate:.2f}%")
    
    # Near-duplicates (sample-based, check input similarity)
    print("\nChecking near-duplicates (input similarity > 0.85)...")
    report_lines.append("\nNear-duplicates (input similarity > 0.85):")
    
    near_duplicate_pairs = []
    checked = set()
    
    for i in range(min(1000, len(samples))):  # Limit to first 1000 for performance
        if i in checked:
            continue
        input1 = (samples[i].get('instruction', '') + ' ' + samples[i].get('input', '')).strip()
        
        for j in range(i + 1, min(i + 100, len(samples))):  # Check next 100 samples
            if j in checked:
                continue
            input2 = (samples[j].get('instruction', '') + ' ' + samples[j].get('input', '')).strip()
            
            similarity = calculate_similarity(input1, input2)
            if similarity > 0.85:
                near_duplicate_pairs.append((i, j, similarity))
                checked.add(j)
    
    near_dup_count = len(near_duplicate_pairs)
    near_dup_rate = near_dup_count / min(1000, len(samples)) * 100 if samples else 0
    
    print(f"Near-duplicate pairs found (sampled): {near_dup_count}")
    print(f"Estimated near-duplicate rate: ~{near_dup_rate:.2f}%")
    report_lines.append(f"Near-duplicate pairs found (sampled): {near_dup_count}")
    report_lines.append(f"Estimated near-duplicate rate: ~{near_dup_rate:.2f}%")
    
    # 4. USER-LIKE VS STRUCTURED
    print("\n4. SAMPLE TYPE DISTRIBUTION")
    print("-" * 70)
    report_lines.append("\n4. SAMPLE TYPE DISTRIBUTION")
    report_lines.append("-" * 70)
    
    structured_count = 0
    low_effort_count = 0
    reassurance_count = 0
    other_count = 0
    
    for sample in samples:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        
        if is_reassurance(instruction, input_text):
            reassurance_count += 1
        elif is_low_effort(input_text, instruction):
            low_effort_count += 1
        elif instruction and input_text and len(instruction) > 10 and len(input_text) > 20:
            structured_count += 1
        else:
            other_count += 1
    
    def print_type(name: str, count: int):
        pct = count / total_count * 100
        print(f"{name:25s}: {count:5d} ({pct:5.1f}%)")
        report_lines.append(f"{name:25s}: {count:5d} ({pct:5.1f}%)")
    
    print_type("Structured prompts", structured_count)
    print_type("Low-effort user prompts", low_effort_count)
    print_type("Reassurance/myth-busting", reassurance_count)
    print_type("Other/Uncategorized", other_count)
    
    # 5. OUTPUT QUALITY CHECKS
    print("\n5. OUTPUT QUALITY METRICS")
    print("-" * 70)
    report_lines.append("\n5. OUTPUT QUALITY METRICS")
    report_lines.append("-" * 70)
    
    has_numbers_count = sum(1 for s in samples if has_numbers(s.get('output', '')))
    has_conditional_count = sum(1 for s in samples if has_conditional_logic(s.get('output', '')))
    has_disclaimer_count = sum(1 for s in samples if has_disclaimer(s.get('output', '')))
    
    def print_metric(name: str, count: int):
        pct = count / total_count * 100
        status = "OK" if pct >= 95 else "WARNING" if pct >= 80 else "CRITICAL"
        print(f"{name:30s}: {count:5d} ({pct:5.1f}%) [{status}]")
        report_lines.append(f"{name:30s}: {count:5d} ({pct:5.1f}%) [{status}]")
    
    print_metric("Outputs with numeric values", has_numbers_count)
    print_metric("Outputs with if/then logic", has_conditional_count)
    print_metric("Outputs with disclaimers", has_disclaimer_count)
    
    # Output length analysis
    short_outputs = sum(1 for length in output_lengths if length < 100)
    long_outputs = sum(1 for length in output_lengths if length > 2000)
    
    print(f"\nOutput length distribution:")
    print(f"  Short outputs (<100 chars): {short_outputs} ({short_outputs/total_count*100:.1f}%)")
    print(f"  Long outputs (>2000 chars): {long_outputs} ({long_outputs/total_count*100:.1f}%)")
    report_lines.append(f"\nOutput length distribution:")
    report_lines.append(f"  Short outputs (<100 chars): {short_outputs} ({short_outputs/total_count*100:.1f}%)")
    report_lines.append(f"  Long outputs (>2000 chars): {long_outputs} ({long_outputs/total_count*100:.1f}%)")
    
    # 6. RISK FLAGS
    print("\n6. RISK ANALYSIS")
    print("-" * 70)
    report_lines.append("\n6. RISK ANALYSIS")
    report_lines.append("-" * 70)
    
    risks = []
    
    # Check for high duplicate rate
    if duplicate_rate > 5:
        risks.append(f"HIGH duplicate rate: {duplicate_rate:.2f}% (may cause overfitting)")
    
    # Check for low diversity in instructions
    instruction_texts = [s.get('instruction', '') for s in samples if s.get('instruction')]
    if instruction_texts:
        unique_instructions = len(set(instruction_texts))
        instruction_diversity = unique_instructions / len(instruction_texts) * 100
        if instruction_diversity < 50:
            risks.append(f"LOW instruction diversity: {instruction_diversity:.1f}% unique")
    
    # Check for repetitive outputs (first 100 chars)
    output_starts = [s.get('output', '')[:100] for s in samples]
    output_start_counts = Counter(output_starts)
    common_starts = sum(count for count in output_start_counts.values() if count > 10)
    if common_starts > total_count * 0.1:
        risks.append(f"REPETITIVE output starts: {common_starts} samples share common beginnings")
    
    # Check disclaimer coverage
    if has_disclaimer_count / total_count < 0.95:
        risks.append(f"LOW disclaimer coverage: {has_disclaimer_count/total_count*100:.1f}% (target: ≥95%)")
    
    # Check conditional logic coverage
    if has_conditional_count / total_count < 0.80:
        risks.append(f"LOW conditional logic coverage: {has_conditional_count/total_count*100:.1f}%")
    
    if risks:
        print("RISK FLAGS DETECTED:")
        for i, risk in enumerate(risks, 1):
            print(f"  {i}. {risk}")
            report_lines.append(f"RISK {i}: {risk}")
    else:
        print("No major risk flags detected.")
        report_lines.append("No major risk flags detected.")
    
    # Summary
    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)
    report_lines.append("\n" + "="*70)
    report_lines.append("AUDIT SUMMARY")
    report_lines.append("="*70)
    
    summary = f"""
Total Samples: {total_count}
Language: {language_counts.most_common(1)[0][0].upper()} ({language_counts.most_common(1)[0][1]} samples)
Duplicate Rate: {duplicate_rate:.2f}%
Quality Metrics:
  - Numeric values: {has_numbers_count/total_count*100:.1f}%
  - Conditional logic: {has_conditional_count/total_count*100:.1f}%
  - Disclaimers: {has_disclaimer_count/total_count*100:.1f}%
Risk Flags: {len(risks)}
"""
    print(summary)
    report_lines.append(summary)
    
    # Save report
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nFull report saved to: {OUTPUT_REPORT_PATH}")


def main():
    """Main execution function."""
    if not INPUT_PATH.exists():
        print(f"ERROR: Dataset file not found at {INPUT_PATH}")
        return
    
    audit_dataset()


if __name__ == '__main__':
    main()

