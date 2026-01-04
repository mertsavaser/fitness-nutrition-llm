"""
Merge train_full.jsonl and train_userlike_fixed.jsonl into final training dataset.
"""

import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
STRUCTURED_PATH = BASE_DIR / 'data' / 'cleaned' / 'train_full.jsonl'
USERLIKE_PATH = BASE_DIR / 'data' / 'cleaned' / 'train_userlike_fixed.jsonl'
OUTPUT_PATH = BASE_DIR / 'data' / 'cleaned' / 'train_final.jsonl'

# Required fields for validation
REQUIRED_FIELDS = ['instruction', 'input', 'output']


def load_and_validate_jsonl(file_path: Path, file_label: str) -> list:
    """
    Load samples from JSONL file with validation.
    
    Args:
        file_path: Path to JSONL file
        file_label: Label for logging
        
    Returns:
        List of valid samples
    """
    samples = []
    invalid_count = 0
    
    if not file_path.exists():
        print(f"WARNING: {file_label} file not found at {file_path}")
        return samples
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                
                # Validate required fields
                if not all(key in sample for key in REQUIRED_FIELDS):
                    print(f"  WARNING: Line {line_num} missing required fields, skipping...")
                    invalid_count += 1
                    continue
                
                samples.append(sample)
                
            except json.JSONDecodeError as e:
                print(f"  WARNING: Line {line_num} invalid JSON, skipping... ({e})")
                invalid_count += 1
                continue
            except Exception as e:
                print(f"  WARNING: Line {line_num} error, skipping... ({e})")
                invalid_count += 1
                continue
    
    if invalid_count > 0:
        print(f"  Skipped {invalid_count} invalid lines")
    
    return samples


def merge_final_datasets():
    """Merge structured and user-like datasets into final training file."""
    print(f"Loading structured dataset from {STRUCTURED_PATH.name}...")
    structured_samples = load_and_validate_jsonl(STRUCTURED_PATH, "Structured dataset")
    structured_count = len(structured_samples)
    print(f"  Structured samples loaded: {structured_count}")
    
    print(f"\nLoading user-like fixed dataset from {USERLIKE_PATH.name}...")
    userlike_samples = load_and_validate_jsonl(USERLIKE_PATH, "User-like dataset")
    userlike_count = len(userlike_samples)
    print(f"  User-like samples loaded: {userlike_count}")
    
    if structured_count == 0 and userlike_count == 0:
        print("\nERROR: No valid samples found in either file.")
        return
    
    # Combine samples (structured first, then user-like)
    all_samples = structured_samples + userlike_samples
    total_count = len(all_samples)
    
    print(f"\nMerging datasets...")
    print(f"  Total merged samples: {total_count}")
    
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Write merged dataset
    print(f"\nWriting final merged dataset to {OUTPUT_PATH.name}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nMerge complete!")
    print(f"  - Structured samples count: {structured_count}")
    print(f"  - User-like samples count: {userlike_count}")
    print(f"  - Total samples count: {total_count}")
    print(f"  - Output: {OUTPUT_PATH}")


def main():
    """Main execution function."""
    if not STRUCTURED_PATH.exists():
        print(f"ERROR: Structured dataset not found at {STRUCTURED_PATH}")
        print("  Please ensure data/cleaned/train_full.jsonl exists.")
        return
    
    if not USERLIKE_PATH.exists():
        print(f"ERROR: User-like dataset not found at {USERLIKE_PATH}")
        print("  Please run fix_disclaimers.py first to generate train_userlike_fixed.jsonl.")
        return
    
    merge_final_datasets()


if __name__ == '__main__':
    main()
