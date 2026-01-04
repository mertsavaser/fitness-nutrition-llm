"""
Clean generated dataset by removing invalid, unsafe, or low-quality samples.
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import is_valid_json, validate_sample

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_PATH = BASE_DIR / 'data' / 'raw_generated' / 'generated_full.jsonl'
OUTPUT_PATH = BASE_DIR / 'data' / 'cleaned' / 'train.jsonl'


def load_raw_samples(file_path: Path):
    """Load raw generated samples from JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Check if line is valid JSON
            if not is_valid_json(line):
                print(f"Line {line_num}: Invalid JSON, skipping...")
                continue
            
            try:
                sample = json.loads(line)
                samples.append(sample)
            except Exception as e:
                print(f"Line {line_num}: Error parsing JSON: {e}")
                continue
    
    return samples


def clean_dataset(input_path: Path, output_path: Path, min_output_length: int = 100):
    """
    Clean dataset by filtering out invalid samples.
    
    Args:
        input_path: Path to raw generated JSONL file
        output_path: Path to save cleaned JSONL file
        min_output_length: Minimum length for output field
    """
    print(f"Loading raw samples from {input_path}...")
    raw_samples = load_raw_samples(input_path)
    print(f"Loaded {len(raw_samples)} raw samples.")
    
    print(f"\nCleaning dataset...")
    cleaned_samples = []
    removed_count = 0
    
    for sample in tqdm(raw_samples, desc="Cleaning"):
        if validate_sample(sample, min_output_length=min_output_length):
            cleaned_samples.append(sample)
        else:
            removed_count += 1
    
    # Save cleaned samples
    print(f"\nSaving {len(cleaned_samples)} cleaned samples to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ Cleaning complete!")
    print(f"  - Original samples: {len(raw_samples)}")
    print(f"  - Cleaned samples: {len(cleaned_samples)}")
    print(f"  - Removed samples: {removed_count}")
    print(f"  - Retention rate: {len(cleaned_samples)/len(raw_samples)*100:.1f}%")


def main():
    """Main execution function."""
    if not INPUT_PATH.exists():
        print(f"ERROR: Input file not found at {INPUT_PATH}")
        print("Please run generate_dataset.py first.")
        return
    
    clean_dataset(INPUT_PATH, OUTPUT_PATH, min_output_length=100)


if __name__ == '__main__':
    main()

