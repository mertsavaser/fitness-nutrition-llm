"""
Generate synthetic instruction dataset from seed examples using OpenAI API.
Batch generation for speed and safety.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
VARIATIONS_PER_SEED = 30
MODEL_NAME = "gpt-4o-mini"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Paths
BASE_DIR = Path(__file__).parent.parent
SEEDS_PATH = BASE_DIR / 'data' / 'seeds' / 'seed_examples.jsonl'
OUTPUT_PATH = BASE_DIR / 'data' / 'raw_generated' / 'generated_raw.jsonl'

# System prompt for dataset generation
SYSTEM_PROMPT = """Sen profesyonel bir fitness ve beslenme koçusun. Türkçe dilinde, sayısal veriler içeren, 
güvenli ve etik fitness tavsiyeleri veriyorsun.

KURALLAR:
1. Sadece Türkçe yanıt ver
2. Mutlaka sayısal değerler içer (kalori, gram, kg, cm, yaş, vb.)
3. Her output mutlaka en az bir ayarlama kuralı içermeli (örn: "Eğer X olursa Y yap")
4. Her output'un sonuna kısa bir güvenlik uyarısı ekle (tıbbi tavsiye niteliği taşımadığını belirt)
5. Tıbbi tavsiye verme, hastalık adı verme, aşırı diyet önerme
6. Profesyonel ve destekleyici bir ton kullan
7. Output formatı: JSONL - her satır bir JSON objesi (instruction, input, output alanları)"""


def load_seeds(file_path: Path) -> List[Dict[str, Any]]:
    """Load seed examples from JSONL file."""
    seeds = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def ensure_clean_output(output_path: Path, expected_samples: int):
    """
    Ensure output file is clean. Delete if exists (partial or complete).
    """
    if output_path.exists():
        print(f"\n⚠ WARNING: Output file already exists at {output_path}")
        print("  Deleting existing file to ensure clean start...")
        output_path.unlink()
        print("  ✓ Existing file deleted\n")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)


def generate_batch_variations(seed: Dict[str, Any], count: int) -> List[Dict[str, Any]]:
    """
    Generate multiple variations of a seed example in a single API call.
    
    Args:
        seed: Original seed example
        count: Number of variations to generate
        
    Returns:
        List of generated variations
    """
    user_prompt = f"""Aşağıdaki örneği temel alarak, tam olarak {count} adet farklı fitness koçluk örneği oluştur.

Örnek:
{{
  "instruction": "{seed['instruction']}",
  "input": "{seed['input']}",
  "output": "{seed['output']}"
}}

KURALLAR:
- Aynı konu kategorisinde kal (kalori, makro, beslenme planı vb.)
- Her örnek farklı sayısal değerler, farklı senaryo veya kullanıcı profili içermeli
- Her output mutlaka sayısal veriler (kalori, gram, kg, cm, yaş vb.) içermeli
- Her output mutlaka en az bir ayarlama kuralı içermeli (örn: "Eğer haftada 1 kg vermek isterseniz...", "Eğer antrenman sonrası 30 dakika içinde...")
- Her output'un sonuna kısa güvenlik uyarısı ekle
- Profesyonel ve destekleyici tonu koru
- Türkçe dilinde yanıt ver

Çıktı formatı: JSONL formatında, her satır bir JSON objesi. Her obje şu alanları içermeli: instruction, input, output
Sadece JSONL formatında yanıt ver, başka açıklama yapma. Tam olarak {count} adet örnek oluştur."""

    # Retry logic: up to 2 attempts total (initial + 1 retry)
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=4096
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if '```jsonl' in generated_text:
                generated_text = generated_text.split('```jsonl')[1].split('```')[0].strip()
            elif '```json' in generated_text:
                generated_text = generated_text.split('```json')[1].split('```')[0].strip()
            elif '```' in generated_text:
                generated_text = generated_text.split('```')[1].split('```')[0].strip()
            
            # Parse JSONL
            variations = []
            for line in generated_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    variation = json.loads(line)
                    # Validate required fields
                    if not all(key in variation for key in ['instruction', 'input', 'output']):
                        continue
                    variations.append(variation)
                except json.JSONDecodeError:
                    continue
            
            # Check for low yield
            expected_min = int(count * 0.9)
            if len(variations) < expected_min:
                print(f"  ⚠ Low yield: generated {len(variations)}/{count} samples for this seed")
            
            return variations
            
        except Exception as e:
            if attempt == 0:  # First failure, wait 2s and retry
                print(f"  ⚠ Attempt 1 failed, retrying in 2s... ({e})")
                time.sleep(2)
            else:  # Second failure, wait 6s then give up
                print(f"  ⚠ Attempt 2 failed, waiting 6s before giving up... ({e})")
                time.sleep(6)
                print(f"  ✗ Error generating batch for seed '{seed.get('instruction', 'unknown')}' after 2 attempts: {e}")
                return []
    
    return []


def generate_dataset(seeds: List[Dict[str, Any]], variations_per_seed: int):
    """
    Generate variations for all seed examples using batch generation.
    
    Args:
        seeds: List of seed examples
        variations_per_seed: Number of variations to generate per seed
    """
    total_expected = len(seeds) * variations_per_seed
    print(f"\n📊 Dataset Generation Plan:")
    print(f"  - Seeds loaded: {len(seeds)}")
    print(f"  - Variations per seed: {variations_per_seed}")
    print(f"  - Total expected samples: {total_expected}")
    print(f"  - Model: {MODEL_NAME}\n")
    
    all_generated = []
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"Processing seed {seed_idx}/{len(seeds)}: {seed['instruction']}")
        
        variations = generate_batch_variations(seed, variations_per_seed)
        if variations:
            all_generated.extend(variations)
            print(f"  ✓ Generated {len(variations)} variations")
        else:
            print(f"  ✗ Failed to generate variations")
    
    # Save all generated samples
    print(f"\n💾 Saving {len(all_generated)} generated samples to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for sample in all_generated:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ Dataset generation complete!")
    print(f"  - Samples written: {len(all_generated)}")
    print(f"  - Expected: {total_expected}")
    print(f"  - Output: {OUTPUT_PATH}")


def main():
    """Main execution function."""
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file.")
        return
    
    # Load seeds
    if not SEEDS_PATH.exists():
        print(f"ERROR: Seed file not found at {SEEDS_PATH}")
        return
    
    seeds = load_seeds(SEEDS_PATH)
    if not seeds:
        print(f"ERROR: No seeds loaded from {SEEDS_PATH}")
        return
    
    # Ensure clean output
    total_expected = len(seeds) * VARIATIONS_PER_SEED
    ensure_clean_output(OUTPUT_PATH, total_expected)
    
    # Generate dataset
    generate_dataset(seeds, VARIATIONS_PER_SEED)


if __name__ == '__main__':
    main()
