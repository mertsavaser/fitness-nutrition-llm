# Fitness & Nutrition Coach LLM (Turkish)

Fine-tuned language model for Turkish fitness and nutrition coaching, providing personalized meal planning and exercise guidance with numeric grounding and safety disclaimers.

## Overview

This project contains the dataset and training pipeline for a Turkish-language fitness coaching assistant. The model is trained to provide detailed, numeric-based nutrition and fitness advice with conditional logic and appropriate medical disclaimers.

## Dataset

The training dataset (`data/cleaned/train_TRAINING_READY.jsonl`) contains 2,303 instruction-following samples with the following characteristics:

- **Total samples**: 2,303
- **Language**: Turkish-dominant (mixed Turkish/English)
- **Prompt types**:
  - Structured prompts with detailed scenarios
  - Low-effort user queries (casual, incomplete)
  - Reassurance and myth-busting queries
- **Quality guarantees**:
  - Numeric grounding: All outputs include specific values (calories, grams, kg, cm, age)
  - Conditional logic: Outputs contain if/then reasoning for different scenarios
  - Safety disclaimers: Medical advice disclaimers included in all outputs

### Dataset Format

Each sample follows JSONL format with three fields:
- `instruction`: Task description or context
- `input`: User query or scenario
- `output`: Coach's response with numeric values, conditional logic, and disclaimers

## Model Training

### Base Model
- **Model**: Qwen2.5-3B-Instruct
- **Size**: 3B parameters
- **Architecture**: Instruction-tuned transformer

### Training Configuration
- **Method**: QLoRA (4-bit quantization)
- **Target Modules**: Attention layers (q_proj, k_proj, v_proj, o_proj)
- **LoRA Rank**: 64
- **LoRA Alpha**: 128
- **Batch Size**: 4
- **Learning Rate**: 2e-4
- **Epochs**: 3

### Deployment
- **Platform**: RunPod (GPU cloud training)
- **Hardware**: A100 40GB or equivalent
- **UI**: Gradio interface (planned)

## Repository Structure

```
fitcoachlm/
├── data/
│   ├── cleaned/
│   │   └── train_TRAINING_READY.jsonl  # Final training dataset
│   └── seeds/
│       └── seed_examples.jsonl         # Seed examples for generation
├── scripts/
│   ├── generate_dataset.py            # Dataset generation from seeds
│   ├── clean_dataset.py               # Dataset cleaning and validation
│   ├── audit_dataset.py               # Dataset quality audit
│   ├── final_audit_dataset.py         # Final dataset audit
│   ├── dedupe_final_dataset.py        # Deduplication pipeline
│   ├── merge_final_dataset.py         # Dataset merging
│   ├── api_health_check.py            # API health verification
│   └── utils.py                       # Utility functions
├── reports/
│   └── final_dedup_report.txt         # Deduplication report
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure API key** (for dataset generation):
   - Create `.env` file in project root
   - Add: `OPENAI_API_KEY=your_key_here`

## Usage

### Dataset Generation

Generate synthetic variations from seed examples:
```bash
python scripts/generate_dataset.py
```

### Dataset Cleaning

Clean and validate generated dataset:
```bash
python scripts/clean_dataset.py
```

### Dataset Audit

Run quality audit on dataset:
```bash
python scripts/audit_dataset.py
```

### Dataset Deduplication

Remove exact and near-duplicates:
```bash
python scripts/dedupe_final_dataset.py
```

## Technical Details

### Dataset Quality Metrics

- Exact duplicate rate: <5%
- Near-duplicate rate: <10%
- Numeric value coverage: >95%
- Conditional logic coverage: >90%
- Disclaimer coverage: >90%

### Validation Criteria

All samples must:
- Include numeric values (calories, grams, etc.)
- Contain conditional logic (if/then statements)
- Include safety disclaimers
- Be in Turkish (with English allowed for technical terms)
- Avoid medical advice or unsafe recommendations

## Disclaimer

This project is for educational and research purposes only. The model provides general fitness and nutrition guidance and does not constitute medical advice. Users should consult healthcare professionals for personalized medical recommendations.

## License

This project is provided as-is for portfolio and research purposes.
