# Fitness & Nutrition Coach LLM

Fine-tuned language model for fitness and nutrition coaching, providing personalized meal planning and exercise guidance with numeric grounding and safety disclaimers. The model is trained on Turkish and English instruction-following data using LoRA fine-tuning on Qwen2.5-3B-Instruct.

## Project Overview

This repository contains the complete machine learning pipeline for a fitness and nutrition coaching assistant. The project demonstrates the full ML lifecycle: dataset generation, cleaning, quality auditing, LoRA fine-tuning, and deployment. The model is fine-tuned using QLoRA (4-bit quantization) on Qwen2.5-3B-Instruct to provide practical, numeric-based fitness and nutrition guidance.

The model is deployed as a Hugging Face Space with a Gradio interface. This repository contains the source code, training pipeline, and dataset preparation scripts. The inference code is deployed separately on Hugging Face Spaces.

## Key Features

- **Instruction-following fitness & nutrition coaching**: Responds to user queries with structured, actionable advice
- **Numeric reasoning**: Provides specific values for calories, macros, meal plans, and body measurements
- **Safety-aware responses**: Includes appropriate disclaimers to avoid medical advice claims
- **Cleaned and audited dataset**: Comprehensive quality checks ensure high-quality training data
- **Efficient fine-tuning**: Uses QLoRA for parameter-efficient training on consumer hardware

## Dataset Pipeline

The training dataset is generated through a multi-stage pipeline that ensures quality and diversity.

### 1. Dataset Generation

Raw synthetic data is generated from seed examples using OpenAI's GPT-4o-mini:

```bash
python scripts/generate_dataset.py
```

This script:
- Loads seed examples from `data/seeds/seed_examples.jsonl`
- Generates variations using OpenAI API with structured prompts
- Ensures each sample includes numeric values, conditional logic, and safety disclaimers
- Outputs raw generated samples

### 2. Dataset Cleaning

Generated samples are filtered to remove invalid, unsafe, or low-quality content:

```bash
python scripts/clean_dataset.py
```

Cleaning criteria:
- Valid JSON format with required fields (instruction, input, output)
- Contains numeric values (calories, grams, kg, cm, age, etc.)
- Minimum output length (100 characters)
- Turkish language validation
- Removes unsafe medical keywords and disease references

### 3. Dataset Merging

Multiple cleaned datasets are merged into a unified training set:

```bash
python scripts/merge_final_dataset.py
```

### 4. Deduplication

Exact and near-duplicates are removed while preserving diversity:

```bash
python scripts/dedupe_final_dataset.py
```

The deduplication process:
- Removes exact duplicates (identical instruction + input + output)
- Removes near-duplicates (input similarity > 0.9 AND output similarity > 0.9)
- Preserves samples with higher quality scores (more conditional logic, numeric specificity, user-like phrasing)
- Maintains diversity in user-like prompts, reassurance queries, and myth-busting samples

### 5. Quality Auditing

Comprehensive audits verify dataset quality:

```bash
python scripts/audit_dataset.py          # General quality audit
python scripts/final_audit_dataset.py    # Final training readiness check
```

Audit metrics:
- Language distribution (Turkish, English, mixed)
- Sample type distribution (structured, low-effort, reassurance)
- Duplication rates (exact and near-duplicates)
- Quality metrics (numeric values, conditional logic, disclaimers)

### Final Dataset

The training-ready dataset (`data/cleaned/train_TRAINING_READY.jsonl`) contains:
- **Total samples**: 2,303
- **Language**: Turkish-dominant (mixed Turkish/English)
- **Quality metrics**:
  - Numeric value coverage: 99.8%
  - Conditional logic coverage: 99.5%
  - Disclaimer coverage: 92.4%
- **Duplicate rates**:
  - Exact duplicates: 0% (after deduplication)
  - Near-duplicates: <10%

Each sample follows JSONL format:
```json
{
  "instruction": "Task description or context",
  "input": "User query or scenario",
  "output": "Coach's response with numeric values, conditional logic, and disclaimers"
}
```

## Training

### Base Model

- **Model**: Qwen2.5-3B-Instruct
- **Size**: 3B parameters
- **Architecture**: Instruction-tuned transformer with Qwen architecture
- **Source**: Hugging Face Model Hub

### Fine-Tuning Approach

**Method**: QLoRA (Quantized Low-Rank Adaptation)

QLoRA was chosen for several reasons:
- **Parameter efficiency**: Trains only a small subset of parameters (LoRA adapters) while keeping the base model frozen
- **Memory efficiency**: 4-bit quantization reduces memory requirements, enabling training on consumer GPUs
- **Cost-effective**: Lower compute requirements compared to full fine-tuning
- **Flexibility**: LoRA adapters can be swapped without retraining the base model

### Training Configuration

- **Quantization**: 4-bit (bitsandbytes)
- **Target Modules**: Attention layers (q_proj, k_proj, v_proj, o_proj)
- **LoRA Rank**: 64
- **LoRA Alpha**: 128
- **Batch Size**: 4
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Training Platform**: RunPod (GPU cloud training)
- **Hardware**: A100 40GB or equivalent

The training dataset size (2,303 samples) is appropriate for LoRA fine-tuning, which requires less data than full fine-tuning while still achieving good performance on the target domain.

## Inference & Deployment

The model is deployed on Hugging Face Spaces with a Gradio interface for interactive use.

**Hugging Face Space**: [https://huggingface.co/spaces/USERNAME/fitness-nutrition-coach](https://huggingface.co/spaces/USERNAME/fitness-nutrition-coach)

The inference code is located in `model/app.py` and includes:
- Model loading with 4-bit quantization for efficient inference
- LoRA adapter loading from Hugging Face Hub
- Gradio interface for user interaction
- Proper chat template formatting for Qwen2.5 models

This repository contains the source code for the inference application. The Hugging Face Space deployment uses the same code but is configured for cloud deployment with automatic model loading from the Hub.

**Note**: Training and deployment are intentionally separated. This repository demonstrates the full ML lifecycle, while the Hugging Face Space focuses solely on inference. This separation allows for:
- Independent versioning of training code and inference code
- Clear separation of concerns
- Easier maintenance and updates

## Repository Structure

```
fitcoachlm/
├── data/
│   ├── cleaned/
│   │   └── train_TRAINING_READY.jsonl    # Final training dataset (2,303 samples)
│   └── seeds/
│       └── seed_examples.jsonl            # Seed examples for dataset generation
├── model/
│   ├── app.py                             # Inference script for Hugging Face Space
│   └── requirements.txt                   # Inference dependencies (transformers, gradio, etc.)
├── scripts/
│   ├── generate_dataset.py               # Dataset generation from seeds (OpenAI API)
│   ├── clean_dataset.py                  # Dataset cleaning and validation
│   ├── audit_dataset.py                  # Dataset quality audit
│   ├── final_audit_dataset.py            # Final training readiness check
│   ├── dedupe_final_dataset.py          # Deduplication pipeline
│   ├── merge_final_dataset.py            # Dataset merging
│   ├── api_health_check.py               # OpenAI API health verification
│   └── utils.py                          # Utility functions (validation, language detection)
├── reports/
│   ├── dataset_audit.txt                 # General dataset audit report
│   └── final_dedup_report.txt            # Deduplication report with quality metrics
├── requirements.txt                      # Dataset pipeline dependencies (openai, tqdm, etc.)
├── .gitignore                            # Git ignore patterns
└── README.md                             # This file
```

### Requirements Files

This repository contains two `requirements.txt` files for different purposes:

- **Root `requirements.txt`**: Dependencies for dataset generation and processing scripts
  - `openai`: OpenAI API client for dataset generation
  - `python-dotenv`: Environment variable management
  - `tqdm`: Progress bars for dataset processing
  - `langdetect`: Language detection for validation

- **`model/requirements.txt`**: Dependencies for inference and deployment
  - `torch`: PyTorch for model inference
  - `transformers`: Hugging Face Transformers library
  - `peft`: Parameter-Efficient Fine-Tuning (LoRA)
  - `accelerate`: Hugging Face Accelerate for distributed inference
  - `bitsandbytes`: 4-bit quantization
  - `gradio`: Web UI framework

## Tech Stack

- **Python**: Core programming language
- **Hugging Face Transformers**: Model loading and inference
- **PEFT (LoRA)**: Parameter-efficient fine-tuning
- **Gradio**: Web UI for model interaction
- **PyTorch**: Deep learning framework
- **Qwen2.5**: Base model architecture
- **OpenAI API**: Dataset generation (GPT-4o-mini)
- **bitsandbytes**: 4-bit quantization

## Setup

### For Dataset Generation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API key:
   - Create `.env` file in project root
   - Add: `OPENAI_API_KEY=your_key_here`

3. Run dataset pipeline:
```bash
python scripts/generate_dataset.py    # Generate raw data
python scripts/clean_dataset.py       # Clean and validate
python scripts/merge_final_dataset.py # Merge datasets
python scripts/dedupe_final_dataset.py # Remove duplicates
python scripts/final_audit_dataset.py # Final quality check
```

### For Inference

1. Install inference dependencies:
```bash
pip install -r model/requirements.txt
```

2. Run inference locally:
```bash
python model/app.py
```

The inference script loads the base model and LoRA adapters from Hugging Face Hub. For local testing, ensure you have sufficient GPU memory (4-bit quantization requires ~3-4GB VRAM).

## Usage Examples

### Dataset Generation

Generate synthetic variations from seed examples:
```bash
python scripts/generate_dataset.py
```

### Dataset Quality Audit

Run comprehensive quality audit:
```bash
python scripts/audit_dataset.py
```

### Dataset Deduplication

Remove duplicates while preserving diversity:
```bash
python scripts/dedupe_final_dataset.py
```

## Technical Details

### Dataset Quality Metrics

The final training dataset meets the following quality thresholds:
- **Numeric value coverage**: 99.8% (target: ≥95%)
- **Conditional logic coverage**: 99.5% (target: ≥90%)
- **Disclaimer coverage**: 92.4% (target: ≥90%)
- **Exact duplicate rate**: 0% (target: ≤5%)
- **Near-duplicate rate**: <10% (target: ≤10%)

### Validation Criteria

All samples in the training dataset must:
- Include numeric values (calories, grams, kg, cm, age, etc.)
- Contain conditional logic (if/then statements for different scenarios)
- Include safety disclaimers (medical advice disclaimers)
- Be in Turkish or English (mixed allowed for technical terms)
- Avoid medical advice or unsafe recommendations

### Model Performance

The fine-tuned model demonstrates:
- Accurate numeric reasoning for calories and macros
- Appropriate conditional logic for different user scenarios
- Consistent safety disclaimers in responses
- Natural language generation in Turkish and English

## Notes

This project demonstrates a complete machine learning lifecycle from data preparation to deployment:

1. **Dataset Preparation**: Synthetic data generation, cleaning, and quality assurance
2. **Training**: LoRA fine-tuning on a base instruction-tuned model
3. **Deployment**: Inference deployment on Hugging Face Spaces with Gradio UI

Training and deployment are intentionally separated. This repository contains the full source code, training pipeline, and dataset preparation scripts. The Hugging Face Space deployment uses the inference code from `model/app.py` but is configured for cloud deployment.

This separation allows for:
- Clear demonstration of the full ML lifecycle
- Independent versioning of training and inference code
- Easier maintenance and updates
- Better organization for portfolio presentation


## Disclaimer

This project is for educational and research purposes only. The model provides general fitness and nutrition guidance and does not constitute medical advice. Users should consult healthcare professionals for personalized medical recommendations. The model is trained on synthetic data and may not reflect real-world medical or nutritional expertise.

## License

This project is provided as-is for portfolio and research purposes.

## Inference & Performance Notes

This project is deployed on Hugging Face Spaces using **CPU-only inference**.

Due to the size of the base model (Qwen2.5-3B) and the use of LoRA adapters, **response latency can be relatively high**, especially for longer generations.  
This is an expected limitation of running large language models on CPU hardware.

Important notes:
- Do not expect real-time or low-latency responses.
- First requests may take significantly longer due to model loading and cold start.
- Generation length is intentionally limited to maintain stability on CPU.

For production or real-time use cases, **GPU-based inference (e.g. dedicated endpoints or services like RunPod)** is recommended.  
This repository focuses on demonstrating the **full machine learning lifecycle** rather than production-optimized inference performance.

