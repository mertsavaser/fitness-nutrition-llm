# Fitness & Nutrition Coach LLM

Fine-tuned language model for fitness and nutrition coaching, providing personalized meal planning and exercise guidance with numeric grounding and safety disclaimers. The model is trained on Turkish and English instruction-following data using QLoRA fine-tuning on Qwen2.5-3B-Instruct.

**Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/mertsavaser/fitness-nutrition-coach)

## Project Overview

This repository contains the complete machine learning pipeline for a fitness and nutrition coaching assistant. The project demonstrates the full ML lifecycle: dataset generation, cleaning, quality auditing, LoRA fine-tuning, and deployment. The model is fine-tuned using QLoRA (4-bit quantization) on Qwen2.5-3B-Instruct to provide practical, numeric-based fitness and nutrition guidance.

**Important**: This repository contains the training pipeline, dataset preparation scripts, and source code. The Hugging Face Space deployment uses the inference code from this repository but runs separately on Hugging Face infrastructure. Training and dataset preparation are performed in this repository; the Hugging Face Space is used exclusively for inference and demonstration.

## Key Features

- Instruction-following fitness and nutrition coaching with structured, actionable advice
- Numeric reasoning with specific values for calories, macros, meal plans, and body measurements
- Safety-aware responses with appropriate disclaimers to avoid medical advice claims
- Cleaned and audited dataset with comprehensive quality checks ensuring high-quality training data
- Parameter-efficient fine-tuning using QLoRA for cost-effective training on consumer hardware

## Live Demo

The model is deployed on Hugging Face Spaces with a Gradio interface:

**[Fitness & Nutrition Coach - Hugging Face Space](https://huggingface.co/spaces/mertsavaser/fitness-nutrition-coach)**

### CPU-Only Inference and Latency

The Hugging Face Space runs on shared CPU infrastructure. This has important implications for inference performance:

- **First Response Latency**: Initial requests may take several minutes due to model loading and cold start on CPU hardware
- **Subsequent Responses**: While faster than the first request, responses still take significantly longer than GPU-based inference
- **Infrastructure Limitation**: This latency is a limitation of the shared CPU infrastructure, not a model performance issue
- **Generation Length**: Response length is intentionally limited to maintain stability on CPU resources

For production use cases requiring low latency, GPU-based inference (dedicated endpoints, RunPod, or similar services) is recommended. The model itself is optimized for inference; the latency observed in the Hugging Face Space demo is due to CPU-only execution on shared infrastructure.

## Dataset Pipeline

The training dataset is generated through a multi-stage pipeline that ensures quality, diversity, and adherence to safety standards.

### 1. Dataset Generation

Raw synthetic data is generated from seed examples using OpenAI's GPT-4o-mini:

```bash
python scripts/generate_dataset.py
```

This script:
- Loads seed examples from `data/seeds/seed_examples.jsonl`
- Generates variations using OpenAI API with structured prompts
- Ensures each sample includes numeric values, conditional logic, and safety disclaimers
- Outputs raw generated samples to `data/raw_generated/`

The generation process uses carefully crafted prompts to ensure:
- Numeric specificity (calories, grams, kg, cm, age, etc.)
- Conditional logic (if/then statements for different scenarios)
- Safety disclaimers in all outputs
- Turkish and English language support

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
- Filters out samples without proper structure

The cleaning process uses `scripts/utils.py` for validation functions including language detection, unsafe keyword filtering, and format validation.

### 3. Dataset Merging

Multiple cleaned datasets are merged into a unified training set:

```bash
python scripts/merge_final_dataset.py
```

This script:
- Loads multiple cleaned dataset files
- Validates required fields for each sample
- Merges datasets while maintaining data integrity
- Outputs unified dataset for further processing

### 4. Deduplication

Exact and near-duplicates are removed while preserving diversity:

```bash
python scripts/dedupe_final_dataset.py
```

The deduplication process:
- Removes exact duplicates (identical instruction + input + output)
- Removes near-duplicates (input similarity > 0.9 AND output similarity > 0.9)
- Preserves samples with higher quality scores based on:
  - More conditional logic occurrences
  - Higher numeric specificity
  - More natural user-like phrasing
  - Presence of safety disclaimers
- Maintains diversity in user-like prompts, reassurance queries, and myth-busting samples

The deduplication algorithm uses Jaccard similarity for text comparison and quality scoring to determine which samples to retain when duplicates are found.

### 5. Quality Auditing

Comprehensive audits verify dataset quality before training:

```bash
python scripts/audit_dataset.py          # General quality audit
python scripts/final_audit_dataset.py    # Final training readiness check
```

Audit metrics include:
- Language distribution (Turkish, English, mixed)
- Sample type distribution (structured, low-effort, reassurance)
- Duplication rates (exact and near-duplicates)
- Quality metrics (numeric values, conditional logic, disclaimers)
- Length distributions (instruction, input, output)
- Risk flags for training readiness

Audit reports are saved to `reports/` directory for documentation and quality tracking.

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

The dataset was reduced from 2,727 samples to 2,303 samples through deduplication, removing 424 exact duplicates (15.55% of original dataset).

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

Qwen2.5-3B-Instruct was selected for its strong instruction-following capabilities, efficient inference, and good performance on multilingual tasks including Turkish and English.

### Fine-Tuning Approach

**Method**: QLoRA (Quantized Low-Rank Adaptation)

QLoRA was chosen over full fine-tuning for several technical and practical reasons:

1. **Parameter Efficiency**: LoRA trains only a small subset of parameters (LoRA adapters) while keeping the base model frozen. This reduces trainable parameters from 3B to approximately 8M (0.27% of original), dramatically reducing memory and compute requirements.

2. **Memory Efficiency**: 4-bit quantization via bitsandbytes reduces memory requirements from ~12GB (FP16) to ~3-4GB (4-bit), enabling training on consumer GPUs and reducing cloud compute costs.

3. **Cost-Effectiveness**: Lower compute requirements translate to significantly lower training costs compared to full fine-tuning, making the project feasible on limited budgets.

4. **Flexibility**: LoRA adapters can be swapped, combined, or removed without retraining the base model, allowing for experimentation and model versioning.

5. **Data Efficiency**: LoRA fine-tuning requires less data than full fine-tuning while achieving comparable performance on the target domain. The 2,303 sample dataset is appropriate for LoRA but would be insufficient for full fine-tuning.

6. **Prevention of Catastrophic Forgetting**: By keeping the base model frozen, LoRA preserves the general knowledge and capabilities of the pre-trained model while adding domain-specific adaptations.

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

The training dataset size (2,303 samples) is appropriate for LoRA fine-tuning. With 3 epochs, the model sees each sample 3 times, providing sufficient exposure for the adapter weights to learn domain-specific patterns without overfitting.

## Inference & Deployment

The model is deployed on Hugging Face Spaces with a Gradio interface for interactive use. The inference code is located in `model/app.py` and includes:

- Model loading with 4-bit quantization for efficient inference
- LoRA adapter loading from Hugging Face Hub (`mertsavaser/fitness-nutrition-qwen2.5-3b-lora`)
- Gradio interface for user interaction
- Proper chat template formatting for Qwen2.5 models
- Generation parameters optimized for quality responses

**Important**: Training and deployment are intentionally separated. This repository contains the full source code, training pipeline, and dataset preparation scripts. The Hugging Face Space deployment uses the inference code from `model/app.py` but is configured for cloud deployment with automatic model loading from the Hub.

This separation provides:
- Independent versioning of training code and inference code
- Clear separation of concerns between development and deployment
- Easier maintenance and updates to either component
- Better organization for portfolio presentation and code review

The Hugging Face Space serves as a demonstration and testing platform. For production deployments requiring low latency, GPU-based inference infrastructure is recommended.

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

### Directory Explanations

- **`data/`**: Contains all dataset files
  - `cleaned/`: Final training-ready datasets after cleaning and deduplication
  - `seeds/`: Seed examples used as templates for synthetic data generation

- **`model/`**: Inference code and deployment configuration
  - `app.py`: Gradio-based inference application for Hugging Face Spaces
  - `requirements.txt`: Dependencies specific to inference (transformers, gradio, etc.)

- **`scripts/`**: Dataset preparation and processing pipeline
  - `generate_dataset.py`: Synthetic data generation using OpenAI API
  - `clean_dataset.py`: Validation and filtering of generated samples
  - `audit_dataset.py`: Comprehensive quality auditing
  - `final_audit_dataset.py`: Final training readiness validation
  - `dedupe_final_dataset.py`: Duplicate removal with quality preservation
  - `merge_final_dataset.py`: Dataset merging and consolidation
  - `api_health_check.py`: OpenAI API connectivity and quota verification
  - `utils.py`: Shared utility functions for validation and processing

- **`reports/`**: Quality audit reports and dataset statistics
  - `dataset_audit.txt`: General quality metrics and analysis
  - `final_dedup_report.txt`: Deduplication statistics and final quality metrics

### Requirements Files

This repository contains two `requirements.txt` files serving different purposes:

**Root `requirements.txt`**: Dependencies for dataset generation and processing scripts
- `openai`: OpenAI API client for dataset generation
- `python-dotenv`: Environment variable management for API keys
- `tqdm`: Progress bars for dataset processing operations
- `langdetect`: Language detection for validation and filtering

**`model/requirements.txt`**: Dependencies for inference and deployment
- `torch`: PyTorch for model inference and tensor operations
- `transformers`: Hugging Face Transformers library for model loading
- `peft`: Parameter-Efficient Fine-Tuning library (LoRA implementation)
- `accelerate`: Hugging Face Accelerate for distributed inference
- `bitsandbytes`: 4-bit quantization for efficient model loading
- `gradio`: Web UI framework for interactive model demonstration

This separation allows for:
- Independent dependency management for different use cases
- Smaller dependency sets for specific workflows
- Clear distinction between development and deployment environments

## Tech Stack

- **Python**: Core programming language
- **Hugging Face Transformers**: Model loading, inference, and chat template handling
- **PEFT (LoRA)**: Parameter-efficient fine-tuning implementation
- **Gradio**: Web UI framework for model interaction
- **PyTorch**: Deep learning framework for model operations
- **Qwen2.5**: Base model architecture and tokenization
- **OpenAI API**: Dataset generation using GPT-4o-mini
- **bitsandbytes**: 4-bit quantization for memory-efficient training and inference

## Setup

### For Dataset Generation and Processing

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API key:
   - Create `.env` file in project root
   - Add: `OPENAI_API_KEY=your_key_here`

3. Run dataset pipeline:
```bash
python scripts/generate_dataset.py    # Generate raw data from seeds
python scripts/clean_dataset.py       # Clean and validate samples
python scripts/merge_final_dataset.py # Merge multiple datasets
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

The inference script loads the base model and LoRA adapters from Hugging Face Hub. For local testing, ensure you have sufficient GPU memory (4-bit quantization requires ~3-4GB VRAM). CPU inference is possible but will be significantly slower.

## Technical Details

### Dataset Quality Metrics

The final training dataset meets the following quality thresholds:

- **Numeric value coverage**: 99.8% (target: ≥95%)
- **Conditional logic coverage**: 99.5% (target: ≥90%)
- **Disclaimer coverage**: 92.4% (target: ≥90%)
- **Exact duplicate rate**: 0% (target: ≤5%)
- **Near-duplicate rate**: <10% (target: ≤10%)

These metrics are verified through automated auditing scripts and documented in `reports/final_dedup_report.txt`.

### Validation Criteria

All samples in the training dataset must meet these criteria:

- Include numeric values (calories, grams, kg, cm, age, etc.)
- Contain conditional logic (if/then statements for different scenarios)
- Include safety disclaimers (medical advice disclaimers)
- Be in Turkish or English (mixed allowed for technical terms)
- Avoid medical advice or unsafe recommendations
- Meet minimum length requirements (100 characters for output)

### Model Performance

The fine-tuned model demonstrates:

- Accurate numeric reasoning for calories and macros
- Appropriate conditional logic for different user scenarios
- Consistent safety disclaimers in responses
- Natural language generation in Turkish and English
- Domain-specific knowledge for fitness and nutrition guidance

## Notes

This project demonstrates a complete machine learning lifecycle from data preparation to deployment:

1. **Dataset Preparation**: Synthetic data generation, cleaning, deduplication, and quality assurance
2. **Training**: LoRA fine-tuning on a base instruction-tuned model using QLoRA
3. **Deployment**: Inference deployment on Hugging Face Spaces with Gradio UI

Training and deployment are intentionally separated. This repository contains the full source code, training pipeline, and dataset preparation scripts. The Hugging Face Space deployment uses the inference code from `model/app.py` but is configured for cloud deployment.

This separation allows for:
- Clear demonstration of the full ML lifecycle
- Independent versioning of training and inference code
- Easier maintenance and updates
- Better organization for portfolio presentation
- Clear distinction between development and production environments

## Disclaimer

This project is for educational and research purposes only. The model provides general fitness and nutrition guidance and does not constitute medical advice. Users should consult healthcare professionals for personalized medical recommendations. The model is trained on synthetic data and may not reflect real-world medical or nutritional expertise.

## License

This project is provided as-is for portfolio and research purposes.
