# Fine-tuning Pipeline

This directory contains the fine-tuning pipeline implementation using Unsloth with QLoRA adapters.

## Setup

The pipeline is configured with:
- **Quantization**: `load_in_4bit=True` for memory efficiency
- **Training Method**: QLoRA adapters (recommended by Unsloth for beginners)
- **Model**: `unsloth/llama-3.1-8b-bnb-4bit` (Unsloth dynamic 4-bit quantized)

## Files

- `train_with_hyperparamter_training.py`: Main fine-tuning pipeline implementation
- `README.md`: This documentation file

## Usage

```python
from finetune_pipeline import FineTunePipeline

# Initialize pipeline
pipeline = FineTunePipeline()

# Load model with QLoRA configuration
model, tokenizer = pipeline.load_model()

# Setup training (dataset preparation needed)
# training_args = pipeline.setup_training_args()
# trainer = pipeline.create_trainer(dataset, training_args)
# trainer.train()
```

## Next Steps

1. Dataset preparation - convert data to proper format for training
2. Training configuration and execution
3. Model evaluation and saving

## Requirements

Make sure to install dependencies from the main requirements.txt:
```bash
pip install -r ../requirements.txt
```
