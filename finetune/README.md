# Fine-tuning

**TLDR**: Train memory-focused LLMs with automated hyperparameter tuning. Export to GGUF/vLLM formats for deployment.

## Quick Start

```bash
# Train with hyperparameter tuning + export to GGUF
python train.py --max-trials 3 --num-epochs 2 --export-formats gguf

# Train with both export formats
python train.py --export-formats gguf vllm --hf-repo-name "your-username/model-name"

# Skip tuning, use default config
python train.py --skip-tuning --export-formats gguf --num-epochs 3

# Create custom dataset
python create_memory_dataset.py

# Upload to Hugging Face
python upload_to_hf.py --model-path ./trained_model --repo-name "your-username/model-name"
```

## What This Does

- **Hyperparameter Tuning**: Automatically tests LoRA ranks, batch sizes, learning rates
- **Memory Training**: Fine-tunes models on 5,000 memory-focused examples
- **Model Export**: Saves in GGUF (local inference) or vLLM (serving) formats
- **VRAM Monitoring**: Real-time GPU memory tracking and optimization

## Files

- `train.py` - Complete training pipeline with hyperparameter tuning
- `memory_dataset.jsonl` - 5,000 training examples in ChatML format
- `create_memory_dataset.py` - Generate custom training data
- `export_to_gguf.py` - Convert models to GGUF format
- `upload_to_hf.py` - Upload models to Hugging Face

## Requirements

**Minimum**: 6GB VRAM, 16GB RAM  
**Recommended**: 8GB+ VRAM, 32GB RAM

## Output

- Trained model in HuggingFace format
- GGUF model for llama.cpp/Ollama
- vLLM model for high-throughput serving
- Training plots and hyperparameter analysis
- Memory usage reports
