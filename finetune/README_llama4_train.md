# Llama 4 Scout Memory Fine-tuning Guide

This guide covers using the `llama_4_train.py` script to fine-tune Llama 4 Scout on the memory dataset.

## Quick Start

### Prerequisites
1. **GPU Requirements**: NVIDIA GPU with at least 12GB VRAM (16GB+ recommended)
2. **Hugging Face Account**: Access to Llama 4 Scout model
3. **Memory Dataset**: The `memory_dataset.jsonl` file in the same directory

### Setup

1. **Install Dependencies**:
```bash
pip install -r ../requirements.txt
```

2. **Set Hugging Face Token**:
```bash
export HF_TOKEN=your_huggingface_token_here
```

3. **Verify Dataset**:
```bash
ls -la memory_dataset.jsonl
# Should show the 5000-line memory training dataset
```

### Running the Fine-tuning

```bash
cd finetune
python llama_4_train.py
```

The script will:
1. ✅ Authenticate with Hugging Face
2. ⚙️ Configure training parameters
3. 🦙 Load Llama 4 Scout with 4-bit quantization
4. 📊 Prepare the memory dataset (5000 examples)
5. 🔧 Apply LoRA adapters (~0.5% trainable parameters)
6. 🚀 Train for 3 epochs with evaluation
7. 🚀 Merge adapters and deploy to Hugging Face Hub

### Configuration

Key parameters in `Llama4TrainingConfig`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_ID` | `meta-llama/Llama-4-Scout-17B-16E` | Base model |
| `DATASET_PATH` | `./memory_dataset.jsonl` | Training data |
| `LORA_RANK` | `64` | LoRA adaptation rank |
| `BATCH_SIZE` | `4` | Per-device batch size |
| `NUM_EPOCHS` | `3` | Training epochs |
| `LEARNING_RATE` | `3e-4` | Learning rate for LoRA |

### Expected Outputs

```
finetune/
├── llama4_memory_finetuned/          # Training checkpoints
├── llama4_memory_finetuned_merged/   # Final merged model
└── llama_4_train.py                  # This script
```

**Hugging Face Hub**: Model uploaded to `your-username/llama4-scout-memory`

### Memory Requirements

- **4-bit Training**: ~12-16GB VRAM
- **Training Time**: ~2-4 hours (depending on GPU)
- **Dataset Size**: 5000 memory conversation examples

### Testing the Model

After training, test with memory conversation format:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-username/llama4-scout-memory")
model = AutoModelForCausalLM.from_pretrained("your-username/llama4-scout-memory")

prompt = "<|im_start|>user\nI love chocolate ice cream.<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Troubleshooting

**Out of Memory**:
- Reduce `BATCH_SIZE` to 2 or 1
- Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size

**Authentication Issues**:
- Verify HF_TOKEN is set: `echo $HF_TOKEN`
- Check model access permissions on Hugging Face

**Dataset Loading Issues**:
- Verify file exists: `ls -la memory_dataset.jsonl`
- Check file format: `head -1 memory_dataset.jsonl`

### Advanced Usage

To modify training parameters, edit the `Llama4TrainingConfig` class:

```python
# Example: Reduce memory usage
self.BATCH_SIZE = 2
self.GRADIENT_ACCUMULATION_STEPS = 16
self.LORA_RANK = 32

# Example: Longer training
self.NUM_EPOCHS = 5
self.LEARNING_RATE = 2e-4
```

This creates a specialized memory-aware Llama 4 Scout model optimized for conversation context retention!

NOTE: 
- Base Model: Loads the full precision Llama 4 Scout model initially
- Quantization: Immediately quantizes weights to 4-bit to save memory (~12GB vs ~34GB)
- Training: Uses BF16 precision for gradients and activations during training
- Final Model: After merging, outputs a full-precision model & then we'll quantize again to 4-bit for post benchmarks 