# Advanced Memory Fine-tuning with Hyperparameter Tuning

This directory contains an advanced fine-tuning pipeline for memory-centric LLMs using Unsloth with automated hyperparameter tuning and VRAM monitoring.

## Files Overview

- `memory_dataset.jsonl` - 5,000 memory-focused training examples in ChatML format
- `create_memory_dataset.py` - Dataset generation script
- `train_with_hyperparameter_tuning.py` - **Complete training pipeline with hyperparameter tuning**
- `README_training.md` - This documentation

## Features

### 🔍 Hyperparameter Tuning
- **LoRA Rank**: Automatically tests ranks 8, 16, 32, 64
- **Batch Size**: Optimizes per-device batch sizes 1, 2, 4
- **Learning Rate**: Tests rates from 1e-4 to 1e-3
- **Sequence Length**: Adapts to 512, 1024, 2048 tokens
- **Memory-Aware**: Only tests feasible configurations for available VRAM

### 📦 Model Export Options
- **GGUF Format**: Exports to GGUF format for llama.cpp and compatible tools
- **vLLM Format**: Exports merged model for vLLM deployment
- **Low-Latency**: Optimized formats for production deployment

### 📊 VRAM Monitoring
- Real-time GPU memory tracking
- Peak memory usage recording
- Automatic cache clearing between trials
- Memory estimation for configuration feasibility
- System RAM monitoring

### 🎯 Training Optimization
- QLoRA with 4-bit quantization for memory efficiency
- Gradient checkpointing with Unsloth optimization
- 8-bit AdamW optimizer
- Dynamic memory management
- Comprehensive error handling

## Quick Start

### 1. Run Hyperparameter Tuning + Training

```bash
cd /home/ubuntu/mem0-assignment/finetune

# Basic training with default settings
python train_with_hyperparameter_tuning.py

# Training with GGUF export for llama.cpp deployment
python train_with_hyperparameter_tuning.py --export-formats gguf

# Training with vLLM export for high-throughput serving
python train_with_hyperparameter_tuning.py --export-formats vllm

# Training with both export formats
python train_with_hyperparameter_tuning.py --export-formats gguf vllm

# Skip hyperparameter tuning and export directly
python train_with_hyperparameter_tuning.py --skip-tuning --export-formats gguf --num-epochs 3
```

This will:
1. **Phase 1**: Test 3 different hyperparameter configurations
2. **Phase 2**: Train final model with best configuration
3. Save results and trained model

### 2. Custom Configuration

```python
from train_with_hyperparameter_tuning import AdvancedMemoryTrainer

# Initialize trainer
trainer = AdvancedMemoryTrainer()

# Run custom hyperparameter search
results = trainer.run_hyperparameter_tuning(
    dataset_path="memory_dataset.jsonl",
    max_trials=5  # Test more configurations
)

# Train final model
trainer.train_final_model(
    dataset_path="memory_dataset.jsonl", 
    num_epochs=3,
    save_path="./my_memory_model"
)
```

## Configuration Details

### Default Hyperparameter Search Space

```python
search_space = {
    "lora_rank": [8, 16, 32, 64],
    "lora_alpha": [8, 16, 32, 64], 
    "batch_size": [1, 2, 4],
    "gradient_accumulation": [2, 4, 8],
    "learning_rate": [1e-4, 2e-4, 5e-4, 1e-3],
    "max_seq_length": [512, 1024, 2048]
}
```

### Memory-Optimized Settings

Based on [Unsloth documentation](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide):

- **4-bit Quantization**: `load_in_4bit=True` reduces memory 4x
- **QLoRA**: Combines LoRA with quantization for efficiency
- **Gradient Checkpointing**: `"unsloth"` mode for memory savings
- **8-bit Optimizer**: `adamw_8bit` reduces optimizer memory

### Training Arguments

```python
TrainingArguments(
    per_device_train_batch_size=2,      # Tuned automatically
    gradient_accumulation_steps=4,       # Tuned automatically  
    learning_rate=2e-4,                 # Tuned automatically
    optim="adamw_8bit",                 # Memory efficient
    fp16=True,                          # Mixed precision
    max_steps=60,                       # Quick tuning trials
    num_train_epochs=3,                 # Final training
)
```

## Expected Outputs

### Hyperparameter Tuning Results
```
📊 HYPERPARAMETER TUNING RESULTS
=================================================================

🏆 Best Configuration:
  Loss: 0.4521
  Config: {'lora_rank': 16, 'batch_size': 2, 'learning_rate': 2e-4}
  Memory: 4.2GB
  Time: 125.3s

📈 All Results (sorted by loss):
  1. Loss: 0.4521 | Rank: 16 | Batch: 2 | Memory: 4.2GB
  2. Loss: 0.4687 | Rank: 8  | Batch: 1 | Memory: 3.8GB  
  3. Loss: 0.4832 | Rank: 32 | Batch: 1 | Memory: 4.9GB
```

### VRAM Monitoring
```
=== Memory Usage After Training ===
GPU: 4.20GB / 8.00GB (52.5%)
System RAM: 12.8GB / 32.0GB (40.0%)
```

## Key Advantages

✅ **Automated Hyperparameter Search** - No manual parameter tuning needed
✅ **Real-time VRAM Monitoring** - Prevents out-of-memory errors  
✅ **Advanced Memory Management** - Optimized for limited GPU resources
✅ **Automated Feasibility Testing** - Only tries configurations that will work
✅ **End-to-end Pipeline** - Dataset loading through model saving
✅ **Comprehensive Error Handling** - Robust training with automatic recovery
✅ **Detailed Results Analysis** - Compare configurations and metrics

## Memory Requirements

### Minimum Requirements
- **GPU**: 6GB VRAM (for basic configurations)
- **RAM**: 16GB system memory
- **Storage**: 10GB free space

### Recommended Requirements  
- **GPU**: 8GB+ VRAM (for optimal configurations)
- **RAM**: 32GB system memory
- **Storage**: 20GB free space

## Troubleshooting

### Out of Memory Errors
1. The script automatically reduces batch size and sequence length
2. Manually set more conservative parameters:
   ```python
   trainer = AdvancedMemoryTrainer(max_seq_length=512)
   ```

### Slow Training
1. Increase batch size if VRAM allows
2. Reduce gradient accumulation steps
3. Use shorter sequences for faster iterations

### Configuration Failures
1. Check VRAM availability with `nvidia-smi`
2. Reduce `max_trials` parameter
3. Manually specify conservative config

## Usage Examples

### Quick Start - Full Pipeline
```python
from train_with_hyperparameter_tuning import AdvancedMemoryTrainer

# Run complete pipeline
trainer = AdvancedMemoryTrainer()
results = trainer.run_hyperparameter_tuning("memory_dataset.jsonl", max_trials=3)
trainer.train_final_model("memory_dataset.jsonl", num_epochs=3)
```

### Custom Configuration
```python
# Use specific configuration without tuning
config = {"lora_rank": 16, "batch_size": 2, "learning_rate": 2e-4, 
          "gradient_accumulation": 4, "max_seq_length": 2048}
trainer.train_final_model("memory_dataset.jsonl", config=config)
```

### Export for Deployment
```python
# Train and export in GGUF format for llama.cpp
trainer.train_final_model("memory_dataset.jsonl", 
                         export_formats=["gguf"])

# Train and export in vLLM format for high-throughput serving
trainer.train_final_model("memory_dataset.jsonl", 
                         export_formats=["vllm"])

# Export both formats
trainer.train_final_model("memory_dataset.jsonl", 
                         export_formats=["gguf", "vllm"])
```

## Export Formats

### GGUF Format
- **Use Case**: Local inference with llama.cpp, Ollama, or similar tools
- **Benefits**: Highly optimized for CPU inference, memory efficient
- **Output**: `{model_path}_gguf/` directory with quantized model
- **Quantization**: Uses `q4_k_m` (4-bit) for optimal size/quality balance

### vLLM Format
- **Use Case**: High-throughput serving, batch inference, production APIs
- **Benefits**: Optimized for GPU serving with PagedAttention
- **Output**: `{model_path}_vllm/` directory with merged HuggingFace model
- **Compatibility**: Works with vLLM server, TGI, and other serving frameworks

## Next Steps

1. **Run Training**: Execute the hyperparameter tuning script
2. **Evaluate Results**: Test the trained model on memory tasks  
3. **Export for Deployment**: Use `--export-formats gguf vllm` for production-ready models
4. **Deploy Model**: Use GGUF for local inference or vLLM for serving APIs

For more details, see the [Unsloth Fine-tuning Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide).
