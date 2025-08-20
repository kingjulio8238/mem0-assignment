# Enhanced Llama 4 Scout Fine-tuning Guide

This enhanced training script provides comprehensive fine-tuning capabilities for Llama 4 Scout with support for:

- **FP16 vs 4-bit QLoRA comparison training**
- **Advanced hyperparameter tuning with VRAM monitoring**
- **Automatic GGUF export**
- **Memory-aware configuration generation**

## Features

### ✨ Key Enhancements

1. **Multiple Training Modes**:
   - `fp16`: Full precision FP16 training for maximum quality
   - `4bit`: Memory-efficient 4-bit QLoRA training
   - `comparison`: Automated comparison between both modes

2. **Smart Hyperparameter Tuning**:
   - VRAM-aware configuration generation
   - Mode-specific parameter ranges
   - Automatic best configuration selection

3. **Enhanced GGUF Export**:
   - Automatic LoRA adapter merging
   - Comprehensive model cards
   - Quantization instructions

4. **Advanced Memory Monitoring**:
   - Real-time VRAM tracking
   - Memory estimation for configurations
   - Automatic cleanup and optimization

## Quick Start

### 1. FP16 Training (High Quality)

```bash
python train_llama4_scout_accelerate.py \
    --model-path ./models/llama4-scout \
    --dataset-path ./finetune/memory_dataset.jsonl \
    --output-dir ./scout_fp16_model \
    --training-mode fp16 \
    --max-trials 3 \
    --export-gguf
```

### 2. 4-bit QLoRA Training (Memory Efficient)

```bash
python train_llama4_scout_accelerate.py \
    --model-path ./models/llama4-scout \
    --dataset-path ./finetune/memory_dataset.jsonl \
    --output-dir ./scout_4bit_model \
    --training-mode 4bit \
    --max-trials 3 \
    --export-gguf
```

### 3. Comparison Training (Best of Both)

```bash
python train_llama4_scout_accelerate.py \
    --model-path ./models/llama4-scout \
    --dataset-path ./finetune/memory_dataset.jsonl \
    --output-dir ./scout_comparison \
    --training-mode comparison \
    --max-trials 3 \
    --export-gguf
```

## Advanced Usage

### Custom Hyperparameters

```bash
python train_llama4_scout_accelerate.py \
    --model-path ./models/llama4-scout \
    --dataset-path ./finetune/memory_dataset.jsonl \
    --output-dir ./scout_custom \
    --training-mode fp16 \
    --lora-rank 32 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --num-epochs 2 \
    --skip-tuning \
    --export-gguf
```

### Skip Hyperparameter Tuning

```bash
python train_llama4_scout_accelerate.py \
    --training-mode fp16 \
    --skip-tuning \
    --lora-rank 16 \
    --batch-size 1 \
    --learning-rate 2e-4
```

## Output Structure

After training, you'll get organized outputs:

```
scout_model/
├── scout_trials/                    # Hyperparameter tuning results
│   ├── tuning_results_fp16.json    # FP16 tuning results
│   ├── tuning_results_4bit.json    # 4-bit tuning results
│   └── fp16_trial_01/               # Individual trial outputs
├── comparison/                      # Comparison mode results
│   └── fp16_vs_4bit_comparison.json
├── scout_model_fp16/               # FP16 model output
├── scout_model_4bit/               # 4-bit model output
├── scout_model_gguf_fp16/          # GGUF export (FP16)
└── scout_model_gguf_4bit/          # GGUF export (4-bit)
```

## Training Modes Explained

### FP16 Mode
- **Precision**: Full 16-bit floating point
- **Memory**: ~40-50GB VRAM required
- **Quality**: Highest model quality
- **Speed**: Faster inference than 4-bit
- **Use Case**: When you have high-end GPUs and want maximum quality

### 4-bit QLoRA Mode
- **Precision**: 4-bit quantized with LoRA adapters
- **Memory**: ~12-20GB VRAM required
- **Quality**: Slight quality reduction but very competitive
- **Speed**: Slower inference but much lower memory
- **Use Case**: When VRAM is limited or for experimentation

### Comparison Mode
- **Automatically trains both FP16 and 4-bit models**
- **Provides detailed performance comparison**
- **Helps you choose the best approach for your use case**

## Memory Requirements

| Training Mode | Min VRAM | Recommended VRAM | Notes |
|---------------|----------|------------------|-------|
| FP16          | 40GB     | 80GB             | A100/H100 recommended |
| 4-bit QLoRA   | 12GB     | 24GB             | RTX 4090/A6000 sufficient |
| Comparison    | 40GB     | 80GB             | Needs FP16 requirements |

## Hyperparameter Tuning

The script automatically generates VRAM-aware configurations:

### FP16 Tuning Space
- **LoRA Ranks**: [4, 8, 16] (conservative for memory)
- **Batch Sizes**: [1, 2] (smaller for FP16)
- **Learning Rates**: [1e-4, 2e-4, 5e-4]
- **Gradient Accumulation**: [32, 64, 128] (higher for stability)

### 4-bit Tuning Space
- **LoRA Ranks**: [4, 8, 16, 32] (more aggressive)
- **Batch Sizes**: [1, 2, 4] (can go higher)
- **Learning Rates**: [1e-4, 2e-4, 5e-4, 1e-3]
- **Gradient Accumulation**: [16, 32, 64]

## GGUF Export

Models are automatically exported to GGUF format for deployment:

```bash
# Convert to GGUF
python -m llama_cpp.convert ./scout_model_gguf_fp16/

# Quantize GGUF
./quantize ./scout_model_gguf_fp16/model.gguf ./scout_model_gguf_fp16/model-q4_k_m.gguf q4_k_m
```

## Monitoring and Debugging

### Real-time VRAM Monitoring
The script provides detailed memory tracking:
- GPU memory allocation
- Peak memory usage
- Available VRAM estimation
- Memory cleanup between trials

### Progress Indicators
- Animated loading indicators for long operations
- Real-time memory statistics
- Detailed training progress

### Error Handling
- Graceful failure recovery
- Memory cleanup on errors
- Detailed error reporting

## Performance Tips

### For FP16 Training
1. Use gradient checkpointing to save memory
2. Enable mixed precision (FP16/FP32)
3. Use larger gradient accumulation steps
4. Monitor memory usage closely

### For 4-bit Training
1. Use higher LoRA ranks if memory allows
2. Experiment with different quantization types
3. Consider nested quantization for memory savings
4. Use smaller learning rates

### General Tips
1. Start with hyperparameter tuning to find optimal settings
2. Use comparison mode to evaluate trade-offs
3. Monitor VRAM usage to avoid OOM errors
4. Export to GGUF for efficient deployment

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch size
--batch-size 1

# Increase gradient accumulation
--gradient-accumulation-steps 64

# Use 4-bit mode instead of FP16
--training-mode 4bit
```

### Slow Training
```bash
# Enable gradient checkpointing
# (automatically enabled)

# Use mixed precision
--training-mode fp16  # or 4bit for lower precision

# Increase batch size if memory allows
--batch-size 2
```

### Poor Convergence
```bash
# Try different learning rates
--learning-rate 1e-4

# Increase LoRA rank
--lora-rank 32

# Use more training epochs
--num-epochs 3
```

## Example Training Session

Here's what a complete comparison training session looks like:

```bash
$ python train_llama4_scout_accelerate.py \
    --model-path ./models/llama4-scout \
    --dataset-path ./finetune/memory_dataset.jsonl \
    --output-dir ./scout_comparison \
    --training-mode comparison \
    --max-trials 3 \
    --export-gguf

🦙 Llama 4 Scout Fine-tuning with Accelerate
Model: ./models/llama4-scout
Dataset: ./finetune/memory_dataset.jsonl
Output: ./scout_comparison
Training Mode: COMPARISON
LoRA Rank: 16
Batch Size: 1
Learning Rate: 0.0002
Skip Tuning: False

🔄 Running Comparison Training

==================================================
🎯 Training with FP16 precision
==================================================

🔍 Generating hyperparameter configurations for fp16 mode...
Available VRAM: 78.50GB
✅ Config 1: Rank=4, BS=1, LR=1.0e-04, GA=32, Est. Mem=45.0GB
✅ Config 2: Rank=4, BS=1, LR=2.0e-04, GA=64, Est. Mem=45.0GB
✅ Config 3: Rank=8, BS=1, LR=1.0e-04, GA=32, Est. Mem=46.0GB

🎯 Starting Hyperparameter Tuning Phase - FP16 Mode
📚 Loading dataset for hyperparameter tuning...
Dataset loaded: 5000 examples

🔄 Running trial: fp16_trial_01
🚀 Loading Llama 4 Scout model (FP16, 17B params)...
✅ Model loaded successfully
✅ Model prepared for FP16 training
✅ LoRA configuration applied successfully
🏋️ Starting training...
✅ Trial fp16_trial_01 completed: Final Loss: 0.4521

🏆 New best config found! Loss: 0.4521

[... continues with 4-bit training ...]

📊 COMPARISON SUMMARY: FP16 vs 4-bit QLoRA
==================================================

FP16 Results:
  Best Loss: 0.4521
  Best Config: {'lora_rank': 8, 'batch_size': 1, 'learning_rate': 2e-4}
  Output Dir: ./scout_comparison_fp16
  GGUF Path: ./scout_comparison_gguf_fp16

4BIT Results:
  Best Loss: 0.4687
  Best Config: {'lora_rank': 16, 'batch_size': 2, 'learning_rate': 1e-4}
  Output Dir: ./scout_comparison_4bit
  GGUF Path: ./scout_comparison_gguf_4bit

🏆 Performance Comparison:
  FP16 outperforms 4-bit by 3.54% (lower loss)
  FP16 Loss: 0.4521
  4-bit Loss: 0.4687

🎉 Comparison training completed!
```

This enhanced script provides everything you need for comprehensive Llama 4 Scout fine-tuning with proper benchmarking capabilities!
