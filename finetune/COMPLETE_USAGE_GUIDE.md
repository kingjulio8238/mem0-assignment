# Complete Llama 4 Scout Fine-tuning and Benchmarking Guide

This guide demonstrates how to fine-tune Llama 4 Scout with FP16 and 4-bit precision, then benchmark the results using the updated benchmark scripts.

## 🚀 Quick Start: Complete Workflow

### 1. FP16 vs 4-bit Comparison Training

Run the complete comparison to get both models:

```bash
cd finetune

# Run comparison training (trains both FP16 and 4-bit models)
python train_llama4_scout_accelerate.py \
    --model-path ./models/llama4-scout \
    --dataset-path ./memory_dataset.jsonl \
    --output-dir ./scout_comparison \
    --training-mode comparison \
    --max-trials 3 \
    --export-gguf
```

This will create:
- `./scout_comparison_fp16/` - FP16 fine-tuned model
- `./scout_comparison_4bit/` - 4-bit fine-tuned model
- `./scout_comparison_gguf_fp16/` - GGUF export of FP16 model
- `./scout_comparison_gguf_4bit/` - GGUF export of 4-bit model
- `./scout_comparison/comparison/fp16_vs_4bit_comparison.json` - Detailed comparison results

### 2. Individual Training Modes

#### FP16 Training (High Quality)

```bash
# FP16 training with hyperparameter tuning
python train_llama4_scout_accelerate.py \
    --model-path ./models/llama4-scout \
    --dataset-path ./memory_dataset.jsonl \
    --output-dir ./scout_fp16_final \
    --training-mode fp16 \
    --max-trials 5 \
    --export-gguf
```

#### 4-bit QLoRA Training (Memory Efficient)

```bash
# 4-bit training with hyperparameter tuning
python train_llama4_scout_accelerate.py \
    --model-path ./models/llama4-scout \
    --dataset-path ./memory_dataset.jsonl \
    --output-dir ./scout_4bit_final \
    --training-mode 4bit \
    --max-trials 5 \
    --export-gguf
```

## 📊 Benchmarking Fine-tuned Models

### 3. Inference Benchmarking

#### Benchmark Base Llama 4 Scout (BF16)

```bash
cd ../benchmarks

# Benchmark the base Llama 4 Scout model in BF16 precision
python unified_inference_benchmark.py \
    --model llama-4-scout-bf16 \
    --num-prompts 50
```

#### Benchmark Your Fine-tuned Models

```bash
# Benchmark FP16 fine-tuned model
python unified_inference_benchmark.py \
    --model-path ../finetune/scout_comparison_fp16 \
    --model-type local \
    --quantization bf16 \
    --num-prompts 50 \
    --output-dir ./finetuned_fp16_results

# Benchmark 4-bit fine-tuned model
python unified_inference_benchmark.py \
    --model-path ../finetune/scout_comparison_4bit \
    --model-type local \
    --quantization 4bit \
    --num-prompts 50 \
    --output-dir ./finetuned_4bit_results

# Benchmark GGUF models
python unified_inference_benchmark.py \
    --model-path ../finetune/scout_comparison_gguf_fp16 \
    --model-type gguf \
    --quantization bf16 \
    --num-prompts 50 \
    --output-dir ./finetuned_gguf_fp16_results
```

### 4. Memory Retrieval Benchmarking

#### Benchmark Base Model Memory Performance

```bash
# Benchmark base Llama 4 Scout memory retrieval
python unified_memory_benchmark.py \
    --model llama-4-scout-bf16 \
    --cleanup
```

#### Benchmark Fine-tuned Model Memory Performance

```bash
# Test memory performance with your fine-tuned models
python unified_memory_benchmark.py \
    --model-path ../finetune/scout_comparison_fp16 \
    --model-type local \
    --output-dir ./finetuned_fp16_memory_results \
    --cleanup

python unified_memory_benchmark.py \
    --model-path ../finetune/scout_comparison_4bit \
    --model-type local \
    --output-dir ./finetuned_4bit_memory_results \
    --cleanup
```

## 🎯 Expected Results Structure

After running the complete workflow, you'll have:

```
finetune/
├── scout_comparison/
│   ├── scout_comparison_fp16/          # FP16 fine-tuned model
│   ├── scout_comparison_4bit/          # 4-bit fine-tuned model
│   ├── scout_comparison_gguf_fp16/     # GGUF FP16 model
│   ├── scout_comparison_gguf_4bit/     # GGUF 4-bit model
│   ├── scout_trials/                   # Hyperparameter tuning results
│   │   ├── tuning_results_fp16.json
│   │   └── tuning_results_4bit.json
│   └── comparison/
│       └── fp16_vs_4bit_comparison.json

benchmarks/
├── scout/
│   └── base_model_results/             # Base model benchmarks
│       ├── inference_benchmark_*.json
│       └── memory_benchmark_*.json
├── finetuned_fp16_results/             # FP16 fine-tuned benchmarks
├── finetuned_4bit_results/             # 4-bit fine-tuned benchmarks
└── finetuned_gguf_fp16_results/        # GGUF benchmarks
```

## ⚙️ Advanced Configuration

### Custom Hyperparameter Training

```bash
# Train with specific hyperparameters (skip tuning)
python train_llama4_scout_accelerate.py \
    --model-path ./models/llama4-scout \
    --dataset-path ./memory_dataset.jsonl \
    --output-dir ./scout_custom \
    --training-mode fp16 \
    --lora-rank 32 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --num-epochs 2 \
    --skip-tuning \
    --export-gguf
```

### Memory-Optimized Training

```bash
# For systems with limited VRAM
python train_llama4_scout_accelerate.py \
    --training-mode 4bit \
    --lora-rank 8 \
    --batch-size 1 \
    --gradient-accumulation-steps 64 \
    --max-trials 3
```

## 📈 Performance Analysis

### Key Metrics to Compare

1. **Training Metrics**:
   - Final training loss
   - Peak memory usage
   - Training time
   - Hyperparameter effectiveness

2. **Inference Metrics**:
   - Average latency per token
   - Throughput (tokens/second)
   - Memory efficiency
   - Response quality

3. **Memory Retrieval Metrics**:
   - Precision@5 for memory queries
   - Retrieval time
   - Query success rate

### Expected Performance Characteristics

| Metric | FP16 | 4-bit QLoRA |
|--------|------|-------------|
| **Training Time** | Slower | Faster |
| **VRAM Usage** | ~40-50GB | ~12-20GB |
| **Model Quality** | Highest | Very Good |
| **Inference Speed** | Fast | Moderate |
| **Deployment Size** | Large | Compact |

## 🔧 Troubleshooting

### Out of Memory Errors

```bash
# Reduce batch size and increase gradient accumulation
--batch-size 1 --gradient-accumulation-steps 128

# Use 4-bit training instead of FP16
--training-mode 4bit

# Reduce LoRA rank
--lora-rank 8
```

### Poor Training Performance

```bash
# Increase LoRA rank for better capacity
--lora-rank 32

# Try different learning rates
--learning-rate 1e-4

# Use more training epochs
--num-epochs 3
```

### Benchmark Issues

```bash
# For memory benchmark issues, ensure mem0 backend is running
cd ../mem0-backend
python main.py

# For inference benchmark issues, check model paths
python unified_inference_benchmark.py --model-path /path/to/model --model-type local
```

## 🎉 Next Steps

1. **Compare Results**: Use the generated JSON files to analyze performance differences
2. **Deploy Models**: Use the GGUF exports for efficient deployment
3. **Iterate**: Based on benchmark results, adjust hyperparameters for better performance
4. **Scale Up**: Apply the same process to larger datasets or different model sizes

This complete workflow gives you production-ready fine-tuned Llama 4 Scout models with comprehensive performance benchmarks!
