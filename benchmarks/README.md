# Benchmarks

**TLDR**: Test model performance (speed, memory usage) and compare base vs fine-tuned models. Generate charts and reports.

## Quick Start

```bash
# Test inference speed
python scripts/unified_inference_benchmark.py --model llama-3.1-8b-instruct-bf16 --num-prompts 100

# Test memory retrieval
python scripts/unified_memory_benchmark.py --model llama3.1-instruct-bf16

# Compare all models and generate reports
python benchmark_comparison_analysis.py \
  --base-results results/base_model_results_4bit \
  --base-bf16-results results/base_model_results_bf16 \
  --bf16-results results/finetuned_bf16_results \
  --q4km-results results/finetuned_q4km_results \
  --output-dir model_comparisons

# Generate visualizations
python generate_comparison_plots.py --comparison-results model_comparisons --output-dir model_comparisons
```

## What This Does

- **Inference Benchmarks**: Measure model speed (tokens/second) and latency
- **Memory Benchmarks**: Test how well models retrieve and use stored memories
- **Model Comparison**: Compare base models vs fine-tuned versions across different quantizations
- **Visualization**: Generate performance charts and analysis reports

## Available Models

**Inference**: `llama-3.1-8b-bnb-4bit`, `llama-3.1-8b-instruct-bf16`, `llama-3.1-bf16-gguf`, `llama-3.1-q4km-gguf`, `llama4-bf16`, `llama4-4bit`, `llama4-gguf`

**Memory**: `llama-3.1-8b-bnb-4bit`, `llama3.1-instruct-bf16`, `llama-3.1-bf16-gguf`, `llama-3.1-q4km-gguf`, `llama4-bf16`, `llama4-4bit`, `llama4-gguf`

## Output

Results saved as timestamped JSON files. Comparison generates:
- Performance analysis reports
- Interactive charts in `model_comparisons/visualizations/`
- Summary files for easy review
