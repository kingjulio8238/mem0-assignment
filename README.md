# Mem0 Assignment

**Project TLDR**: 
Memory-focused pipeline - wrap models with Mem0, benchmark baselines (inference and memory), fine-tune models,benchmark fine-tuned model performance, run analyses and integrate with memory storage/retrieval via CLI.

## Directory Structure

```
mem0-assignment/
├── README.md                    # Project documentation
├── mem0                         # Main CLI executable
├── install.sh                   # Installation script
├── setup.py                     # Package setup configuration
├── requirements.txt             # Python dependencies
├── memories.json                # Memory storage file
├── .gitignore                   # Git ignore rules
├── venv/                        # Virtual environment
├── .git/                        # Git repository
│
├── finetune/                    # Model fine-tuning module
│   ├── train.py                 # Main training script
│   ├── create_memory_dataset.py # Dataset creation
│   ├── memory_dataset.jsonl     # Training dataset
│   ├── export_to_gguf.py        # GGUF export functionality
│   ├── upload_to_hf.py          # HuggingFace upload
│   ├── README.md                # Fine-tuning documentation
│   ├── training_details/        # Training logs and configs
│   └── unsloth_compiled_cache/  # Unsloth compilation cache
│
├── benchmarks/                  # Performance benchmarking
│   ├── scripts/
│   │   ├── unified_memory_benchmark.py    # Memory capability tests
│   │   └── unified_inference_benchmark.py # Speed and efficiency tests
│   ├── results/                 # Benchmark results storage
│   │   ├── base_model_results_4bit/
│   │   ├── base_model_results_bf16/
│   │   ├── finetuned_bf16_results/
│   │   └── finetuned_q4km_results/
│   ├── model_comparisons/       # Comparison analysis
│   ├── generate_comparison_plots.py       # Visualization scripts
│   ├── benchmark_comparison_analysis.py   # Analysis tools
│   ├── test_prompts.txt         # Benchmark test prompts
│   ├── synthetic_memories.txt   # Synthetic memory data
│   └── README.md                # Benchmark documentation
│
└── mem0-backend/                # Memory storage and retrieval
    ├── mem0.py                  # Core memory operations
    ├── cli.py                   # CLI interface
    ├── wrap.py                  # Model wrapper utilities
    └── README.md                # Backend documentation
```

## CLI Installation

```bash
# Install the CLI for system-wide access (optional)
./install.sh

# Or run directly from project directory
./mem0 [command]
```

## Quick Start

```bash
# Enter interactive CLI mode
./mem0

# Or run commands directly
mem0 memory add "I love continual learning" --user julian
mem0 memory search "continual learning" --user julian
mem0 memory chat "What do I like?" --user julian --model llama3.1-finetuned
mem0 benchmark inference --model llama3.1 --num-prompts 100
mem0 benchmark memory --model llama3.1 
mem0 train --max-trials 3 --num-epochs 2 --export-formats gguf --hf-repo-name "kingJulio/memory-model"
```

## Interactive CLI Usage

```bash
./mem0
🤖 Welcome to Mem0 CLI!
Type 'help' for available commands, 'exit' to quit.

mem0> memory add "I love continual learning" --user julian
mem0> memory search "learning" --user julian
mem0> benchmark memory --model llama3.1-instruct-bf16
mem0> train --max-trials 3 --num-epochs 2
mem0> exit
```

## What This Does

### 🎯 Fine-tuning (`finetune/`)
- Train memory-focused LLMs with automated hyperparameter tuning
- Export to GGUF (local) or vLLM (serving) formats
- Real-time VRAM monitoring and optimization

### 📊 Benchmarks (`benchmarks/`)
- Test model speed (tokens/second) and memory capabilities
- Compare base vs fine-tuned models across quantizations
- Generate performance charts and analysis reports

### 💾 Memory Backend (`mem0-backend/`)
- Store and retrieve user memories with timestamps
- AI-powered chat using retrieved memories as context
- Multi-user support with separate memory spaces

## Complete Workflow

```bash
# Requirements
export HF_TOKEN 
export OPENAI_API_KEY 

# 1. Test memory system
mem0 memory add "User preference: I enjoy coding" --user developer
mem0 memory chat "What should I work on?" --user developer --model llama3.1

# 2. Benchmark performance
mem0 benchmark inference --model llama3.1 --num-prompts 100
mem0 benchmark memory --model llama3.1-instruct-bf16

# 3. Train your model
mem0 train --export-formats gguf vllm --hf-repo-name "your-username/memory-model"

# 4. Compare results
mem0 benchmark compare --base-results results/base_model_results_4bit --bf16-results results/finetuned_bf16_results
```

## Available Models

**Fine-tuning**: Llama 3.1 8B, Llama 4 Scout 17B  
**Benchmarks**: 4-bit, bf16, finetuned GGUF variants 
**Backend**: llama3.1, llama3.1-instruct-bf16, llama3.1-finetuned, llama4-bf16, llama4-4bit

## Output / Results

- **Fine-tuned models** in HuggingFace, GGUF, and vLLM formats
- **Performance reports** with charts and analysis
- **Memory system** with AI chat capabilities
- **Benchmark comparisons** across model variants

## Fine-tuned Models

*Had issues with loading the scout model  

### 🤖 Llama 3.1 8B Memory-Finetuned

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/kingJulio/llama-3.1-8b-memory-finetune)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![Model Size](https://img.shields.io/badge/Size-8B%20Parameters-orange)](https://huggingface.co/kingJulio/llama-3.1-8b-memory-finetune)

**Model**: [kingJulio/llama-3.1-8b-memory-finetune](https://huggingface.co/kingJulio/llama-3.1-8b-memory-finetune)

**Base Model**: Meta-Llama-3.1-8B-Instruct

**Fine-tuning**: Memory-focused instruction tuning on synthetic memory datasets

**Use Cases**:
- Memory-augmented conversations
- Contextual memory retrieval
- Personalized AI interactions
- Long-term memory management

**Quick Start**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "kingJulio/llama-3.1-8b-memory-finetune"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Memory-enhanced conversation
prompt = "User memory: I love coding in Python\nUser: What should I work on today?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Performance**: Optimized for memory retrieval accuracy and contextual understanding 