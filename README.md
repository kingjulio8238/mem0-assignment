# Mem0 Assignment

**TLDR**: Complete memory-focused LLM pipeline - fine-tune models, benchmark performance, and deploy with memory storage/retrieval.

## Installation

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
mem0 memory add "I love playing basketball" --user alice
mem0 benchmark inference --model llama3.1 --num-prompts 100
mem0 train --max-trials 3 --num-epochs 2 --export-formats gguf
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
- Test model speed (tokens/second) and memory usage
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