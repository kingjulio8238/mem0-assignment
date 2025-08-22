# Mem0 Backend

**TLDR**: Store and retrieve user memories with AI-powered chat. Supports multiple LLM models and CLI interface.

## Quick Start

```bash
# Use the main CLI (from project root)
./mem0 memory add "I love playing basketball" --user alice
./mem0 memory search "sports" --user alice
./mem0 memory chat "What do I like?" --user alice --model llama3.1

# Or run directly from this directory
python cli.py add "I love soccer" --user john
python cli.py search "soccer" --user john --limit 3
python cli.py chat "Tell me about my interests" --user john --model llama3.1-instruct-bf16
```

## What This Does

- **Memory Storage**: Store user memories with timestamps and IDs
- **Memory Search**: Find relevant memories using keyword matching
- **AI Chat**: Generate responses using retrieved memories as context
- **Multi-User**: Support multiple users with separate memory spaces
- **Model Integration**: Works with various LLM models (Llama 3.1, Llama 4 Scout)

## File Structure & Architecture

```
mem0-backend/
├── mem0.py          # Core memory storage engine (pure logic, no AI)
├── wrap.py          # AI model integration + memory chat
├── cli.py           # Command-line interface
└── memories.json    # Memory storage file
```

### Component Relationships

1. **`mem0.py`** - Pure memory system
   - Handles storage, retrieval, search
   - No AI dependencies
   - Used by both `cli.py` and `wrap.py`

2. **`wrap.py`** - AI integration layer
   - Imports `mem0.py` for memory operations
   - Loads LLM models (Llama 3.1, Llama 4 Scout)
   - Provides chat with memory context
   - Used by `cli.py` for AI features

3. **`cli.py`** - Command interface
   - Uses `mem0.py` for basic memory operations
   - Uses `wrap.py` for AI chat functionality
   - Provides user-friendly CLI commands

## Available Models

- `llama3.1` - Llama 3.1 8B (4-bit quantized)
- `llama3.1-instruct-bf16` - Llama 3.1 8B Instruct (bf16)
- `llama3.1-finetuned` - Fine-tuned memory model
- `llama4-bf16` - Llama 4 Scout 17B (bf16)
- `llama4-4bit` - Llama 4 Scout 17B (4-bit quantized)

## Output

- Memories stored in `memories.json` with user separation
- Search results with relevance ranking
- AI-generated responses using memory context
- JSON or text formatted output
