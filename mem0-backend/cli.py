#!/usr/bin/env python3
"""
Mem0 CLI - A minimal command-line interface for storing and recalling user memories
"""

import argparse
import sys
import json
import torch
from typing import Optional, List, Dict, Any
from mem0 import Memory
from wrap import load_model

class Mem0CLI:
    def __init__(self, model_choice: str = "llama3.1"):
        """Initialize the CLI with a memory instance and optional model"""
        self.memory = Memory()
        self.model = None
        self.tokenizer = None
        self.model_choice = model_choice
        
        # Load model if specified
        if model_choice != "none":
            try:
                self.model, self.tokenizer = load_model(model_choice)
                print(f"✅ Model loaded: {model_choice}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load model '{model_choice}': {e}")
                print("   CLI will work without AI generation capabilities")
    
    def add_memory(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """Add a memory to the system"""
        try:
            result = self.memory.add(message, user_id=user_id)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_memories(self, query: str, user_id: str = "default", limit: int = 5) -> Dict[str, Any]:
        """Search for memories"""
        try:
            results = self.memory.search(query, user_id=user_id)
            # Limit results if specified
            if isinstance(results, dict) and 'results' in results:
                results['results'] = results['results'][:limit]
            elif isinstance(results, list):
                results = results[:limit]
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def chat_with_memory(self, user_input: str, user_id: str = "default") -> Dict[str, Any]:
        """Chat with memory integration (requires model)"""
        if not self.model or not self.tokenizer:
            return {"success": False, "error": "No model loaded for chat functionality"}
        
        try:
            # Search for relevant memories
            search_result = self.search_memories(user_input, user_id=user_id, limit=3)
            if not search_result["success"]:
                return search_result
            
            relevant_memories = search_result["results"]
            if isinstance(relevant_memories, dict):
                relevant_memories = relevant_memories.get('results', [])
            
            # Create context from memories
            memory_context = ""
            if relevant_memories:
                memory_context = "\nRelevant memories:\n" + "\n".join([mem['memory'] for mem in relevant_memories])
            
            # Generate response using the model directly
            prompt = f"User: {user_input}{memory_context}\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the generated text
            prompt_length = len(prompt)
            response = generated_text[prompt_length:].strip()
            
            # Store the conversation
            conversation = f"User said: {user_input}. Assistant responded: {response}"
            add_result = self.add_memory(conversation, user_id=user_id)
            
            return {
                "success": True,
                "response": response,
                "memories_used": len(relevant_memories),
                "memory_added": add_result
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_users(self) -> Dict[str, Any]:
        """List all users with memories"""
        try:
            # This would require accessing internal memory storage
            # For now, return a placeholder
            return {"success": True, "message": "User listing not yet implemented"}
        except Exception as e:
            return {"success": False, "error": str(e)}

def format_output(data: Dict[str, Any], format_type: str = "text") -> str:
    """Format output based on specified format"""
    if format_type == "json":
        return json.dumps(data, indent=2)
    elif format_type == "text":
        if not data.get("success", True):
            return f"❌ Error: {data.get('error', 'Unknown error')}"
        
        if "response" in data:
            # Chat response
            result = f"🤖 Assistant: {data['response']}\n"
            if data.get("memories_used", 0) > 0:
                result += f"📚 Used {data['memories_used']} relevant memories\n"
            return result
        elif "results" in data:
            # Search results
            results = data["results"]
            if isinstance(results, dict) and "results" in results:
                results = results["results"]
            
            if not results:
                return "🔍 No memories found matching your query"
            
            result = f"🔍 Found {len(results)} memories:\n\n"
            for i, mem in enumerate(results, 1):
                result += f"{i}. {mem.get('memory', 'Unknown memory')}\n"
            return result
        elif "result" in data:
            # Add memory result
            return f"✅ Memory added successfully: {data['result']}"
        else:
            return str(data)
    
    return str(data)

def main():
    parser = argparse.ArgumentParser(
        description="Mem0 CLI - Store and recall user memories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a memory
  python cli.py add "I love playing basketball" --user alice
  
  # Search memories
  python cli.py search "basketball" --user alice
  
  # Chat with memory (requires model)
  python cli.py chat "What sports do I like?" --user alice --model llama3.1-finetuned
  
  # Add memory with custom format
  python cli.py add "I enjoy reading sci-fi" --user bob --format json

Available Models for Chat:
  llama3.1              - Llama 3.1 8B (4-bit)
  llama3.1-instruct-bf16 - Llama 3.1 8B Instruct (bf16)
  llama3.1-finetuned    - Llama 3.1 8B Memory Finetuned
  llama4-bf16           - Llama 4 Scout 17B (bf16)
  llama4-4bit           - Llama 4 Scout 17B (4-bit)
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add memory command
    add_parser = subparsers.add_parser("add", help="Add a memory")
    add_parser.add_argument("message", help="Memory message to store")
    add_parser.add_argument("--user", "-u", default="default", help="User ID (default: default)")
    add_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    
    # Search memories command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--user", "-u", default="default", help="User ID (default: default)")
    search_parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of results")
    search_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with memory integration")
    chat_parser.add_argument("message", help="User message")
    chat_parser.add_argument("--user", "-u", default="default", help="User ID (default: default)")
    chat_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    chat_parser.add_argument("--model", "-m", 
                           choices=["llama3.1", "llama3.1-instruct-bf16", "llama3.1-finetuned", "llama4-bf16", "llama4-4bit"],
                           required=True,
                           help="Model to use for chat functionality")
    
    # List users command
    list_parser = subparsers.add_parser("list-users", help="List all users")
    list_parser.add_argument("--format", "-f", choices=["text", "json"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI without model (models are loaded per command as needed)
    cli = Mem0CLI("none")
    
    # Execute command
    if args.command == "add":
        result = cli.add_memory(args.message, args.user)
        print(format_output(result, args.format))
    
    elif args.command == "search":
        result = cli.search_memories(args.query, args.user, args.limit)
        print(format_output(result, args.format))
    
    elif args.command == "chat":
        # Get model from chat parser (now required)
        model_choice = args.model
        
        # Initialize CLI with the specified model
        cli = Mem0CLI(model_choice)
        result = cli.chat_with_memory(args.message, args.user)
        print(format_output(result, args.format))
    
    elif args.command == "list-users":
        result = cli.list_users()
        print(format_output(result, args.format))

if __name__ == "__main__":
    main()
