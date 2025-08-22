from mem0 import Memory
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Available models
MODELS = {
    "llama3.1": "unsloth/llama-3.1-8b-bnb-4bit",
    "llama3.1-instruct-bf16": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-finetuned": "kingJulio/llama-3.1-8b-memory-finetune",
    "llama4-bf16": "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct",
    "llama4-4bit": "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit"
}

def load_model(model_choice):
    """Load the specified model and tokenizer"""
    model_path = MODELS[model_choice]
    
    # Determine model name and quantization settings
    if model_choice == "llama3.1":
        model_name = "Llama 3.1 8B (4-bit)"
        quantization = "4bit"
    elif model_choice == "llama3.1-instruct-bf16":
        model_name = "Llama 3.1 8B Instruct (bf16)"
        quantization = "bf16"
    elif model_choice == "llama3.1-finetuned":
        model_name = "Llama 3.1 8B Memory Finetuned"
        quantization = "4bit"
    elif model_choice == "llama4-bf16":
        model_name = "Llama 4 Scout 17B (bf16)"
        quantization = "bf16"
    elif model_choice == "llama4-4bit":
        model_name = "Llama 4 Scout 17B (4-bit)"
        quantization = "4bit"
    
    print(f"Loading {model_name} model from: {model_path}")
    
    # Handle HuggingFace cache directory structure for Llama 4 Scout
    if "models--meta-llama--Llama-4-Scout" in model_path and os.path.isdir(model_path):
        # Look for the actual model files in snapshots directory
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_dir):
            # Find the latest snapshot
            snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshot_dirs:
                # Use the first (and likely only) snapshot
                model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                print(f"🔍 Using snapshot: {model_path}")
    
    # Transformers model loading
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure model loading based on quantization
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True
    }
    
    # Set quantization and torch dtype
    if quantization == "4bit":
        print("🔧 Using 4-bit quantization...")
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif quantization == "bf16":
        print("🔧 Using bf16 precision...")
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        # Default to bfloat16 for no quantization
        load_kwargs["torch_dtype"] = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    print(f"Model loaded on device: {next(model.parameters()).device}")

    return model, tokenizer

# Global variables for when wrap.py is used as a module
model = None
tokenizer = None
memory = None

def initialize_wrap_module(model_choice="llama3.1"):
    """Initialize the wrap module with a specific model"""
    global model, tokenizer, memory
    model, tokenizer = load_model(model_choice)
    memory = Memory()
    print("Memory initialized with default configuration!")
    return model, tokenizer, memory

def generate_with_local_model(prompt, max_length=512):
    """Generate text using the local Llama model"""
    global model, tokenizer
    if model is None or tokenizer is None:
        raise ValueError("Model not initialized. Call initialize_wrap_module() first or use wrap.py directly.")
    
    # Use transformers model generation
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the generated text
    prompt_length = len(prompt)
    return generated_text[prompt_length:].strip()

def add_memory(message, user_id=None):
    """Add a memory to the Mem0 instance"""
    global memory
    if memory is None:
        raise ValueError("Memory not initialized. Call initialize_wrap_module() first or use wrap.py directly.")
    result = memory.add(message, user_id=user_id)
    return result

def search_memory(query, user_id=None):
    """Search memories in the Mem0 instance"""
    global memory
    if memory is None:
        raise ValueError("Memory not initialized. Call initialize_wrap_module() first or use wrap.py directly.")
    results = memory.search(query, user_id=user_id)
    return results

def chat_with_memory(user_input, user_id="alice"):
    """Chat with the local model and store/retrieve memories"""
    global model, tokenizer, memory
    if model is None or tokenizer is None or memory is None:
        raise ValueError("Model or memory not initialized. Call initialize_wrap_module() first or use wrap.py directly.")
    
    # First search for relevant memories
    print(f"Searching for relevant memories...")
    search_result = search_memory(user_input, user_id=user_id)
    relevant_memories = search_result.get('results', []) if isinstance(search_result, dict) else search_result
    print(f"Found {len(relevant_memories)} relevant memories")
    
    # Create context from memories
    memory_context = ""
    if relevant_memories:
        memory_context = "\nRelevant memories:\n" + "\n".join([mem['memory'] for mem in relevant_memories[:3]])
    
    # Generate response with local model
    prompt = f"User: {user_input}{memory_context}\nAssistant:"
    response = generate_with_local_model(prompt, max_length=256)
    
    # Store the conversation in memory
    conversation = f"User said: {user_input}. Assistant responded: {response}"
    add_result = add_memory(conversation, user_id=user_id)
    print(f"Added to memory: {add_result}")
    
    return response

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Mem0 with different Llama models")
    parser.add_argument("--model", choices=["llama3.1", "llama3.1-instruct-bf16", "llama3.1-finetuned", "llama4-bf16", "llama4-4bit"], default="llama3.1",
                       help="Choose which model to use (default: llama3.1)")
    args = parser.parse_args()
    
    # Initialize the module
    initialize_wrap_module(args.model)
    
    if args.model == "llama3.1":
        model_name = "Llama 3.1 8B (4-bit)"
    elif args.model == "llama3.1-instruct-bf16":
        model_name = "Llama 3.1 8B Instruct (bf16)"
    elif args.model == "llama3.1-finetuned":
        model_name = "Llama 3.1 8B Memory Finetuned"
    elif args.model == "llama4-bf16":
        model_name = "Llama 4 Scout 17B (bf16)"
    elif args.model == "llama4-4bit":
        model_name = "Llama 4 Scout 17B (4-bit)"
    print("="*50)
    print(f"Testing Mem0 with Local {model_name} Model")
    print("="*50)
    
    # Test 1: Basic memory add
    print("\n1. Testing basic memory.add...")
    add_result = add_memory("I love playing basketball and watching NBA games", user_id="alice")
    print(f"Add result: {add_result}")
    
    # Test 2: Basic memory search
    print("\n2. Testing basic memory.search...")
    search_results = search_memory("sports", user_id="alice")
    print(f"Search results: {search_results}")
    
    # Test 3: Chat with memory integration
    print("\n3. Testing chat with memory integration...")
    user_inputs = [
        "What sports do I like?",
        "Tell me about my hobbies",
        "I also enjoy reading science fiction books"
    ]
    
    for user_input in user_inputs:
        print(f"\n--- User: {user_input} ---")
        response = chat_with_memory(user_input, user_id="alice")
        print(f"Assistant: {response}")
    
    # Test 4: Final memory search to see what was learned
    print("\n4. Final memory search - what does the system know about Alice?")
    final_search_result = search_memory("What do you know about Alice?", user_id="alice")
    final_search = final_search_result.get('results', []) if isinstance(final_search_result, dict) else final_search_result
    print(f"Final memories about Alice ({len(final_search)} found):")
    for i, mem in enumerate(final_search[:5], 1):
        print(f"  {i}. {mem['memory']}")
    
    print("\n" + "="*50)
    print("Testing completed!")
    print("="*50)

def get_model_choice_for_benchmark(benchmark_model_name):
    """
    Map benchmark model names to wrap.py model choices
    This function helps the unified_memory_benchmark.py script
    """
    mapping = {
        "llama3.1": "llama3.1",
        "llama3.1-instruct-bf16": "llama3.1-instruct-bf16",
        "llama3.1-finetuned": "llama3.1-finetuned",
        "llama4-bf16": "llama4-bf16", 
        "llama4-4bit": "llama4-4bit"
    }
    return mapping.get(benchmark_model_name, "llama3.1")  # Default fallback
