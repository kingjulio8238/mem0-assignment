from mem0 import Memory
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Available models
MODELS = {
    "llama3.1": "unsloth/llama-3.1-8b-bnb-4bit",
    "llama4": "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct",
    "llama4-bf16": "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct",
    "llama4-4bit": "mlx-community/meta-llama-Llama-4-Scout-17B-16E-4bit",
    "llama4-gguf": "/home/ubuntu/mem0-assignment/model_cache/scout_gguf/Q4_K_M"
}

def load_model(model_choice):
    """Load the specified model and tokenizer"""
    model_path = MODELS[model_choice]
    
    # Determine model name and quantization settings
    if model_choice == "llama3.1":
        model_name = "Llama 3.1 8B (4-bit)"
        quantization = "4bit"
    elif model_choice == "llama4":
        model_name = "Llama 4 Scout 17B (default)"
        quantization = None
    elif model_choice == "llama4-bf16":
        model_name = "Llama 4 Scout 17B (bf16)"
        quantization = "bf16"
    elif model_choice == "llama4-4bit":
        model_name = "Llama 4 Scout 17B (4-bit)"
        quantization = "4bit"
    else:  # llama4-gguf
        model_name = "Llama 4 Scout 17B GGUF Q4_K_M"
        quantization = "gguf"
    
    print(f"Loading {model_name} model from: {model_path}")
    
    # Handle GGUF model differently
    if quantization == "gguf":
        try:
            from llama_cpp import Llama
            from pathlib import Path
            
            # Find GGUF files in directory
            gguf_files = list(Path(model_path).glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF files found in {model_path}")
            
            # Use the first file
            model_file = str(gguf_files[0])
            print(f"Loading GGUF file: {model_file}")
            
            model = Llama(
                model_path=model_file,
                n_ctx=2048,
                n_threads=8,
                verbose=False
            )
            tokenizer = None  # GGUF models handle tokenization internally
            print("GGUF model loaded successfully")
            
        except ImportError:
            print("llama-cpp-python not found, installing...")
            import subprocess
            subprocess.run(["pip", "install", "llama-cpp-python"], check=True)
            return load_model(model_choice)  # Retry after installation
    else:
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

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test Mem0 with different Llama models")
parser.add_argument("--model", choices=["llama3.1", "llama4", "llama4-bf16", "llama4-4bit", "llama4-gguf"], default="llama3.1",
                   help="Choose which model to use (default: llama3.1)")
args = parser.parse_args()

# Load the selected model
model, tokenizer = load_model(args.model)

# Initialize Mem0 memory instance with default configuration
# Using default config since it works perfectly for our needs
memory = Memory()
print("Memory initialized with default configuration!")

def generate_with_local_model(prompt, max_length=512):
    """Generate text using the local Llama model"""
    if args.model == "llama4-gguf":
        # Use GGUF model generation
        output = model(
            prompt,
            max_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        return output['choices'][0]['text'].strip()
    else:
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
    result = memory.add(message, user_id=user_id)
    return result

def search_memory(query, user_id=None):
    """Search memories in the Mem0 instance"""
    results = memory.search(query, user_id=user_id)
    return results

def chat_with_memory(user_input, user_id="alice"):
    """Chat with the local model and store/retrieve memories"""
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
    if args.model == "llama3.1":
        model_name = "Llama 3.1 8B (4-bit)"
    elif args.model == "llama4":
        model_name = "Llama 4 Scout 17B (default)"
    elif args.model == "llama4-bf16":
        model_name = "Llama 4 Scout 17B (bf16)"
    elif args.model == "llama4-4bit":
        model_name = "Llama 4 Scout 17B (4-bit)"
    else:  # llama4-gguf
        model_name = "Llama 4 Scout 17B GGUF Q4_K_M"
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
    Map benchmark model names to main.py model choices
    This function helps the unified_memory_benchmark.py script
    """
    mapping = {
        "llama3.1": "llama3.1",
        "llama4": "llama4",
        "llama4-bf16": "llama4-bf16", 
        "llama4-4bit": "llama4-4bit",
        "llama4-gguf": "llama4-gguf"
    }
    return mapping.get(benchmark_model_name, "llama3.1")  # Default fallback
