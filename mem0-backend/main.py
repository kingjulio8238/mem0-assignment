from mem0 import Memory
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Available models
MODELS = {
    "llama3.1": "unsloth/llama-3.1-8b-bnb-4bit",
    "llama4": "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/92f3b1597a195b523d8d9e5700e57e4fbb8f20d3"
}

def load_model(model_choice):
    """Load the specified model and tokenizer"""
    model_path = MODELS[model_choice]
    model_name = "Llama 3.1 8B" if model_choice == "llama3.1" else "Llama 4 Scout 17B"
    
    print(f"Loading {model_name} model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate settings
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    
    # Use 4-bit quantization for Llama 3.1, regular loading for Llama 4
    if model_choice == "llama3.1":
        load_kwargs["load_in_4bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    print(f"Model loaded on device: {next(model.parameters()).device}")
    
    return model, tokenizer

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test Mem0 with different Llama models")
parser.add_argument("--model", choices=["llama3.1", "llama4"], default="llama3.1",
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
    model_name = "Llama 3.1 8B" if args.model == "llama3.1" else "Llama 4 Scout 17B"
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
