from mem0 import Memory
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup for local Llama model without Ollama
model_path = "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--unsloth--llama-3.1-8b-bnb-4bit/snapshots/b80adf5d249b569469d0a19192ff36e88f133413"

# Load the model and tokenizer
print("Loading Llama 3.1 8B model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    load_in_4bit=True
)
print(f"Model loaded on device: {next(model.parameters()).device}")

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
    print("="*50)
    print("Testing Mem0 with Local Llama 3.1 8B Model")
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
