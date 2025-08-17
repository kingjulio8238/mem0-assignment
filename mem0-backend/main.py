from mem0 import Memory

# Initialize Mem0 memory instance
memory = Memory()

def add_memory(message, user_id=None):
    """Add a memory to the Mem0 instance"""
    result = memory.add(message, user_id=user_id)
    return result

def search_memory(query, user_id=None):
    """Search memories in the Mem0 instance"""
    results = memory.search(query, user_id=user_id)
    return results

if __name__ == "__main__":
    # Test memory.add
    print("Testing memory.add...")
    add_result = add_memory("I love playing basketball", user_id="alice")
    print(f"Add result: {add_result}")
    
    # Test memory.search
    print("\nTesting memory.search...")
    search_results = search_memory("sports", user_id="alice")
    print(f"Search results: {search_results}")
