#!/usr/bin/env python3
"""
Memory Retrieval Quality Benchmark for Mem0
Measures retrieval precision@5 on synthetic memories using representative queries
"""

import time
import json
import sys
import os
from datetime import datetime

# Add the mem0-backend path to import the memory functions
sys.path.append('/home/ubuntu/mem0-assignment/mem0-backend')
from main import add_memory, search_memory, memory

def load_synthetic_memories():
    """Load synthetic memories from text file"""
    memories_file = os.path.join(os.path.dirname(__file__), "synthetic_memories.txt")
    with open(memories_file, 'r', encoding='utf-8') as f:
        memories = [line.strip() for line in f if line.strip()]
    return memories

def create_query_set():
    """Create representative queries for testing memory retrieval"""
    queries = [
        # Personal preferences
        {"query": "favorite food", "relevant_memories": ["coffee", "pizza", "chocolate", "bread", "ice cream"]},
        {"query": "sports activities", "relevant_memories": ["basketball", "tennis", "running", "hiking", "climbing"]},
        {"query": "education background", "relevant_memories": ["Stanford", "Computer Science", "college", "studied abroad"]},
        {"query": "work and career", "relevant_memories": ["software engineer", "Google", "Mountain View"]},
        {"query": "pets and animals", "relevant_memories": ["dog Max", "golden retriever", "animal shelter", "therapy animal"]},
        
        # Hobbies and interests
        {"query": "music and instruments", "relevant_memories": ["guitar", "Electric Dreams", "jazz", "vinyl records"]},
        {"query": "reading and books", "relevant_memories": ["Hitchhiker's Guide", "Sapiens", "novel", "bookstore"]},
        {"query": "travel experiences", "relevant_memories": ["Japan", "Tokyo", "Switzerland", "Iceland", "Prague"]},
        {"query": "cooking and food", "relevant_memories": ["Italian cuisine", "pasta", "vegetarian", "roast beans"]},
        {"query": "outdoor activities", "relevant_memories": ["hiking", "Pacific Crest Trail", "sailing", "skydiving"]},
        
        # Skills and learning
        {"query": "languages spoken", "relevant_memories": ["English", "Spanish", "French", "Portuguese", "sign language"]},
        {"query": "creative hobbies", "relevant_memories": ["guitar", "photography", "calligraphy", "woodworking", "blacksmithing"]},
        {"query": "games and puzzles", "relevant_memories": ["chess", "Dungeons & Dragons", "Scrabble", "escape rooms", "jigsaw puzzles"]},
        {"query": "health and fitness", "relevant_memories": ["marathon", "yoga", "tennis", "rock climbing"]},
        {"query": "collections", "relevant_memories": ["vinyl records", "pocket watches", "snow globes", "postcards"]},
        
        # Personal characteristics
        {"query": "fears and phobias", "relevant_memories": ["heights", "spiders", "claustrophobic", "public speaking"]},
        {"query": "allergies and dietary", "relevant_memories": ["shellfish", "lactose intolerant", "vegetarian", "EpiPen"]},
        {"query": "family relationships", "relevant_memories": ["sister", "doctor", "Mount Sinai", "wife Sarah", "grandfather"]},
        {"query": "physical characteristics", "relevant_memories": ["left-handed", "birthmark", "star", "scar", "double-jointed"]},
        {"query": "daily routines", "relevant_memories": ["5:30 AM", "morning", "crossword", "meditation", "yoga"]}
    ]
    return queries

class MemoryRetrievalBenchmark:
    def __init__(self):
        """Initialize the memory retrieval benchmark"""
        print("Initializing Memory Retrieval Benchmark...")
        self.user_id = "benchmark_user"
        
    def setup_memories(self, memories):
        """Add synthetic memories to Mem0"""
        print(f"Adding {len(memories)} synthetic memories...")
        added_count = 0
        
        for i, memory_text in enumerate(memories, 1):
            try:
                result = add_memory(memory_text, user_id=self.user_id)
                added_count += 1
                if i % 10 == 0:
                    print(f"  Added {i}/{len(memories)} memories...")
            except Exception as e:
                print(f"  Error adding memory {i}: {e}")
                
        print(f"Successfully added {added_count}/{len(memories)} memories")
        return added_count
    
    def calculate_precision_at_k(self, retrieved_memories, relevant_keywords, k=5):
        """Calculate precision@k for retrieved memories"""
        if not retrieved_memories:
            return 0.0
            
        # Take top k results
        top_k = retrieved_memories[:k]
        
        # Count how many contain relevant keywords
        relevant_count = 0
        for mem in top_k:
            memory_text = mem.get('memory', '').lower()
            if any(keyword.lower() in memory_text for keyword in relevant_keywords):
                relevant_count += 1
                
        precision = relevant_count / min(len(top_k), k)
        return precision
    
    def run_retrieval_benchmark(self, queries):
        """Run retrieval benchmark on query set"""
        print(f"Running retrieval benchmark with {len(queries)} queries...")
        
        results = {
            'benchmark_info': {
                'user_id': self.user_id,
                'num_queries': len(queries),
                'precision_k': 5,
                'timestamp': datetime.now().isoformat()
            },
            'individual_results': [],
            'summary': {}
        }
        
        total_precision = 0
        total_retrieval_time = 0
        
        for i, query_data in enumerate(queries, 1):
            query = query_data['query']
            relevant_keywords = query_data['relevant_memories']
            
            print(f"Processing query {i}/{len(queries)}: '{query}'")
            
            # Measure retrieval time
            start_time = time.time()
            search_result = search_memory(query, user_id=self.user_id)
            end_time = time.time()
            
            retrieval_time = end_time - start_time
            
            # Extract results
            retrieved_memories = search_result.get('results', []) if isinstance(search_result, dict) else search_result
            
            # Calculate precision@5
            precision = self.calculate_precision_at_k(retrieved_memories, relevant_keywords, k=5)
            
            # Store individual result
            individual_result = {
                'query_id': i,
                'query': query,
                'relevant_keywords': relevant_keywords,
                'num_retrieved': len(retrieved_memories),
                'precision_at_5': precision,
                'retrieval_time': retrieval_time,
                'top_5_memories': [mem.get('memory', '') for mem in retrieved_memories[:5]]
            }
            results['individual_results'].append(individual_result)
            
            # Accumulate for averages
            total_precision += precision
            total_retrieval_time += retrieval_time
            
            print(f"  Retrieved: {len(retrieved_memories)} memories, Precision@5: {precision:.3f}, Time: {retrieval_time:.3f}s")
        
        # Calculate summary statistics
        avg_precision = total_precision / len(queries)
        avg_retrieval_time = total_retrieval_time / len(queries)
        
        # Calculate precision distribution
        precision_scores = [r['precision_at_5'] for r in results['individual_results']]
        perfect_queries = sum(1 for p in precision_scores if p == 1.0)
        zero_queries = sum(1 for p in precision_scores if p == 0.0)
        
        results['summary'] = {
            'num_queries': len(queries),
            'average_precision_at_5': avg_precision,
            'average_retrieval_time': avg_retrieval_time,
            'perfect_precision_queries': perfect_queries,
            'zero_precision_queries': zero_queries,
            'total_retrieval_time': total_retrieval_time,
            'precision_distribution': {
                'min': min(precision_scores),
                'max': max(precision_scores),
                'median': sorted(precision_scores)[len(precision_scores)//2]
            }
        }
        
        return results
    
    def save_results(self, results, filename=None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/home/ubuntu/mem0-assignment/benchmarks/memory_retrieval_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filename}")
        return filename

def main():
    # Initialize benchmark
    benchmark = MemoryRetrievalBenchmark()
    
    # Load synthetic memories and add them to Mem0
    print("Loading synthetic memories...")
    memories = load_synthetic_memories()
    added_count = benchmark.setup_memories(memories)
    
    if added_count == 0:
        print("No memories were added successfully. Exiting.")
        return
    
    # Create query set
    queries = create_query_set()
    
    # Run retrieval benchmark
    results = benchmark.run_retrieval_benchmark(queries)
    
    # Save results
    results_file = benchmark.save_results(results)
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*60)
    print("MEMORY RETRIEVAL BENCHMARK SUMMARY")
    print("="*60)
    print(f"Number of synthetic memories: {added_count}")
    print(f"Number of test queries: {summary['num_queries']}")
    print(f"Average Precision@5: {summary['average_precision_at_5']:.3f}")
    print(f"Average retrieval time: {summary['average_retrieval_time']:.3f} seconds")
    print(f"Queries with perfect precision (1.0): {summary['perfect_precision_queries']}")
    print(f"Queries with zero precision (0.0): {summary['zero_precision_queries']}")
    print(f"Precision range: {summary['precision_distribution']['min']:.3f} - {summary['precision_distribution']['max']:.3f}")
    print(f"Median precision: {summary['precision_distribution']['median']:.3f}")
    print("="*60)
    
    return results_file

if __name__ == "__main__":
    main()
