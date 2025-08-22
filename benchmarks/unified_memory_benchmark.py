#!/usr/bin/env python3
"""
Unified Memory Retrieval Benchmark Script
Tests memory retrieval quality across different fine-tuned models
Measures precision@k and retrieval performance for memory-aware conversations
"""

import time
import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add the mem0-backend path to import the memory functions
sys.path.append('/home/ubuntu/mem0-assignment/mem0-backend')

class UnifiedMemoryBenchmark:
    def __init__(self, model_path=None, model_type="transformers"):
        """
        Initialize memory benchmark
        
        Args:
            model_path: Path to model (for future model-specific memory integration)
            model_type: Type of model being tested
        """
        print("🧠 Initializing Memory Retrieval Benchmark...")
        self.model_path = model_path
        self.model_type = model_type
        self.user_id = f"benchmark_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Import memory functions and set up model if specified
        try:
            from main import add_memory, search_memory, memory, get_model_choice_for_benchmark
            self.add_memory = add_memory
            self.search_memory = search_memory
            self.memory = memory
            
            # If a model is specified, we need to ensure the correct model is loaded
            if model_path:
                # Get the benchmark model name from the path
                benchmark_model_name = model_path.split('/')[-1] if '/' in str(model_path) else str(model_path)
                # This is mainly for reference - the actual model loading happens in main.py
                print(f"🎯 Benchmark will reference model: {benchmark_model_name}")
            
            print("✅ Memory functions loaded successfully")
        except ImportError as e:
            print(f"❌ Failed to import memory functions: {e}")
            sys.exit(1)
    
    def load_synthetic_memories(self):
        """Load synthetic memories from text file"""
        memories_file = Path(__file__).parent / "synthetic_memories.txt"
        
        if not memories_file.exists():
            # Create default synthetic memories if file doesn't exist
            default_memories = [
                "I love drinking coffee in the morning, especially dark roast.",
                "My favorite food is pizza, particularly with pepperoni and mushrooms.",
                "I enjoy playing basketball on weekends with my friends.",
                "I studied Computer Science at Stanford University.",
                "I work as a software engineer at Google in Mountain View.",
                "My dog Max is a golden retriever who loves to play fetch.",
                "I play guitar and my favorite band is Electric Dreams.",
                "I've traveled to Japan and loved visiting Tokyo.",
                "I'm learning Spanish and can speak basic French.",
                "I practice yoga every morning at 5:30 AM.",
                "My sister works as a doctor at Mount Sinai Hospital.",
                "I'm allergic to shellfish and carry an EpiPen.",
                "I love reading science fiction, especially Hitchhiker's Guide to the Galaxy.",
                "I'm afraid of heights but love rock climbing paradoxically.",
                "I collect vinyl records from the 1970s jazz era.",
                "I'm left-handed and have a star-shaped birthmark on my shoulder.",
                "My wife Sarah is a photographer who specializes in nature shots.",
                "I volunteer at the local animal shelter on Sundays.",
                "I'm training for a marathon and run 5 miles daily.",
                "I meditate for 20 minutes before breakfast.",
                "My grandfather taught me woodworking when I was young.",
                "I'm lactose intolerant but love ice cream anyway.",
                "I studied abroad in Switzerland during college.",
                "I play chess competitively and have a rating of 1800.",
                "I'm learning calligraphy as a hobby.",
                "My favorite book is 'Sapiens' by Yuval Noah Harari.",
                "I have a fear of spiders but love other insects.",
                "I enjoy cooking Italian cuisine, especially pasta dishes.",
                "I've been skydiving three times and love the adrenaline rush.",
                "I practice photography with vintage film cameras.",
                "I'm a vegetarian and have been for five years.",
                "I play Dungeons & Dragons every Friday night.",
                "I love hiking and completed the Pacific Crest Trail section in Oregon.",
                "I have a collection of pocket watches from different eras.",
                "I'm claustrophobic but love exploring caves.",
                "I speak Portuguese fluently from living in Brazil for two years.",
                "I enjoy sailing and own a small sailboat.",
                "I'm double-jointed in my thumbs.",
                "I love solving jigsaw puzzles, especially 1000+ pieces.",
                "I have a scar on my knee from a childhood bicycle accident.",
                "I practice blacksmithing as a weekend hobby.",
                "I'm afraid of public speaking but love teaching.",
                "I enjoy escape rooms and have completed over 50.",
                "I love chocolate dark chocolate with at least 70% cocoa.",
                "I collect snow globes from different countries I visit.",
                "I'm learning sign language to communicate better with deaf community.",
                "I love bread baking and make sourdough every weekend.",
                "I have a therapy animal certification for my dog.",
                "I enjoy crossword puzzles and do the New York Times puzzle daily.",
                "I love vintage postcards and have over 300 in my collection."
            ]
            
            with open(memories_file, 'w') as f:
                for memory in default_memories:
                    f.write(memory + '\n')
            print(f"📝 Created default synthetic memories file: {memories_file}")
        
        with open(memories_file, 'r', encoding='utf-8') as f:
            memories = [line.strip() for line in f if line.strip()]
        return memories
    
    def create_query_set(self):
        """Create representative queries for testing memory retrieval"""
        queries = [
            # Personal preferences and food
            {"query": "favorite food", "relevant_memories": ["coffee", "pizza", "chocolate", "bread", "ice cream"]},
            {"query": "dietary restrictions", "relevant_memories": ["shellfish", "lactose intolerant", "vegetarian", "EpiPen"]},
            {"query": "cooking and food preferences", "relevant_memories": ["Italian cuisine", "pasta", "vegetarian", "bread", "sourdough"]},
            
            # Sports and fitness activities
            {"query": "sports activities", "relevant_memories": ["basketball", "tennis", "running", "hiking", "climbing"]},
            {"query": "fitness and exercise", "relevant_memories": ["marathon", "yoga", "rock climbing", "5 miles", "5:30 AM"]},
            {"query": "outdoor activities", "relevant_memories": ["hiking", "Pacific Crest Trail", "sailing", "skydiving"]},
            
            # Education and career
            {"query": "education background", "relevant_memories": ["Stanford", "Computer Science", "college", "studied abroad"]},
            {"query": "work and career", "relevant_memories": ["software engineer", "Google", "Mountain View"]},
            {"query": "international experience", "relevant_memories": ["Japan", "Switzerland", "Brazil", "Portuguese"]},
            
            # Pets and animals
            {"query": "pets and animals", "relevant_memories": ["dog Max", "golden retriever", "animal shelter", "therapy animal"]},
            
            # Hobbies and interests
            {"query": "music and instruments", "relevant_memories": ["guitar", "Electric Dreams", "jazz", "vinyl records"]},
            {"query": "reading and books", "relevant_memories": ["Hitchhiker's Guide", "Sapiens", "science fiction", "crossword"]},
            {"query": "creative hobbies", "relevant_memories": ["guitar", "photography", "calligraphy", "woodworking", "blacksmithing"]},
            {"query": "games and puzzles", "relevant_memories": ["chess", "Dungeons & Dragons", "escape rooms", "jigsaw puzzles"]},
            {"query": "collections", "relevant_memories": ["vinyl records", "pocket watches", "snow globes", "postcards"]},
            
            # Travel experiences
            {"query": "travel experiences", "relevant_memories": ["Japan", "Tokyo", "Switzerland", "Brazil", "countries"]},
            
            # Skills and learning
            {"query": "languages spoken", "relevant_memories": ["Spanish", "French", "Portuguese", "sign language"]},
            
            # Personal characteristics
            {"query": "fears and phobias", "relevant_memories": ["heights", "spiders", "claustrophobic", "public speaking"]},
            {"query": "family relationships", "relevant_memories": ["sister", "doctor", "Mount Sinai", "wife Sarah", "grandfather"]},
            {"query": "physical characteristics", "relevant_memories": ["left-handed", "birthmark", "star", "scar", "double-jointed"]},
            {"query": "daily routines", "relevant_memories": ["5:30 AM", "morning", "crossword", "meditation", "yoga"]}
        ]
        return queries
    
    def setup_memories(self, memories):
        """Add synthetic memories to Mem0"""
        print(f"📚 Adding {len(memories)} synthetic memories...")
        added_count = 0
        failed_memories = []
        
        for i, memory_text in enumerate(memories, 1):
            try:
                result = self.add_memory(memory_text, user_id=self.user_id)
                added_count += 1
                if i % 10 == 0:
                    print(f"  ✅ Added {i}/{len(memories)} memories...")
            except Exception as e:
                print(f"  ❌ Error adding memory {i}: {e}")
                failed_memories.append((i, memory_text, str(e)))
                
        print(f"✅ Successfully added {added_count}/{len(memories)} memories")
        if failed_memories:
            print(f"⚠️ Failed to add {len(failed_memories)} memories")
        
        return added_count, failed_memories
    
    def calculate_precision_at_k(self, retrieved_memories, relevant_keywords, k=5):
        """Calculate precision@k for retrieved memories"""
        if not retrieved_memories:
            return 0.0
            
        # Take top k results
        top_k = retrieved_memories[:k]
        
        # Count how many contain relevant keywords
        relevant_count = 0
        matched_keywords = []
        
        for mem in top_k:
            memory_text = mem.get('memory', '').lower()
            matched_in_this_memory = []
            
            for keyword in relevant_keywords:
                if keyword.lower() in memory_text:
                    matched_in_this_memory.append(keyword)
            
            if matched_in_this_memory:
                relevant_count += 1
                matched_keywords.extend(matched_in_this_memory)
                
        precision = relevant_count / min(len(top_k), k)
        return precision, list(set(matched_keywords))
    
    def run_retrieval_benchmark(self, queries):
        """Run retrieval benchmark on query set"""
        print(f"🎯 Running retrieval benchmark with {len(queries)} queries...")
        
        results = {
            'benchmark_info': {
                'user_id': self.user_id,
                'model_path': self.model_path,
                'model_type': self.model_type,
                'num_queries': len(queries),
                'precision_k': 5,
                'timestamp': datetime.now().isoformat()
            },
            'individual_results': [],
            'summary': {}
        }
        
        total_precision = 0
        total_retrieval_time = 0
        precision_scores = []
        retrieval_times = []
        
        for i, query_data in enumerate(queries, 1):
            query = query_data['query']
            relevant_keywords = query_data['relevant_memories']
            
            print(f"🔍 Processing query {i}/{len(queries)}: '{query}'")
            
            # Measure retrieval time
            start_time = time.time()
            try:
                search_result = self.search_memory(query, user_id=self.user_id)
                end_time = time.time()
                retrieval_time = end_time - start_time
                
                # Extract results
                if isinstance(search_result, dict):
                    retrieved_memories = search_result.get('results', [])
                else:
                    retrieved_memories = search_result or []
                
                # Calculate precision@5
                precision, matched_keywords = self.calculate_precision_at_k(
                    retrieved_memories, relevant_keywords, k=5
                )
                
                # Store individual result
                individual_result = {
                    'query_id': i,
                    'query': query,
                    'relevant_keywords': relevant_keywords,
                    'matched_keywords': matched_keywords,
                    'num_retrieved': len(retrieved_memories),
                    'precision_at_5': precision,
                    'retrieval_time': retrieval_time,
                    'top_5_memories': [mem.get('memory', '') for mem in retrieved_memories[:5]],
                    'success': True
                }
                
                # Accumulate metrics
                total_precision += precision
                total_retrieval_time += retrieval_time
                precision_scores.append(precision)
                retrieval_times.append(retrieval_time)
                
                print(f"  📊 Retrieved: {len(retrieved_memories)} memories | Precision@5: {precision:.3f} | Time: {retrieval_time:.3f}s")
                
            except Exception as e:
                print(f"  ❌ Error processing query: {e}")
                individual_result = {
                    'query_id': i,
                    'query': query,
                    'relevant_keywords': relevant_keywords,
                    'error': str(e),
                    'success': False
                }
                retrieval_time = 0
            
            results['individual_results'].append(individual_result)
        
        # Calculate summary statistics
        successful_queries = [r for r in results['individual_results'] if r.get('success', False)]
        num_successful = len(successful_queries)
        
        if num_successful > 0:
            avg_precision = total_precision / num_successful
            avg_retrieval_time = total_retrieval_time / num_successful
            
            # Calculate precision distribution
            precision_scores.sort()
            retrieval_times.sort()
            
            perfect_queries = sum(1 for p in precision_scores if p == 1.0)
            zero_queries = sum(1 for p in precision_scores if p == 0.0)
            
            results['summary'] = {
                'num_queries': len(queries),
                'successful_queries': num_successful,
                'failed_queries': len(queries) - num_successful,
                'average_precision_at_5': avg_precision,
                'median_precision_at_5': precision_scores[len(precision_scores)//2] if precision_scores else 0,
                'average_retrieval_time': avg_retrieval_time,
                'median_retrieval_time': retrieval_times[len(retrieval_times)//2] if retrieval_times else 0,
                'perfect_precision_queries': perfect_queries,
                'zero_precision_queries': zero_queries,
                'total_retrieval_time': total_retrieval_time,
                'precision_distribution': {
                    'min': min(precision_scores) if precision_scores else 0,
                    'max': max(precision_scores) if precision_scores else 0,
                    'p95': precision_scores[int(len(precision_scores)*0.95)] if precision_scores else 0
                },
                'retrieval_time_distribution': {
                    'min': min(retrieval_times) if retrieval_times else 0,
                    'max': max(retrieval_times) if retrieval_times else 0,
                    'p95': retrieval_times[int(len(retrieval_times)*0.95)] if retrieval_times else 0
                }
            }
        else:
            results['summary'] = {
                'num_queries': len(queries),
                'successful_queries': 0,
                'failed_queries': len(queries),
                'error': 'No successful queries'
            }
        
        return results
    
    def save_results(self, results, output_dir):
        """Save benchmark results to specified directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on model info
        if self.model_path:
            model_name = Path(self.model_path).name.replace('/', '_')
        else:
            model_name = "default"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"memory_benchmark_{model_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def cleanup_memories(self):
        """Clean up test memories from the system"""
        try:
            # This would need to be implemented based on the mem0 cleanup API
            print(f"🧹 Cleaning up memories for user: {self.user_id}")
            # Implementation depends on mem0's deletion capabilities
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up memories: {e}")

def get_model_configs():
    """Get predefined model configurations"""
    return {
        "llama3.1": {
            "path": "unsloth/llama-3.1-8b-bnb-4bit",
            "type": "transformers",
            "quantization": "4bit"
        },
        "llama3.1-instruct-bf16": {
            "path": "meta-llama/Llama-3.1-8B-Instruct",
            "type": "transformers",
            "quantization": "bf16"
        },
        "llama4": {
            "path": "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct",
            "type": "transformers",
            "quantization": None
        },
        "llama4-bf16": {
            "path": "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct",
            "type": "transformers",
            "quantization": "bf16"
        },
        "llama4-4bit": {
            "path": "mlx-community/meta-llama-Llama-4-Scout-17B-16E-4bit",
            "type": "transformers",
            "quantization": "4bit"
        },
        "llama4-gguf": {
            "path": "/home/ubuntu/mem0-assignment/model_cache/scout_gguf/Q4_K_M",
            "type": "gguf",
            "quantization": "4bit"
        }
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified Memory Retrieval Benchmark")
    
    # Model selection - either predefined or custom
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", type=str, 
                           choices=list(get_model_configs().keys()),
                           help="Predefined model to benchmark")
    model_group.add_argument("--model-path", type=str,
                           help="Custom path to model being tested (for reference)")
    
    # Model configuration (only used with --model-path)
    parser.add_argument("--model-type", type=str, 
                       choices=["transformers", "gguf", "local"],
                       default="transformers",
                       help="Type of model being tested (only used with --model-path)")
    
    # Benchmark configuration
    parser.add_argument("--output-dir", type=str,
                       default="/home/ubuntu/mem0-assignment/benchmarks",
                       help="Output directory for results")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up test memories after benchmark")
    
    return parser.parse_args()

def main():
    """Main benchmark function"""
    args = parse_args()
    
    try:
        # Determine model configuration
        if args.model:
            # Use predefined model configuration
            model_configs = get_model_configs()
            config = model_configs[args.model]
            model_path = config["path"]
            model_type = config["type"]
            print(f"🎯 Using predefined model: {args.model}")
        else:
            # Use custom model configuration or default
            model_path = args.model_path
            model_type = args.model_type
            if model_path:
                print(f"🎯 Using custom model: {model_path}")
            else:
                print("🎯 Using default memory benchmark (no specific model)")
        
        # Set output directory based on model
        if args.model in ["llama4", "llama4-bf16", "llama4-4bit", "llama4-gguf"]:
            output_dir = "/home/ubuntu/mem0-assignment/benchmarks/scout/base_model_results"
        elif args.model == "llama3.1-instruct-bf16":
            output_dir = "/home/ubuntu/mem0-assignment/benchmarks/base_model_results_bf16"
        else:
            output_dir = args.output_dir
        
        # Initialize benchmark
        benchmark = UnifiedMemoryBenchmark(
            model_path=model_path,
            model_type=model_type
        )
        
        # Load synthetic memories and add them to Mem0
        print("📚 Loading synthetic memories...")
        memories = benchmark.load_synthetic_memories()
        added_count, failed_memories = benchmark.setup_memories(memories)
        
        if added_count == 0:
            print("❌ No memories were added successfully. Exiting.")
            return None
        
        # Create query set
        queries = benchmark.create_query_set()
        
        # Run retrieval benchmark
        results = benchmark.run_retrieval_benchmark(queries)
        
        # Save results
        results_file = benchmark.save_results(results, output_dir)
        
        # Print summary
        if 'summary' in results and 'successful_queries' in results['summary']:
            summary = results['summary']
            print("\n" + "="*70)
            print("🧠 MEMORY RETRIEVAL BENCHMARK SUMMARY")
            print("="*70)
            print(f"Model: {model_path or 'Default'}")
            print(f"Type: {model_type}")
            print(f"Synthetic memories added: {added_count}")
            print(f"Successful queries: {summary['successful_queries']}/{summary['num_queries']}")
            
            if summary['successful_queries'] > 0:
                print(f"Average Precision@5: {summary['average_precision_at_5']:.3f}")
                print(f"Median Precision@5: {summary['median_precision_at_5']:.3f}")
                print(f"Perfect precision queries: {summary['perfect_precision_queries']}")
                print(f"Zero precision queries: {summary['zero_precision_queries']}")
                print(f"Average retrieval time: {summary['average_retrieval_time']:.3f}s")
                print(f"Median retrieval time: {summary['median_retrieval_time']:.3f}s")
                print(f"Precision range: {summary['precision_distribution']['min']:.3f} - {summary['precision_distribution']['max']:.3f}")
            
            print("="*70)
        
        # Cleanup if requested
        if args.cleanup:
            benchmark.cleanup_memories()
        
        return results_file
        
    except Exception as e:
        print(f"❌ Memory benchmark failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
