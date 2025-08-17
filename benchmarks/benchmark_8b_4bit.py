#!/usr/bin/env python3
"""
Benchmark script for Llama 3.1 8B model in 4-bit mode
Measures latency and token throughput for 100 text prompts
"""

import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import os

# Model path from the existing setup
MODEL_PATH = "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--unsloth--llama-3.1-8b-bnb-4bit/snapshots/b80adf5d249b569469d0a19192ff36e88f133413"

# Function to load test prompts from file
def load_test_prompts():
    """Load test prompts from text file for consistent benchmarking"""
    prompts_file = os.path.join(os.path.dirname(__file__), "test_prompts.txt")
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

class Benchmark8B:
    def __init__(self):
        print("Loading Llama 3.1 8B model in 4-bit mode...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            load_in_4bit=True
        )
        print(f"Model loaded on device: {next(self.model.parameters()).device}")
        
    def generate_text(self, prompt, max_new_tokens=128):
        """Generate text and measure latency"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Calculate tokens generated
        input_length = inputs['input_ids'].shape[1]
        total_length = outputs.shape[1]
        tokens_generated = total_length - input_length
        
        # Calculate throughput (tokens per second)
        throughput = tokens_generated / latency if latency > 0 else 0
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        return {
            'latency': latency,
            'tokens_generated': tokens_generated,
            'throughput': throughput,
            'response': response
        }
    
    def run_benchmark(self, num_prompts=100):
        """Run benchmark on specified number of prompts"""
        print(f"Starting benchmark with {num_prompts} prompts...")
        
        results = {
            'model_info': {
                'model_path': MODEL_PATH,
                'quantization': '4-bit',
                'device': str(next(self.model.parameters()).device),
                'timestamp': datetime.now().isoformat()
            },
            'individual_results': [],
            'summary': {}
        }
        
        total_latency = 0
        total_tokens = 0
        total_throughput = 0
        
        # Load and use first num_prompts from our test set
        test_prompts = load_test_prompts()
        prompts_to_test = test_prompts[:num_prompts]
        
        for i, prompt in enumerate(prompts_to_test, 1):
            print(f"Processing prompt {i}/{num_prompts}...")
            
            result = self.generate_text(prompt)
            
            # Store individual result
            individual_result = {
                'prompt_id': i,
                'prompt': prompt,
                'latency': result['latency'],
                'tokens_generated': result['tokens_generated'],
                'throughput': result['throughput'],
                'response_length': len(result['response'])
            }
            results['individual_results'].append(individual_result)
            
            # Accumulate for averages
            total_latency += result['latency']
            total_tokens += result['tokens_generated']
            total_throughput += result['throughput']
            
            print(f"  Latency: {result['latency']:.3f}s, Tokens: {result['tokens_generated']}, Throughput: {result['throughput']:.2f} tok/s")
        
        # Calculate summary statistics
        avg_latency = total_latency / num_prompts
        avg_tokens = total_tokens / num_prompts
        avg_throughput = total_throughput / num_prompts
        
        results['summary'] = {
            'num_prompts': num_prompts,
            'average_latency': avg_latency,
            'average_tokens_generated': avg_tokens,
            'average_throughput': avg_throughput,
            'total_latency': total_latency,
            'total_tokens': total_tokens
        }
        
        return results
    
    def save_results(self, results, filename=None):
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/home/ubuntu/benchmarks/8b_4bit_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filename}")
        return filename

def main():
    # Initialize benchmark
    benchmark = Benchmark8B()
    
    # Run benchmark
    results = benchmark.run_benchmark(num_prompts=100)
    
    # Save results
    results_file = benchmark.save_results(results)
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY - Llama 3.1 8B (4-bit)")
    print("="*50)
    print(f"Number of prompts: {summary['num_prompts']}")
    print(f"Average latency: {summary['average_latency']:.3f} seconds")
    print(f"Average tokens generated: {summary['average_tokens_generated']:.1f}")
    print(f"Average throughput: {summary['average_throughput']:.2f} tokens/second")
    print(f"Total processing time: {summary['total_latency']:.3f} seconds")
    print("="*50)
    
    return results_file

if __name__ == "__main__":
    main()
