#!/usr/bin/env python3
"""
Unified Inference Benchmark Script
Supports multiple model types: base HF models, GGUF models, and fine-tuned models
Measures latency and token throughput for text generation tasks
"""

import time
import json
import torch
import argparse
import os
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from huggingface_hub import hf_hub_download, snapshot_download
import sys

# Try to import Llama4 model for multimodal support
try:
    from transformers import Llama4ForConditionalGeneration
    LLAMA4_AVAILABLE = True
except ImportError:
    LLAMA4_AVAILABLE = False

class UnifiedInferenceBenchmark:
    def __init__(self, model_path, model_type="transformers", quantization=None):
        """
        Initialize benchmark with specified model
        
        Args:
            model_path: Path to model (local or HuggingFace repo)
            model_type: "transformers", "gguf", or "local"
            quantization: "4bit", "8bit", or None for full precision
        """
        self.model_path = model_path
        self.model_type = model_type
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_llama4 = False
        
        print(f"🚀 Initializing {model_type} model: {model_path}")
        self.load_model()
        
    def load_model(self):
        """Load model based on type and configuration"""
        if self.model_type == "transformers":
            self.load_transformers_model()
        elif self.model_type == "gguf":
            self.load_gguf_model()
        elif self.model_type == "local":
            self.load_local_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def load_transformers_model(self):
        """Load standard transformers model from HuggingFace or local path"""
        print(f"📥 Loading transformers model...")
        
        # Handle HuggingFace cache directory structure for Llama 4 Scout
        model_path = self.model_path
        if ("models--meta-llama--Llama-4-Scout" in model_path or "models--unsloth--Llama-4-Scout" in model_path) and os.path.isdir(model_path):
            # Look for the actual model files in snapshots directory
            snapshots_dir = os.path.join(model_path, "snapshots")
            if os.path.exists(snapshots_dir):
                # Find the latest snapshot
                snapshot_dirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshot_dirs:
                    # Use the first (and likely only) snapshot
                    model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                    print(f"🔍 Using snapshot: {model_path}")
        
        # Check if this is a Llama4 model by looking at config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                if config.get("model_type") == "llama4":
                    self.is_llama4 = True
                    print("🦙 Detected Llama4 multimodal model")
        
        # Load tokenizer/processor based on model type
        if self.is_llama4 and LLAMA4_AVAILABLE:
            print("📥 Loading Llama4 processor...")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            # For text-only tasks, we can still use the tokenizer
            self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else None
        else:
            print("📥 Loading standard tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure model loading parameters
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        
        # Set torch dtype and quantization based on configuration
        if self.quantization == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.quantization == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            # Default to bfloat16 for no quantization
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        # Load model with appropriate class
        if self.is_llama4 and LLAMA4_AVAILABLE:
            print("📥 Loading Llama4ForConditionalGeneration...")
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs
            )
        else:
            print("📥 Loading AutoModelForCausalLM...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
        
        print(f"✅ Model loaded on device: {next(self.model.parameters()).device}")
        
    def load_gguf_model(self):
        """Load GGUF model using llama-cpp-python"""
        try:
            from llama_cpp import Llama
            print(f"📥 Loading GGUF model...")
            
            # For GGUF models, we need the actual .gguf file
            if os.path.isdir(self.model_path):
                # Find GGUF files in directory
                gguf_files = list(Path(self.model_path).glob("*.gguf"))
                if not gguf_files:
                    raise FileNotFoundError(f"No GGUF files found in {self.model_path}")
                
                # Choose based on quantization preference
                if self.quantization == "4bit":
                    gguf_file = next((f for f in gguf_files if "Q4" in f.name), gguf_files[0])
                elif self.quantization == "bf16":
                    gguf_file = next((f for f in gguf_files if "BF16" in f.name), gguf_files[0])
                else:
                    gguf_file = gguf_files[0]
                    
                model_file = str(gguf_file)
            else:
                model_file = self.model_path
            
            # Load GGUF model
            self.model = Llama(
                model_path=model_file,
                n_ctx=2048,
                n_threads=8,
                verbose=False
            )
            
            # For GGUF models, we'll use the model's built-in tokenization
            self.tokenizer = None
            print(f"✅ GGUF model loaded: {model_file}")
            
        except ImportError:
            print("❌ llama-cpp-python not installed. Installing...")
            os.system("pip install llama-cpp-python")
            # Retry import
            from llama_cpp import Llama
            self.load_gguf_model()
    
    def load_local_model(self):
        """Load local fine-tuned model"""
        print(f"📥 Loading local model...")
        
        # Check if it's a LoRA adapter or full model
        if (Path(self.model_path) / "adapter_config.json").exists():
            # It's a LoRA adapter, need to load with base model
            print("🔄 Detected LoRA adapter, loading with base model...")
            from peft import PeftModel
            
            # Load base model and tokenizer
            base_model_path = "unsloth/llama-3.1-8b-bnb-4bit"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_4bit=(self.quantization == "4bit")
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            print("✅ LoRA adapter loaded successfully")
        else:
            # Full model
            self.load_transformers_model()
    
    def generate_text(self, prompt, max_new_tokens=128):
        """Generate text and measure performance metrics"""
        if self.model_type == "gguf":
            return self.generate_gguf(prompt, max_new_tokens)
        else:
            return self.generate_transformers(prompt, max_new_tokens)
    
    def generate_transformers(self, prompt, max_new_tokens=128):
        """Generate text using transformers models"""
        if self.is_llama4 and self.processor:
            return self.generate_llama4(prompt, max_new_tokens)
        else:
            return self.generate_standard_transformers(prompt, max_new_tokens)
    
    def generate_llama4(self, prompt, max_new_tokens=128):
        """Generate text using Llama4 multimodal models"""
        # For text-only generation with Llama4, we create a text-only message
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        start_time = time.time()
        
        # Use processor to prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Calculate tokens generated
        input_length = inputs['input_ids'].shape[-1]
        total_length = outputs.shape[-1]
        tokens_generated = total_length - input_length
        
        # Calculate throughput
        throughput = tokens_generated / latency if latency > 0 else 0
        
        # Decode response
        response = self.processor.batch_decode(outputs[:, input_length:])[0]
        
        return {
            'latency': latency,
            'tokens_generated': tokens_generated,
            'throughput': throughput,
            'response': response.strip(),
            'input_tokens': input_length,
            'total_tokens': total_length
        }
    
    def generate_standard_transformers(self, prompt, max_new_tokens=128):
        """Generate text using standard transformers models"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Calculate tokens generated
        input_length = inputs['input_ids'].shape[1]
        total_length = outputs.shape[1]
        tokens_generated = total_length - input_length
        
        # Calculate throughput
        throughput = tokens_generated / latency if latency > 0 else 0
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        return {
            'latency': latency,
            'tokens_generated': tokens_generated,
            'throughput': throughput,
            'response': response,
            'input_tokens': input_length,
            'total_tokens': total_length
        }
    
    def generate_gguf(self, prompt, max_new_tokens=128):
        """Generate text using GGUF models"""
        start_time = time.time()
        
        # Generate with GGUF model
        output = self.model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Extract response
        response = output['choices'][0]['text']
        
        # Estimate tokens (rough approximation)
        tokens_generated = len(response.split()) * 1.3  # Rough token estimate
        throughput = tokens_generated / latency if latency > 0 else 0
        
        return {
            'latency': latency,
            'tokens_generated': int(tokens_generated),
            'throughput': throughput,
            'response': response,
            'input_tokens': len(prompt.split()) * 1.3,  # Rough estimate
            'total_tokens': int(len(prompt.split()) * 1.3 + tokens_generated)
        }
    
    def load_test_prompts(self):
        """Load test prompts from file"""
        prompts_file = Path(__file__).parent / "test_prompts.txt"
        if not prompts_file.exists():
            # Create default prompts if file doesn't exist
            default_prompts = [
                "Tell me about your favorite hobby and why you enjoy it.",
                "What's the most interesting place you've ever visited?",
                "Describe a challenging problem you solved recently.",
                "What are your thoughts on the future of artificial intelligence?",
                "Share a memorable experience from your childhood.",
                "What skills would you like to develop in the next year?",
                "Describe your ideal weekend activity.",
                "What book or movie has influenced you the most?",
                "How do you handle stress and pressure?",
                "What's something you're passionate about?",
                "Describe a person who has inspired you.",
                "What would you do with a free day?",
                "Share your thoughts on work-life balance.",
                "What's your favorite way to learn new things?",
                "Describe a goal you're working towards."
            ] * 7  # Repeat to get 105 prompts
            
            with open(prompts_file, 'w') as f:
                for prompt in default_prompts[:100]:
                    f.write(prompt + '\n')
        
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    
    def run_benchmark(self, num_prompts=50):
        """Run inference benchmark"""
        print(f"🎯 Starting inference benchmark with {num_prompts} prompts...")
        
        results = {
            'model_info': {
                'model_path': self.model_path,
                'model_type': self.model_type,
                'quantization': self.quantization,
                'timestamp': datetime.now().isoformat()
            },
            'individual_results': [],
            'summary': {}
        }
        
        # Load test prompts
        test_prompts = self.load_test_prompts()
        prompts_to_test = test_prompts[:num_prompts]
        
        total_latency = 0
        total_tokens = 0
        total_throughput = 0
        latencies = []
        throughputs = []
        
        for i, prompt in enumerate(prompts_to_test, 1):
            print(f"📝 Processing prompt {i}/{num_prompts}...")
            
            try:
                result = self.generate_text(prompt)
                
                # Store individual result
                individual_result = {
                    'prompt_id': i,
                    'prompt': prompt,
                    'latency': result['latency'],
                    'tokens_generated': result['tokens_generated'],
                    'throughput': result['throughput'],
                    'response_length': len(result['response']),
                    'input_tokens': result.get('input_tokens', 0),
                    'total_tokens': result.get('total_tokens', 0)
                }
                results['individual_results'].append(individual_result)
                
                # Accumulate metrics
                total_latency += result['latency']
                total_tokens += result['tokens_generated']
                total_throughput += result['throughput']
                latencies.append(result['latency'])
                throughputs.append(result['throughput'])
                
                print(f"  ⏱️ Latency: {result['latency']:.3f}s | 🔢 Tokens: {result['tokens_generated']} | 🚀 Throughput: {result['throughput']:.2f} tok/s")
                
            except Exception as e:
                print(f"❌ Error processing prompt {i}: {e}")
                continue
        
        # Calculate summary statistics
        num_successful = len(results['individual_results'])
        if num_successful > 0:
            avg_latency = total_latency / num_successful
            avg_tokens = total_tokens / num_successful
            avg_throughput = total_throughput / num_successful
            
            # Calculate percentiles
            latencies.sort()
            throughputs.sort()
            
            results['summary'] = {
                'num_prompts': num_successful,
                'average_latency': avg_latency,
                'median_latency': latencies[len(latencies)//2] if latencies else 0,
                'p95_latency': latencies[int(len(latencies)*0.95)] if latencies else 0,
                'average_tokens_generated': avg_tokens,
                'average_throughput': avg_throughput,
                'median_throughput': throughputs[len(throughputs)//2] if throughputs else 0,
                'total_latency': total_latency,
                'total_tokens': total_tokens,
                'successful_prompts': num_successful,
                'failed_prompts': num_prompts - num_successful
            }
        
        return results
    
    def save_results(self, results, output_dir):
        """Save benchmark results to specified directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename based on model info
        model_name = Path(self.model_path).name.replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"inference_benchmark_{model_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        return filename

def get_model_configs():
    """Get predefined model configurations"""
    return {
        "llama-3.1-8b-bnb-4bit": {
            "path": "unsloth/llama-3.1-8b-bnb-4bit",
            "type": "transformers",
            "quantization": "4bit"
        },
        "llama-3.1-8b-instruct-bf16": {
            "path": "meta-llama/Llama-3.1-8B-Instruct",
            "type": "transformers",
            "quantization": "bf16"
        },
        "llama-3.1-bf16-gguf": {
            "path": "/home/ubuntu/mem0-assignment/model_cache/finetuned_gguf/unsloth.BF16.gguf",
            "type": "gguf",
            "quantization": "bf16"
        },
        "llama-3.1-q4km-gguf": {
            "path": "/home/ubuntu/mem0-assignment/model_cache/finetuned_gguf/unsloth.Q4_K_M.gguf",
            "type": "gguf", 
            "quantization": "4bit"
        },
        "llama-4-scout": {
            "path": "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct",
            "type": "transformers",
            "quantization": None
        },
        "llama-4-scout-bf16": {
            "path": "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--meta-llama--Llama-4-Scout-17B-16E-Instruct",
            "type": "transformers",
            "quantization": "bf16"
        },
        "llama-4-scout-4bit-unsloth": {
            "path": "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--unsloth--Llama-4-Scout-17B-16E-unsloth-bnb-4bit",
            "type": "transformers",
            "quantization": "4bit"
        },
        "llama-4-scout-gguf-q4km": {
            "path": "/home/ubuntu/mem0-assignment/model_cache/scout_gguf/Q4_K_M",
            "type": "gguf",
            "quantization": "4bit"
        }
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Unified Inference Benchmark")
    
    # Model selection - either predefined or custom
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, 
                           choices=list(get_model_configs().keys()),
                           help="Predefined model to benchmark")
    model_group.add_argument("--model-path", type=str,
                           help="Custom path to model (local path or HuggingFace repo)")
    
    # Model configuration (only used with --model-path)
    parser.add_argument("--model-type", type=str, 
                       choices=["transformers", "gguf", "local"],
                       default="transformers",
                       help="Type of model to load (only used with --model-path)")
    parser.add_argument("--quantization", type=str,
                       choices=["4bit", "8bit", "bf16", None],
                       help="Quantization method (only used with --model-path)")
    
    # Benchmark configuration
    parser.add_argument("--num-prompts", type=int, default=50,
                       help="Number of prompts to benchmark")
    parser.add_argument("--output-dir", type=str,
                       default="/home/ubuntu/mem0-assignment/benchmarks",
                       help="Output directory for results")
    
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
            quantization = config["quantization"]
            print(f"🎯 Using predefined model: {args.model}")
        else:
            # Use custom model configuration
            model_path = args.model_path
            model_type = args.model_type
            quantization = args.quantization
            print(f"🎯 Using custom model: {model_path}")
        
        # Set output directory based on model
        if args.model in ["llama-4-scout", "llama-4-scout-bf16", "llama-4-scout-4bit-unsloth"]:
            if args.model == "llama-4-scout-4bit-unsloth":
                output_dir = "/home/ubuntu/mem0-assignment/benchmarks/scout/base_model_results_4bit"
            else:
                output_dir = "/home/ubuntu/mem0-assignment/benchmarks/scout/base_model_results"
        elif args.model == "llama-3.1-8b-instruct-bf16":
            output_dir = "/home/ubuntu/mem0-assignment/benchmarks/base_model_results_bf16"
        else:
            output_dir = args.output_dir
        
        # Initialize benchmark
        benchmark = UnifiedInferenceBenchmark(
            model_path=model_path,
            model_type=model_type,
            quantization=quantization
        )
        
        # Run benchmark
        results = benchmark.run_benchmark(num_prompts=args.num_prompts)
        
        # Save results
        results_file = benchmark.save_results(results, output_dir)
        
        # Print summary
        if 'summary' in results and results['summary']:
            summary = results['summary']
            print("\n" + "="*60)
            print("🎯 INFERENCE BENCHMARK SUMMARY")
            print("="*60)
            print(f"Model: {model_path}")
            print(f"Type: {model_type} | Quantization: {quantization or 'None'}")
            print(f"Successful prompts: {summary['successful_prompts']}/{args.num_prompts}")
            print(f"Average latency: {summary['average_latency']:.3f}s")
            print(f"Median latency: {summary['median_latency']:.3f}s")
            print(f"P95 latency: {summary['p95_latency']:.3f}s")
            print(f"Average tokens generated: {summary['average_tokens_generated']:.1f}")
            print(f"Average throughput: {summary['average_throughput']:.2f} tokens/second")
            print(f"Median throughput: {summary['median_throughput']:.2f} tokens/second")
            print("="*60)
        
        return results_file
        
    except Exception as e:
        print(f"❌ Benchmark failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
