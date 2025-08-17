"""
Advanced Fine-tuning Script with Hyperparameter Tuning and VRAM Monitoring
Based on Unsloth documentation: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide

Features:
- Automated hyperparameter tuning (LoRA rank, batch size, learning rate)
- Real-time VRAM monitoring and optimization
- Dynamic memory management
- Comprehensive training metrics and evaluation
- Integration with memory dataset
"""

import torch
import gc
import psutil
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import Dataset
import numpy as np

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


class VRAMMonitor:
    """Monitor and manage VRAM usage during training."""
    
    def __init__(self):
        self.initial_memory = self.get_gpu_memory()
        self.peak_memory = 0
        self.memory_history = []
    
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
            return {
                "allocated": memory_allocated,
                "reserved": memory_reserved, 
                "free": memory_free,
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get system RAM usage in GB."""
        memory = psutil.virtual_memory()
        return {
            "used": memory.used / 1024**3,
            "available": memory.available / 1024**3,
            "percent": memory.percent,
            "total": memory.total / 1024**3
        }
    
    def log_memory_usage(self, stage: str = ""):
        """Log current memory usage."""
        gpu_mem = self.get_gpu_memory()
        sys_mem = self.get_system_memory()
        
        self.peak_memory = max(self.peak_memory, gpu_mem["allocated"])
        self.memory_history.append({
            "stage": stage,
            "timestamp": time.time(),
            "gpu": gpu_mem,
            "system": sys_mem
        })
        
        print(f"\n=== Memory Usage {stage} ===")
        print(f"GPU: {gpu_mem['allocated']:.2f}GB / {gpu_mem['total']:.2f}GB ({gpu_mem['allocated']/gpu_mem['total']*100:.1f}%)")
        print(f"System RAM: {sys_mem['used']:.2f}GB / {sys_mem['total']:.2f}GB ({sys_mem['percent']:.1f}%)")
        
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 Cleared GPU cache and garbage collection")


class HyperparameterTuner:
    """Automated hyperparameter tuning for LoRA fine-tuning."""
    
    def __init__(self, vram_monitor: VRAMMonitor):
        self.vram_monitor = vram_monitor
        self.results = []
        
        # Hyperparameter search spaces based on Unsloth recommendations
        self.search_space = {
            "lora_rank": [8, 16, 32, 64],  # LoRA rank values
            "lora_alpha": [8, 16, 32, 64],  # LoRA alpha values  
            "batch_size": [1, 2, 4],  # Per device batch size
            "gradient_accumulation": [2, 4, 8],  # Gradient accumulation steps
            "learning_rate": [1e-4, 2e-4, 5e-4, 1e-3],  # Learning rates
            "max_seq_length": [512, 1024, 2048]  # Sequence lengths
        }
    
    def estimate_memory_usage(self, config: Dict) -> float:
        """Estimate VRAM usage for given configuration."""
        # Rough estimation based on empirical data
        base_memory = 3.0  # Base model memory in GB
        
        # Memory scaling factors
        rank_factor = config["lora_rank"] / 16  # Relative to rank 16
        batch_factor = config["batch_size"] / 2  # Relative to batch size 2
        seq_factor = config["max_seq_length"] / 2048  # Relative to 2048
        
        estimated_memory = base_memory * rank_factor * batch_factor * seq_factor
        return estimated_memory
    
    def is_config_feasible(self, config: Dict) -> bool:
        """Check if configuration is feasible given available VRAM."""
        estimated_memory = self.estimate_memory_usage(config)
        available_memory = self.vram_monitor.get_gpu_memory()["free"]
        
        # Leave 1GB buffer for safety
        return estimated_memory < (available_memory - 1.0)
    
    def generate_configs(self, max_configs: int = 10) -> List[Dict]:
        """Generate feasible hyperparameter configurations."""
        configs = []
        
        # Start with conservative configurations and increase complexity
        base_configs = [
            # Conservative config for limited VRAM
            {"lora_rank": 8, "lora_alpha": 16, "batch_size": 1, "gradient_accumulation": 8, 
             "learning_rate": 2e-4, "max_seq_length": 1024},
            
            # Balanced config 
            {"lora_rank": 16, "lora_alpha": 16, "batch_size": 2, "gradient_accumulation": 4,
             "learning_rate": 2e-4, "max_seq_length": 2048},
            
            # Higher rank config for better performance
            {"lora_rank": 32, "lora_alpha": 32, "batch_size": 1, "gradient_accumulation": 4,
             "learning_rate": 1e-4, "max_seq_length": 1024},
        ]
        
        # Filter feasible configurations
        for config in base_configs:
            if self.is_config_feasible(config) and len(configs) < max_configs:
                configs.append(config)
        
        print(f"Generated {len(configs)} feasible configurations")
        return configs


class AdvancedMemoryTrainer:
    """Advanced training pipeline with memory monitoring and hyperparameter tuning."""
    
    def __init__(self, model_name: str = "unsloth/llama-3.1-8b-bnb-4bit"):
        self.model_name = model_name
        self.vram_monitor = VRAMMonitor()
        self.hyperparameter_tuner = HyperparameterTuner(self.vram_monitor)
        self.model = None
        self.tokenizer = None
        self.best_config = None
        self.best_loss = float('inf')
        
    def load_dataset(self, dataset_path: str) -> Dataset:
        """Load and prepare the memory dataset."""
        print(f"Loading dataset from {dataset_path}")
        
        # Load JSONL dataset
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        
        print(f"Dataset loaded: {len(dataset)} examples")
        print(f"Sample example: {dataset[0]['text'][:100]}...")
        
        return dataset
    
    def load_model_with_config(self, config: Dict):
        """Load model with specific configuration."""
        self.vram_monitor.log_memory_usage("Before Model Loading")
        
        # Clear any existing model
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.vram_monitor.clear_cache()
        
        print(f"\n🚀 Loading model with config: {config}")
        
        # Load model with configuration
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=config["max_seq_length"],
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Always use 4-bit for memory efficiency
        )
        
        # Setup LoRA with configuration
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=config["lora_rank"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=0,  # Optimized for Unsloth
            bias="none",    # Optimized for Unsloth
            use_gradient_checkpointing="unsloth",  # Memory efficient
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        self.vram_monitor.log_memory_usage("After Model Loading")
        
    def create_training_args(self, config: Dict, output_dir: str) -> TrainingArguments:
        """Create training arguments based on configuration."""
        return TrainingArguments(
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation"],
            warmup_steps=5,
            max_steps=60,  # Quick training for hyperparameter testing
            learning_rate=config["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",  # Memory efficient optimizer
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_strategy="no",  # Don't save intermediate checkpoints during tuning
            evaluation_strategy="no",
            report_to="none",  # No external logging during tuning
            dataloader_pin_memory=False,  # Reduce memory usage
            remove_unused_columns=False,
        )
    
    def train_with_config(self, config: Dict, dataset: Dataset, trial_name: str) -> Dict:
        """Train model with specific configuration and return metrics."""
        print(f"\n{'='*60}")
        print(f"🏋️ Training Trial: {trial_name}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        try:
            # Load model with this configuration
            self.load_model_with_config(config)
            
            # Create training arguments
            output_dir = f"./training_trials/{trial_name}"
            training_args = self.create_training_args(config, output_dir)
            
            # Create trainer
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=config["max_seq_length"],
                dataset_num_proc=2,
                args=training_args,
            )
            
            self.vram_monitor.log_memory_usage("Before Training")
            
            # Train model
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            self.vram_monitor.log_memory_usage("After Training")
            
            # Extract metrics
            final_loss = train_result.training_loss
            
            # Prepare results
            result = {
                "config": config,
                "final_loss": final_loss,
                "training_time": training_time,
                "peak_memory": self.vram_monitor.peak_memory,
                "trial_name": trial_name
            }
            
            print(f"\n✅ Trial {trial_name} Complete!")
            print(f"Final Loss: {final_loss:.4f}")
            print(f"Training Time: {training_time:.2f}s")
            print(f"Peak Memory: {self.vram_monitor.peak_memory:.2f}GB")
            
            # Update best configuration
            if final_loss < self.best_loss:
                self.best_loss = final_loss
                self.best_config = config.copy()
                print(f"🏆 New best configuration found!")
            
            return result
            
        except Exception as e:
            print(f"❌ Trial {trial_name} failed: {str(e)}")
            return {
                "config": config,
                "final_loss": float('inf'),
                "training_time": 0,
                "peak_memory": 0,
                "trial_name": trial_name,
                "error": str(e)
            }
        finally:
            # Clean up memory
            self.vram_monitor.clear_cache()
    
    def run_hyperparameter_tuning(self, dataset_path: str, max_trials: int = 5) -> Dict:
        """Run hyperparameter tuning with multiple configurations."""
        print("🔍 Starting Hyperparameter Tuning")
        print(f"Dataset: {dataset_path}")
        print(f"Max trials: {max_trials}")
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Generate configurations
        configs = self.hyperparameter_tuner.generate_configs(max_trials)
        
        if not configs:
            raise RuntimeError("No feasible configurations found for available VRAM")
        
        # Run trials
        results = []
        for i, config in enumerate(configs):
            trial_name = f"trial_{i+1:02d}"
            result = self.train_with_config(config, dataset, trial_name)
            results.append(result)
            
            # Save intermediate results
            self.save_tuning_results(results, "intermediate_results.json")
        
        # Final analysis
        self.analyze_results(results)
        return {
            "best_config": self.best_config,
            "best_loss": self.best_loss,
            "all_results": results
        }
    
    def analyze_results(self, results: List[Dict]):
        """Analyze hyperparameter tuning results."""
        print(f"\n{'='*80}")
        print("📊 HYPERPARAMETER TUNING RESULTS")
        print(f"{'='*80}")
        
        # Sort by loss
        valid_results = [r for r in results if r["final_loss"] != float('inf')]
        valid_results.sort(key=lambda x: x["final_loss"])
        
        print(f"\n🏆 Best Configuration:")
        if valid_results:
            best = valid_results[0]
            print(f"  Loss: {best['final_loss']:.4f}")
            print(f"  Config: {best['config']}")
            print(f"  Memory: {best['peak_memory']:.2f}GB")
            print(f"  Time: {best['training_time']:.2f}s")
        
        print(f"\n📈 All Results (sorted by loss):")
        for i, result in enumerate(valid_results):
            print(f"  {i+1}. Loss: {result['final_loss']:.4f} | "
                  f"Rank: {result['config']['lora_rank']} | "
                  f"Batch: {result['config']['batch_size']} | "
                  f"Memory: {result['peak_memory']:.2f}GB")
    
    def save_tuning_results(self, results: List[Dict], filename: str):
        """Save tuning results to file."""
        output_path = Path("./training_trials") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {output_path}")
    
    def export_model(self, save_path: str, export_formats: List[str]):
        """Export model in specified formats for deployment."""
        print(f"\n📦 Exporting model in formats: {export_formats}")
        
        for export_format in export_formats:
            if export_format == "gguf":
                self.export_gguf(save_path)
            elif export_format == "vllm":
                self.export_vllm(save_path)
            else:
                print(f"⚠️ Unknown export format: {export_format}")
    
    def export_gguf(self, save_path: str):
        """Export model in GGUF format for llama.cpp and similar tools."""
        try:
            print("🔄 Exporting to GGUF format...")
            gguf_path = f"{save_path}_gguf"
            
            # Unsloth provides GGUF export functionality
            self.model.save_pretrained_gguf(
                gguf_path, 
                self.tokenizer,
                quantization_method="q4_k_m"  # Efficient 4-bit quantization
            )
            
            print(f"✅ GGUF model exported to {gguf_path}")
            
        except Exception as e:
            print(f"❌ GGUF export failed: {str(e)}")
    
    def export_vllm(self, save_path: str):
        """Export model in vLLM-compatible format."""
        try:
            print("🔄 Exporting to vLLM-compatible format...")
            vllm_path = f"{save_path}_vllm"
            
            # vLLM can work with standard HuggingFace format
            # We merge the LoRA weights for vLLM compatibility
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(vllm_path)
            self.tokenizer.save_pretrained(vllm_path)
            
            print(f"✅ vLLM-compatible model exported to {vllm_path}")
            
        except Exception as e:
            print(f"❌ vLLM export failed: {str(e)}")

    def train_final_model(self, dataset_path: str, config: Optional[Dict] = None, 
                         num_epochs: int = 3, save_path: str = "./final_memory_model",
                         export_formats: Optional[List[str]] = None):
        """Train final model with best configuration for full epochs."""
        if config is None:
            if self.best_config is None:
                raise ValueError("No best configuration found. Run hyperparameter tuning first.")
            config = self.best_config
        
        print(f"\n🎯 Training Final Model")
        print(f"Configuration: {config}")
        print(f"Epochs: {num_epochs}")
        
        # Load dataset and model
        dataset = self.load_dataset(dataset_path)
        self.load_model_with_config(config)
        
        # Create training arguments for full training
        training_args = TrainingArguments(
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation"],
            warmup_steps=10,
            num_train_epochs=num_epochs,  # Full training
            learning_rate=config["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=save_path,
            save_strategy="epoch",
            evaluation_strategy="no",
            report_to="none",
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config["max_seq_length"],
            dataset_num_proc=2,
            args=training_args,
        )
        
        # Train
        self.vram_monitor.log_memory_usage("Before Final Training")
        print("🚀 Starting final training...")
        
        train_result = trainer.train()
        
        self.vram_monitor.log_memory_usage("After Final Training")
        
        # Save model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"✅ Final model saved to {save_path}")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Export in requested formats
        if export_formats:
            self.export_model(save_path, export_formats)
        
        return train_result


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Memory Fine-tuning with Export Options")
    parser.add_argument("--dataset-path", type=str, 
                       default="/home/ubuntu/mem0-assignment/finetune/memory_dataset.jsonl",
                       help="Path to the training dataset")
    parser.add_argument("--max-trials", type=int, default=3,
                       help="Maximum number of hyperparameter tuning trials")
    parser.add_argument("--num-epochs", type=int, default=1,
                       help="Number of epochs for final training")
    parser.add_argument("--save-path", type=str, default="./memory_model_final",
                       help="Path to save the final model")
    parser.add_argument("--export-formats", nargs="*", choices=["gguf", "vllm"],
                       help="Export formats for deployment (gguf, vllm)")
    parser.add_argument("--skip-tuning", action="store_true",
                       help="Skip hyperparameter tuning and use default config")
    return parser.parse_args()


def main():
    """Main training script with hyperparameter tuning."""
    args = parse_args()
    
    print("🚀 Advanced Memory Fine-tuning with Hyperparameter Tuning")
    print("Based on Unsloth documentation and best practices")
    print(f"Dataset: {args.dataset_path}")
    print(f"Export formats: {args.export_formats or 'None'}")
    
    # Initialize trainer
    trainer = AdvancedMemoryTrainer()
    
    if not args.skip_tuning:
        print("Phase 1: Hyperparameter Tuning")
        tuning_results = trainer.run_hyperparameter_tuning(
            dataset_path=args.dataset_path,
            max_trials=args.max_trials
        )
        print(f"Best configuration: {tuning_results['best_config']}")
        print(f"Best loss: {tuning_results['best_loss']:.4f}")
    else:
        print("Skipping hyperparameter tuning - using default configuration")
    
    print("\nPhase 2: Final Model Training")
    final_result = trainer.train_final_model(
        dataset_path=args.dataset_path,
        num_epochs=args.num_epochs,
        save_path=args.save_path,
        export_formats=args.export_formats
    )
    
    print("\n🎉 Training Pipeline Complete!")
    if args.export_formats:
        print(f"Model exported in formats: {args.export_formats}")


if __name__ == "__main__":
    main()
