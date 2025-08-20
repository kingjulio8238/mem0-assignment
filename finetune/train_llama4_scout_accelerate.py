"""
Llama 4 Scout Fine-tuning Script with Accelerate
Handles fine-tuning using pure Transformers + Accelerate for better Llama 4 compatibility
Features:
- Hyperparameter tuning (LoRA rank, batch size, learning rate)
- Real-time VRAM monitoring and optimization
- Dynamic memory management
"""

import torch
import json
import os
import gc
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from tqdm import tqdm
import psutil
import GPUtil
import threading
import sys


@dataclass
class ScoutTrainingConfig:
    """Configuration for Llama 4 Scout training"""
    model_path: str = "./models/llama4-scout"
    dataset_path: str = "./finetune/memory_dataset.jsonl"
    output_dir: str = "./scout_finetuned_model"
    
    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training parameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_seq_length: int = 512
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True
    
    # Export
    export_gguf: bool = True
    gguf_quantization: str = "q4_k_m"


class MemoryMonitor:
    """Monitor GPU and system memory usage with enhanced VRAM tracking"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.memory_history = []
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {}
        
        # System RAM
        memory = psutil.virtual_memory()
        stats['system_ram_used'] = memory.used / (1024**3)  # GB
        stats['system_ram_total'] = memory.total / (1024**3)  # GB
        stats['system_ram_percent'] = memory.percent
        
        # GPU Memory - Enhanced tracking
        if self.gpu_available:
            try:
                # Clear cache for accurate reading
                torch.cuda.empty_cache()
                
                # Get detailed GPU memory info
                stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
                stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)
                stats['gpu_memory_max_allocated'] = torch.cuda.max_memory_allocated() / (1024**3)
                
                # Try GPUtil for total memory
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        stats['gpu_memory_total'] = gpu.memoryTotal / 1024  # GB
                        stats['gpu_memory_percent'] = (stats['gpu_memory_allocated'] / stats['gpu_memory_total']) * 100
                    else:
                        # Fallback estimation
                        stats['gpu_memory_total'] = 80.0  # H100 approximation
                        stats['gpu_memory_percent'] = (stats['gpu_memory_allocated'] / stats['gpu_memory_total']) * 100
                except:
                    stats['gpu_memory_total'] = 80.0  # H100 approximation
                    stats['gpu_memory_percent'] = (stats['gpu_memory_allocated'] / stats['gpu_memory_total']) * 100
                    
            except Exception as e:
                print(f"Warning: Could not get detailed GPU memory stats: {e}")
                stats['gpu_memory_allocated'] = 0
                stats['gpu_memory_total'] = 80.0
        
        # Record in history
        stats['timestamp'] = time.time()
        self.memory_history.append(stats.copy())
        
        return stats
    
    def log_memory_usage(self, stage: str = ""):
        """Log current memory usage with enhanced details"""
        stats = self.get_memory_stats()
        print(f"\n=== VRAM Monitoring {stage} ===")
        print(f"System RAM: {stats.get('system_ram_used', 0):.2f}GB / {stats.get('system_ram_total', 0):.2f}GB ({stats.get('system_ram_percent', 0):.1f}%)")
        
        if self.gpu_available:
            print(f"GPU Allocated: {stats.get('gpu_memory_allocated', 0):.2f}GB")
            print(f"GPU Reserved: {stats.get('gpu_memory_reserved', 0):.2f}GB") 
            print(f"GPU Total: {stats.get('gpu_memory_total', 0):.2f}GB")
            print(f"GPU Usage: {stats.get('gpu_memory_percent', 0):.1f}%")
            print(f"Peak Allocated: {stats.get('gpu_memory_max_allocated', 0):.2f}GB")
        
        print("=" * 50)
    
    def get_available_vram(self) -> float:
        """Get available VRAM in GB"""
        if not self.gpu_available:
            return 0.0
        stats = self.get_memory_stats()
        return stats.get('gpu_memory_total', 80.0) - stats.get('gpu_memory_allocated', 0)
    
    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            gc.collect()
            
    def estimate_memory_for_config(self, lora_rank: int, batch_size: int) -> float:
        """Estimate memory usage for given config"""
        # Base model memory (17B params in 4-bit ≈ 10-12GB)
        base_memory = 12.0
        
        # LoRA memory (roughly proportional to rank)
        lora_memory = (lora_rank / 16) * 2.0  # 2GB for rank 16
        
        # Batch size memory (activations)
        batch_memory = batch_size * 1.5  # 1.5GB per batch
        
        # Safety buffer
        buffer = 3.0
        
        return base_memory + lora_memory + batch_memory + buffer


class ModelLoadingProgress:
    """Show progress indicator during model loading"""
    
    def __init__(self):
        self.loading = False
        self.thread = None
        
    def start(self, message="Loading model"):
        """Start the progress indicator"""
        self.loading = True
        self.thread = threading.Thread(target=self._show_progress, args=(message,))
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the progress indicator"""
        self.loading = False
        if self.thread:
            self.thread.join()
        print()  # New line after progress
        
    def _show_progress(self, message):
        """Show animated progress"""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        idx = 0
        while self.loading:
            sys.stdout.write(f"\r{message} {spinner[idx % len(spinner)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1


class HyperparameterTuner:
    """Hyperparameter tuning with VRAM-aware configuration generation"""
    
    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor
        
    def generate_configs(self, max_trials: int = 5) -> List[Dict]:
        """Generate feasible hyperparameter configurations based on available VRAM"""
        configs = []
        
        # Define search space
        lora_ranks = [4, 8, 16, 32]
        batch_sizes = [1, 2, 4]
        learning_rates = [1e-4, 2e-4, 5e-4, 1e-3]
        gradient_accumulations = [16, 32, 64]
        
        print("🔍 Generating hyperparameter configurations...")
        available_vram = self.memory_monitor.get_available_vram()
        print(f"Available VRAM: {available_vram:.2f}GB")
        
        attempted = 0
        for lora_rank in lora_ranks:
            for batch_size in batch_sizes:
                for lr in learning_rates:
                    for grad_acc in gradient_accumulations:
                        attempted += 1
                        
                        # Estimate memory usage
                        estimated_memory = self.memory_monitor.estimate_memory_for_config(lora_rank, batch_size)
                        
                        if estimated_memory <= available_vram and len(configs) < max_trials:
                            config = {
                                'lora_rank': lora_rank,
                                'lora_alpha': lora_rank * 2,  # Common ratio
                                'batch_size': batch_size,
                                'gradient_accumulation_steps': grad_acc,
                                'learning_rate': lr,
                                'estimated_memory': estimated_memory
                            }
                            configs.append(config)
                            print(f"✅ Config {len(configs)}: Rank={lora_rank}, BS={batch_size}, LR={lr:.1e}, GA={grad_acc}, Est. Mem={estimated_memory:.1f}GB")
                        
                        if len(configs) >= max_trials:
                            break
                    if len(configs) >= max_trials:
                        break
                if len(configs) >= max_trials:
                    break
            if len(configs) >= max_trials:
                break
        
        if not configs:
            # Fallback to minimal config
            print("⚠️ No configs fit in available VRAM, using minimal config")
            configs = [{
                'lora_rank': 4,
                'lora_alpha': 8,
                'batch_size': 1,
                'gradient_accumulation_steps': 16,
                'learning_rate': 2e-4,
                'estimated_memory': 15.0
            }]
        
        print(f"Generated {len(configs)} feasible configurations from {attempted} attempts")
        return configs
    
    def is_config_feasible(self, config: Dict) -> bool:
        """Check if a configuration is feasible given current VRAM"""
        estimated_memory = self.memory_monitor.estimate_memory_for_config(
            config['lora_rank'], 
            config['batch_size']
        )
        available_vram = self.memory_monitor.get_available_vram()
        return estimated_memory <= available_vram


class Llama4ScoutTrainer:
    """Fine-tuning trainer for Llama 4 Scout using Accelerate"""
    
    def __init__(self, config: ScoutTrainingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.tuner = HyperparameterTuner(self.memory_monitor)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=f"{config.output_dir}/logs"
        )
        
        # Set seed for reproducibility
        set_seed(42)
        
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.tuning_results = []
        self.best_config = None
        self.best_loss = float('inf')
        
    def load_dataset(self) -> Dataset:
        """Load and preprocess the training dataset"""
        print(f"📚 Loading dataset from {self.config.dataset_path}")
        
        # Load JSONL dataset
        data_list = []
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line.strip()))
        
        dataset = Dataset.from_list(data_list)
        print(f"Dataset loaded: {len(dataset)} examples")
        
        # Show sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample example: {sample.get('text', str(sample))[:200]}...")
        
        return dataset
    
    def setup_model_and_tokenizer(self):
        """Setup the model and tokenizer with quantization"""
        print(f"🚀 Loading Llama 4 Scout model from {self.config.model_path}")
        
        self.memory_monitor.log_memory_usage("Before Model Loading")
        
        # Setup quantization config
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        else:
            bnb_config = None
        
        # Load tokenizer with progress
        progress = ModelLoadingProgress()
        progress.start("📝 Loading tokenizer")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                use_fast=True
            )
        finally:
            progress.stop()
        
        print("✅ Tokenizer loaded successfully")
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with progress indicator
        progress.start("🦙 Loading Llama 4 Scout model (17B params, this may take 5-10 minutes)")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="eager"  # Use eager attention for Llama 4
            )
        finally:
            progress.stop()
        
        print("✅ Model loaded successfully")
        
        # Prepare model for training
        if self.config.use_4bit:
            progress.start("🔧 Preparing model for 4-bit training")
            try:
                self.model = prepare_model_for_kbit_training(self.model)
            finally:
                progress.stop()
            print("✅ Model prepared for 4-bit training")
        
        self.memory_monitor.log_memory_usage("After Model Loading")
        
    def setup_lora(self):
        """Setup LoRA configuration"""
        progress = ModelLoadingProgress()
        progress.start("🔧 Setting up LoRA configuration")
        
        try:
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, lora_config)
        finally:
            progress.stop()
        
        print("✅ LoRA configuration applied successfully")
        self.model.print_trainable_parameters()
        
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize and preprocess the dataset"""
        print("🔤 Preprocessing dataset...")
        
        def tokenize_function(examples):
            # Handle different input formats
            if 'text' in examples:
                texts = examples['text']
            elif 'messages' in examples:
                # Convert messages to text format for Llama 4
                texts = []
                for messages in examples['messages']:
                    if isinstance(messages, list):
                        # Format as chat template
                        formatted_text = ""
                        for msg in messages:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            formatted_text += f"<|header_start|>{role}<|header_end|>\n\n{content}<|eot|>"
                        texts.append(formatted_text)
                    else:
                        texts.append(str(messages))
            else:
                # Fallback: convert entire example to string
                texts = [str(example) for example in examples]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length,
                return_tensors=None,
            )
            
            # Set labels = input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    def train(self):
        """Main training function"""
        print("🚀 Starting Llama 4 Scout Fine-tuning with Accelerate")
        
        # Load dataset
        self.dataset = self.load_dataset()
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Setup LoRA
        self.setup_lora()
        
        # Preprocess dataset
        tokenized_dataset = self.preprocess_dataset(self.dataset)
        
        # Split dataset (80% train, 20% eval)
        train_size = int(0.8 * len(tokenized_dataset))
        eval_size = len(tokenized_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            tokenized_dataset, [train_size, eval_size]
        )
        
        print(f"Training set: {len(train_dataset)} examples")
        print(f"Evaluation set: {len(eval_dataset)} examples")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="tensorboard",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            bf16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        print("\n🏋️ Starting training...")
        self.memory_monitor.log_memory_usage("Before Training")
        
        train_result = trainer.train()
        
        self.memory_monitor.log_memory_usage("After Training")
        
        # Save the final model
        print("💾 Saving final model...")
        trainer.save_model()
        trainer.save_state()
        
        # Print training summary
        print("\n🎉 Training completed!")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
        print(f"Samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
        
        return trainer
    
    def train_with_config(self, config: Dict, trial_name: str) -> Dict:
        """Train with specific hyperparameter configuration"""
        print(f"\n🔄 Running trial: {trial_name}")
        print(f"Config: {config}")
        
        self.memory_monitor.log_memory_usage(f"Before Trial {trial_name}")
        
        try:
            # Update config for this trial
            self.config.lora_rank = config['lora_rank']
            self.config.lora_alpha = config['lora_alpha']
            self.config.batch_size = config['batch_size']
            self.config.gradient_accumulation_steps = config['gradient_accumulation_steps']
            self.config.learning_rate = config['learning_rate']
            
            # Clear any existing model to free memory
            if self.model is not None:
                del self.model
                self.model = None
                self.memory_monitor.clear_cache()
            
            # Setup model and tokenizer for this config
            self.setup_model_and_tokenizer()
            self.setup_lora()
            
            # Preprocess dataset
            tokenized_dataset = self.preprocess_dataset(self.dataset)
            
            # Split dataset
            train_size = int(0.8 * len(tokenized_dataset))
            eval_size = len(tokenized_dataset) - train_size
            train_dataset, eval_dataset = torch.utils.data.random_split(
                tokenized_dataset, [train_size, eval_size]
            )
            
            # Setup training arguments for this trial
            trial_output_dir = f"{self.config.output_dir}/scout_trials/{trial_name}"
            training_args = TrainingArguments(
                output_dir=trial_output_dir,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                num_train_epochs=1,  # Use 1 epoch for tuning trials
                warmup_steps=50,  # Reduced for trials
                logging_steps=self.config.logging_steps,
                save_steps=1000,  # Less frequent saves for trials
                eval_steps=500,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None,  # Disable reporting for trials
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                bf16=True,
                gradient_checkpointing=True,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Train
            self.memory_monitor.log_memory_usage(f"During Trial {trial_name}")
            train_result = trainer.train()
            
            # Get final evaluation
            eval_result = trainer.evaluate()
            final_loss = eval_result.get('eval_loss', float('inf'))
            
            # Record memory stats
            memory_stats = self.memory_monitor.get_memory_stats()
            peak_memory = memory_stats.get('gpu_memory_max_allocated', 0)
            
            result = {
                'trial_name': trial_name,
                'config': config,
                'final_loss': final_loss,
                'training_loss': train_result.training_loss,
                'peak_memory': peak_memory,
                'runtime': train_result.metrics['train_runtime'],
                'samples_per_second': train_result.metrics['train_samples_per_second']
            }
            
            print(f"✅ Trial {trial_name} completed:")
            print(f"   Final Loss: {final_loss:.4f}")
            print(f"   Peak Memory: {peak_memory:.2f}GB")
            print(f"   Runtime: {result['runtime']:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Trial {trial_name} failed: {str(e)}")
            return {
                'trial_name': trial_name,
                'config': config,
                'final_loss': float('inf'),
                'error': str(e),
                'peak_memory': 0,
                'runtime': 0,
                'samples_per_second': 0
            }
        finally:
            self.memory_monitor.clear_cache()
    
    def run_hyperparameter_tuning(self, max_trials: int = 5) -> Dict:
        """Run hyperparameter tuning"""
        print("🎯 Starting Hyperparameter Tuning Phase")
        print("💡 Note: Model will be loaded fresh for each trial to ensure accurate memory measurements")
        
        # Generate configurations
        configs = self.tuner.generate_configs(max_trials)
        
        if not configs:
            raise RuntimeError("No feasible configurations found for available VRAM")
        
        # Load dataset once (lightweight operation)
        print("📚 Loading dataset for hyperparameter tuning...")
        self.dataset = self.load_dataset()
        
        results = []
        for i, config in enumerate(configs):
            trial_name = f"trial_{i+1:02d}"
            result = self.train_with_config(config, trial_name)
            results.append(result)
            
            # Update best config
            if result['final_loss'] < self.best_loss:
                self.best_loss = result['final_loss']
                self.best_config = config.copy()
                print(f"🏆 New best config found! Loss: {self.best_loss:.4f}")
        
        self.tuning_results = results
        
        # Save results
        self.save_tuning_results(results)
        
        # Print summary
        print("\n📊 Hyperparameter Tuning Results:")
        for result in results:
            print(f"  {result['trial_name']}: Loss={result['final_loss']:.4f}, "
                  f"Memory={result['peak_memory']:.1f}GB, "
                  f"Runtime={result['runtime']:.1f}s")
        
        return {
            'results': results,
            'best_config': self.best_config,
            'best_loss': self.best_loss
        }
    
    def save_tuning_results(self, results: List[Dict]):
        """Save tuning results to file"""
        output_dir = Path(f"{self.config.output_dir}/scout_trials")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "tuning_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Tuning results saved to {results_file}")
    
    def export_gguf(self, trainer):
        """Export model to GGUF format"""
        if not self.config.export_gguf:
            return
            
        print("\n📦 Exporting to GGUF format...")
        try:
            # First save in HF format
            merged_model = trainer.model.merge_and_unload()
            gguf_output_dir = f"{self.config.output_dir}_gguf"
            
            # Save merged model
            merged_model.save_pretrained(gguf_output_dir)
            self.tokenizer.save_pretrained(gguf_output_dir)
            
            print(f"✅ Model saved for GGUF conversion at {gguf_output_dir}")
            print("💡 To convert to GGUF, use:")
            print(f"   python -m llama_cpp.convert {gguf_output_dir}")
            
        except Exception as e:
            print(f"❌ GGUF export failed: {str(e)}")
            print("💡 You can manually convert using llama.cpp tools")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Llama 4 Scout Fine-tuning with Accelerate")
    
    parser.add_argument("--model-path", type=str, default="./models/llama4-scout",
                       help="Path to Llama 4 Scout model")
    parser.add_argument("--dataset-path", type=str, default="./finetune/memory_dataset.jsonl",
                       help="Path to training dataset")
    parser.add_argument("--output-dir", type=str, default="./scout_finetuned_accelerate",
                       help="Output directory for trained model")
    parser.add_argument("--lora-rank", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--max-seq-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--export-gguf", action="store_true",
                       help="Export model to GGUF format")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--max-trials", type=int, default=3,
                       help="Maximum number of hyperparameter tuning trials")
    parser.add_argument("--skip-tuning", action="store_true",
                       help="Skip hyperparameter tuning and use default config")
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Create config
    config = ScoutTrainingConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
        export_gguf=args.export_gguf,
        use_4bit=not args.no_4bit,
    )
    
    print("🦙 Llama 4 Scout Fine-tuning with Accelerate")
    print(f"Model: {config.model_path}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    print(f"LoRA Rank: {config.lora_rank}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"4-bit Quantization: {config.use_4bit}")
    print(f"Skip Tuning: {args.skip_tuning}")
    
    # Create trainer
    trainer_obj = Llama4ScoutTrainer(config)
    
    if not args.skip_tuning:
        print("\n🎯 Phase 1: Hyperparameter Tuning")
        tuning_results = trainer_obj.run_hyperparameter_tuning(args.max_trials)
        print(f"🏆 Best configuration: {tuning_results['best_config']}")
        print(f"🏆 Best loss: {tuning_results['best_loss']:.4f}")
        
        # Update config with best hyperparameters for final training
        best_config = tuning_results['best_config']
        config.lora_rank = best_config['lora_rank']
        config.lora_alpha = best_config['lora_alpha']
        config.batch_size = best_config['batch_size']
        config.gradient_accumulation_steps = best_config['gradient_accumulation_steps']
        config.learning_rate = best_config['learning_rate']
        
        print(f"\n🚀 Phase 2: Final Training with Best Config")
        print(f"Using: Rank={config.lora_rank}, BS={config.batch_size}, LR={config.learning_rate}")
    else:
        print("⏭️ Skipping hyperparameter tuning - using provided configuration")
    
    # Final training with best/provided config
    trainer = trainer_obj.train()
    
    # Export if requested
    if config.export_gguf:
        trainer_obj.export_gguf(trainer)
    
    print("\n🎉 All done! Your Llama 4 Scout model is ready!")
    
    if not args.skip_tuning:
        print(f"💡 Hyperparameter tuning results saved in: {config.output_dir}/scout_trials/")


if __name__ == "__main__":
    main()
