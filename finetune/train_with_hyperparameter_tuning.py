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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
# Note: HuggingFace upload functionality moved to upload_to_hf.py
import os


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
    
    def __init__(self, vram_monitor: VRAMMonitor, model_type: str = "llama3.1"):
        self.vram_monitor = vram_monitor
        self.model_type = model_type
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
        
        # Get model-appropriate configurations based on model type
        if 'scout' in str(self.model_type).lower():
            # More conservative configs for Scout (109B parameters)
            base_configs = [
                # Ultra conservative for Scout
                {"lora_rank": 4, "lora_alpha": 8, "batch_size": 1, "gradient_accumulation": 16, 
                 "learning_rate": 1e-4, "max_seq_length": 512},
                
                # Conservative config for Scout
                {"lora_rank": 8, "lora_alpha": 16, "batch_size": 1, "gradient_accumulation": 8, 
                 "learning_rate": 5e-5, "max_seq_length": 1024},
                
                # Moderate config for Scout
                {"lora_rank": 16, "lora_alpha": 16, "batch_size": 1, "gradient_accumulation": 4,
                 "learning_rate": 2e-5, "max_seq_length": 1024},
            ]
        else:
            # Standard configs for Llama 3.1 8B
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


class TrainingPlotter:
    """Generate comprehensive plots for training metrics and analysis."""
    
    def __init__(self, output_dir: str = "./training_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_hyperparameter_results(self, results: List[Dict], save_name: str = "hyperparameter_comparison"):
        """Create comprehensive hyperparameter tuning visualization."""
        valid_results = [r for r in results if r["final_loss"] != float('inf')]
        if not valid_results:
            print("⚠️ No valid results to plot")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
        
        # Extract data
        configs = [r['config'] for r in valid_results]
        losses = [r['final_loss'] for r in valid_results]
        memories = [r['peak_memory'] for r in valid_results]
        times = [r['training_time'] for r in valid_results]
        
        # 1. Loss vs LoRA Rank
        ranks = [c['lora_rank'] for c in configs]
        axes[0,0].scatter(ranks, losses, s=100, alpha=0.7)
        axes[0,0].set_xlabel('LoRA Rank')
        axes[0,0].set_ylabel('Final Loss')
        axes[0,0].set_title('Loss vs LoRA Rank')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Loss vs Batch Size
        batch_sizes = [c['batch_size'] for c in configs]
        axes[0,1].scatter(batch_sizes, losses, s=100, alpha=0.7, color='orange')
        axes[0,1].set_xlabel('Batch Size')
        axes[0,1].set_ylabel('Final Loss')
        axes[0,1].set_title('Loss vs Batch Size')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Loss vs Learning Rate
        lrs = [c['learning_rate'] for c in configs]
        axes[0,2].scatter(lrs, losses, s=100, alpha=0.7, color='green')
        axes[0,2].set_xlabel('Learning Rate')
        axes[0,2].set_ylabel('Final Loss')
        axes[0,2].set_title('Loss vs Learning Rate')
        axes[0,2].set_xscale('log')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Memory Usage vs Performance
        axes[1,0].scatter(memories, losses, s=100, alpha=0.7, color='red')
        axes[1,0].set_xlabel('Peak Memory (GB)')
        axes[1,0].set_ylabel('Final Loss')
        axes[1,0].set_title('Memory vs Performance Trade-off')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Training Time vs Performance
        axes[1,1].scatter(times, losses, s=100, alpha=0.7, color='purple')
        axes[1,1].set_xlabel('Training Time (s)')
        axes[1,1].set_ylabel('Final Loss')
        axes[1,1].set_title('Time vs Performance Trade-off')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Configuration Rankings
        trial_names = [r['trial_name'] for r in valid_results]
        colors = sns.color_palette("RdYlGn_r", len(valid_results))
        bars = axes[1,2].bar(range(len(trial_names)), losses, color=colors)
        axes[1,2].set_xlabel('Trial')
        axes[1,2].set_ylabel('Final Loss')
        axes[1,2].set_title('Trial Comparison (Lower is Better)')
        axes[1,2].set_xticks(range(len(trial_names)))
        axes[1,2].set_xticklabels([t.replace('trial_', '') for t in trial_names])
        axes[1,2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{loss:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')  # Also save as PDF
        print(f"📊 Hyperparameter plots saved to {save_path}")
        plt.close()
        
    def plot_memory_usage_timeline(self, memory_history: List[Dict], save_name: str = "memory_timeline"):
        """Plot memory usage over time during training."""
        if not memory_history:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Memory Usage Timeline', fontsize=16, fontweight='bold')
        
        # Extract timeline data
        timestamps = [(h['timestamp'] - memory_history[0]['timestamp'])/60 for h in memory_history]  # Convert to minutes
        gpu_allocated = [h['gpu']['allocated'] for h in memory_history]
        gpu_total = memory_history[0]['gpu']['total']
        sys_used = [h['system']['used'] for h in memory_history]
        sys_total = memory_history[0]['system']['total']
        stages = [h['stage'] for h in memory_history]
        
        # GPU Memory Plot
        ax1.plot(timestamps, gpu_allocated, 'b-', linewidth=2, label='GPU Allocated')
        ax1.axhline(y=gpu_total, color='r', linestyle='--', alpha=0.7, label=f'GPU Total ({gpu_total:.1f}GB)')
        ax1.fill_between(timestamps, gpu_allocated, alpha=0.3)
        ax1.set_ylabel('GPU Memory (GB)')
        ax1.set_title('GPU Memory Usage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # System Memory Plot  
        ax2.plot(timestamps, sys_used, 'g-', linewidth=2, label='System RAM Used')
        ax2.axhline(y=sys_total, color='r', linestyle='--', alpha=0.7, label=f'System Total ({sys_total:.1f}GB)')
        ax2.fill_between(timestamps, sys_used, alpha=0.3, color='green')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('System RAM (GB)')
        ax2.set_title('System Memory Usage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add stage annotations
        for i, (time, stage) in enumerate(zip(timestamps, stages)):
            if stage and i % 3 == 0:  # Avoid overcrowding
                ax1.annotate(stage, (time, gpu_allocated[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8, rotation=45)
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Memory timeline plot saved to {save_path}")
        plt.close()
        
    def plot_training_loss_curve(self, trainer_logs: List[Dict], save_name: str = "training_loss"):
        """Plot training loss curve if available."""
        if not trainer_logs:
            print("⚠️ No training logs available for loss curve")
            return
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        steps = [log.get('step', i) for i, log in enumerate(trainer_logs)]
        losses = [log.get('train_loss', log.get('loss', 0)) for log in trainer_logs]
        
        ax.plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(steps) > 1:
            z = np.polyfit(steps, losses, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), "r--", alpha=0.8, linewidth=1, label=f'Trend (slope: {z[0]:.4f})')
            ax.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training loss curve saved to {save_path}")
        plt.close()
        
    def create_summary_report(self, results: List[Dict], best_config: Dict, memory_history: List[Dict]):
        """Generate a comprehensive summary report with key metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Summary Report', fontsize=18, fontweight='bold')
        
        valid_results = [r for r in results if r["final_loss"] != float('inf')]
        
        if valid_results:
            # 1. Trial Performance Comparison
            trial_names = [r['trial_name'] for r in valid_results]
            losses = [r['final_loss'] for r in valid_results]
            colors = ['gold' if r['config'] == best_config else 'lightblue' for r in valid_results]
            
            bars = ax1.bar(range(len(trial_names)), losses, color=colors)
            ax1.set_title('Trial Performance (Gold = Best)')
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Final Loss')
            ax1.set_xticks(range(len(trial_names)))
            ax1.set_xticklabels([t.replace('trial_', '') for t in trial_names])
            
            for bar, loss in zip(bars, losses):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{loss:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 2. Resource Usage Distribution
            memories = [r['peak_memory'] for r in valid_results]
            times = [r['training_time'] for r in valid_results]
            
            ax2.scatter(memories, times, s=100, alpha=0.7, c=losses, cmap='RdYlBu_r')
            ax2.set_xlabel('Peak Memory (GB)')
            ax2.set_ylabel('Training Time (s)')
            ax2.set_title('Resource Usage vs Performance')
            cbar = plt.colorbar(ax2.collections[0], ax=ax2)
            cbar.set_label('Final Loss')
            
            # 3. Best Configuration Visualization
            config_keys = ['lora_rank', 'batch_size', 'learning_rate', 'max_seq_length']
            config_values = [best_config[key] for key in config_keys]
            config_labels = ['LoRA Rank', 'Batch Size', 'Learning Rate', 'Max Seq Length']
            
            # Normalize values for radar chart
            normalized_values = []
            for key, val in zip(config_keys, config_values):
                if key == 'learning_rate':
                    normalized_values.append(val * 10000)  # Scale up for visibility
                else:
                    normalized_values.append(val)
            
            bars = ax3.bar(config_labels, normalized_values, color='lightgreen', alpha=0.7)
            ax3.set_title('Best Configuration Parameters')
            ax3.set_ylabel('Value (Learning Rate × 10000)')
            
            for bar, val, orig_val in zip(bars, normalized_values, config_values):
                height = bar.get_height()
                display_val = f'{orig_val}' if isinstance(orig_val, int) else f'{orig_val:.4f}'
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(normalized_values)*0.01,
                        display_val, ha='center', va='bottom', fontsize=9, rotation=45)
        
        # 4. Memory Usage Summary
        if memory_history:
            peak_gpu = max(h['gpu']['allocated'] for h in memory_history)
            avg_gpu = np.mean([h['gpu']['allocated'] for h in memory_history])
            total_gpu = memory_history[0]['gpu']['total']
            
            memory_data = [avg_gpu, peak_gpu - avg_gpu, total_gpu - peak_gpu]
            memory_labels = ['Average Used', 'Peak Additional', 'Remaining']
            colors = ['orange', 'red', 'lightgray']
            
            wedges, texts, autotexts = ax4.pie(memory_data, labels=memory_labels, colors=colors, 
                                               autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'GPU Memory Utilization\n(Peak: {peak_gpu:.1f}GB / {total_gpu:.1f}GB)')
        
        plt.tight_layout()
        save_path = self.output_dir / "training_summary_report.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Summary report saved to {save_path}")
        plt.close()


class AdvancedMemoryTrainer:
    """Advanced training pipeline with memory monitoring and hyperparameter tuning."""
    
    def __init__(self, model_name: str = "unsloth/llama-3.1-8b-bnb-4bit", enable_plotting: bool = True):
        self.model_name = model_name
        self.vram_monitor = VRAMMonitor()
        
        # Determine model type and configure accordingly first
        self.model_type = self._determine_model_type(model_name)
        self.target_modules = self._get_target_modules()
        
        # Set output prefix based on model type
        self.output_prefix = "scout_" if "scout" in model_name.lower() else ""
        
        self.hyperparameter_tuner = HyperparameterTuner(self.vram_monitor, self.model_type)
        
        # Initialize plotter with custom output directory if using scout model
        if enable_plotting:
            plots_dir = f"./finetune/{self.output_prefix}training_plots" if self.output_prefix else "./training_plots"
            self.plotter = TrainingPlotter(output_dir=plots_dir)
        else:
            self.plotter = None
            
        self.model = None
        self.tokenizer = None
        self.best_config = None
        self.best_loss = float('inf')
        self.training_logs = []
        
    def _determine_model_type(self, model_name: str) -> str:
        """Determine the type of model being used."""
        if "scout" in model_name.lower() or "meta-llama/Llama-4-Scout" in model_name:
            return "scout"
        elif "llama-3.1" in model_name.lower():
            return "llama3.1"
        else:
            return "other"
    
    def _get_target_modules(self) -> List[str]:
        """Get appropriate target modules for LoRA based on model type."""
        if self.model_type == "scout":
            # Scout model has MoE architecture, may need different target modules
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
        else:
            # Standard Llama target modules
            return ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]
    
    @staticmethod
    def get_model_path(model_name: str, custom_path: Optional[str] = None) -> str:
        """Get the appropriate model path based on model name or custom path."""
        if custom_path:
            return custom_path
            
        model_paths = {
            "llama3.1-8b": "unsloth/llama-3.1-8b-bnb-4bit",
            "llama4-scout": "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth"
        }
        
        return model_paths.get(model_name, model_name)
        
    def load_dataset(self, dataset_path: str) -> Dataset:
        """Load and prepare the memory dataset."""
        print(f"Loading dataset from {dataset_path}")
        
        try:
            # Check if file exists
            if not Path(dataset_path).exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            # Load JSONL dataset
            data = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if line:  # Skip empty lines
                            data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Warning: Invalid JSON on line {line_num}: {e}")
                        continue
            
            if not data:
                raise ValueError("No valid data found in dataset file")
            
            # Convert to HuggingFace dataset
            dataset = Dataset.from_list(data)
            
            # Validate dataset structure
            if len(dataset) == 0:
                raise ValueError("Dataset is empty after loading")
            
            required_field = "text"
            if required_field not in dataset.column_names:
                raise ValueError(f"Dataset missing required field: '{required_field}'. Found fields: {dataset.column_names}")
            
            print(f"Dataset loaded: {len(dataset)} examples")
            print(f"Sample example: {dataset[0]['text'][:100]}...")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {str(e)}")
            raise
    
    def load_model_with_config(self, config: Dict):
        """Load model with specific configuration."""
        try:
            self.vram_monitor.log_memory_usage("Before Model Loading")
            
            # Validate configuration
            required_keys = ["max_seq_length", "lora_rank", "lora_alpha"]
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
            
            # Clear any existing model
            if self.model is not None:
                del self.model
                del self.tokenizer
                self.vram_monitor.clear_cache()
            
            print(f"\n🚀 Loading model with config: {config}")
            
            # Check available VRAM before loading
            gpu_memory = self.vram_monitor.get_gpu_memory()
            if gpu_memory["free"] < 2.0:  # Need at least 2GB free
                print(f"⚠️ Warning: Low GPU memory ({gpu_memory['free']:.1f}GB free)")
                self.vram_monitor.clear_cache()
            
            # Load model with configuration using Unsloth
            # Both Llama 3.1 and Scout models should work with FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=config["max_seq_length"],
                dtype=None,  # Auto-detect
                load_in_4bit=True,  # Always use 4-bit for memory efficiency
                trust_remote_code=True,  # Required for Scout model
            )
            
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Failed to load model or tokenizer")
            
            # Setup LoRA with configuration using Unsloth
            # Both models should work with FastLanguageModel.get_peft_model
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=config["lora_rank"],
                target_modules=self.target_modules,
                lora_alpha=config["lora_alpha"],
                lora_dropout=0,  # Optimized for Unsloth
                bias="none",    # Optimized for Unsloth
                use_gradient_checkpointing="unsloth",  # Memory efficient
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
            
            self.vram_monitor.log_memory_usage("After Model Loading")
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA Out of Memory during model loading: {str(e)}")
            print("💡 Try reducing max_seq_length or lora_rank")
            self.vram_monitor.clear_cache()
            raise
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            self.vram_monitor.clear_cache()
            raise
        
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
            eval_strategy="no",
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
            trials_dir = f"./finetune/{self.output_prefix}training_trials" if self.output_prefix else "./training_trials"
            output_dir = f"{trials_dir}/{trial_name}"
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
        try:
            print("🔍 Starting Hyperparameter Tuning")
            print(f"Dataset: {dataset_path}")
            print(f"Max trials: {max_trials}")
            
            # Validate inputs
            if max_trials <= 0:
                raise ValueError("max_trials must be greater than 0")
            
            # Load dataset
            dataset = self.load_dataset(dataset_path)
            
            # Generate configurations
            configs = self.hyperparameter_tuner.generate_configs(max_trials)
            
            if not configs:
                raise RuntimeError("No feasible configurations found for available VRAM")
            
            # Run trials
            results = []
            successful_trials = 0
            
            for i, config in enumerate(configs):
                trial_name = f"trial_{i+1:02d}"
                print(f"\n🔄 Running trial {i+1}/{len(configs)}")
                
                result = self.train_with_config(config, dataset, trial_name)
                results.append(result)
                
                if result["final_loss"] != float('inf'):
                    successful_trials += 1
                
                # Save intermediate results
                try:
                    self.save_tuning_results(results, "intermediate_results.json")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to save intermediate results: {e}")
            
            if successful_trials == 0:
                raise RuntimeError("All hyperparameter tuning trials failed")
            
            print(f"\n✅ Completed {successful_trials}/{len(configs)} trials successfully")
            
            # Final analysis
            self.analyze_results(results)
            
            # Generate plots if plotting is enabled
            if self.plotter:
                try:
                    print("\n📊 Generating training plots...")
                    self.plotter.plot_hyperparameter_results(results)
                    self.plotter.plot_memory_usage_timeline(self.vram_monitor.memory_history)
                    if self.best_config:
                        self.plotter.create_summary_report(results, self.best_config, self.vram_monitor.memory_history)
                except Exception as e:
                    print(f"⚠️ Warning: Failed to generate plots: {e}")
            
            return {
                "best_config": self.best_config,
                "best_loss": self.best_loss,
                "all_results": results,
                "successful_trials": successful_trials
            }
            
        except Exception as e:
            print(f"❌ Hyperparameter tuning failed: {str(e)}")
            raise
    
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
        try:
            trials_dir = f"./finetune/{self.output_prefix}training_trials" if self.output_prefix else "./training_trials"
            output_path = Path(trials_dir) / filename
            output_path.parent.mkdir(exist_ok=True)
            
            # Create backup if file exists
            if output_path.exists():
                backup_path = output_path.with_suffix(f".backup_{int(time.time())}.json")
                output_path.rename(backup_path)
                print(f"📋 Backed up existing results to {backup_path}")
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)  # default=str handles non-serializable objects
            
            print(f"💾 Results saved to {output_path}")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to save results to {filename}: {e}")
            # Try to save to a temporary location
            try:
                temp_path = Path(f"./training_results_backup_{int(time.time())}.json")
                with open(temp_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"💾 Results saved to backup location: {temp_path}")
            except Exception as backup_e:
                print(f"❌ Failed to save backup results: {backup_e}")
    
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
                         export_formats: Optional[List[str]] = None,
                         hf_repo_name: Optional[str] = None, hf_token: Optional[str] = None):
        """Train final model with best configuration for full epochs."""
        try:
            if config is None:
                if self.best_config is None:
                    raise ValueError("No best configuration found. Run hyperparameter tuning first.")
                config = self.best_config
            
            print(f"\n🎯 Training Final Model")
            print(f"Configuration: {config}")
            print(f"Epochs: {num_epochs}")
            
            # Validate parameters
            if num_epochs <= 0:
                raise ValueError("num_epochs must be greater than 0")
            
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
                eval_strategy="no",
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
            try:
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"✅ Final model saved to {save_path}")
            except Exception as e:
                print(f"⚠️ Failed to save model: {str(e)}")
            
            print(f"Final training loss: {train_result.training_loss:.4f}")
            
            # Export in requested formats
            if export_formats:
                try:
                    self.export_model(save_path, export_formats)
                except Exception as e:
                    print(f"⚠️ Export failed: {str(e)}")
            
            # Note: Use upload_to_hf.py script for HuggingFace uploads
            if hf_repo_name:
                print(f"💡 To upload to HuggingFace, run: python upload_to_hf.py --model-path {save_path} --repo-name {hf_repo_name}")
            
            # Generate final training plots
            if self.plotter and self.training_logs:
                try:
                    print("\n📊 Generating final training plots...")
                    self.plotter.plot_training_loss_curve(self.training_logs, "final_training_loss")
                except Exception as e:
                    print(f"⚠️ Failed to generate loss curve: {str(e)}")
            
            return train_result
        
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA Out of Memory during final training: {str(e)}")
            print("💡 Try reducing batch_size or max_seq_length in config")
            self.vram_monitor.clear_cache()
            raise
        except Exception as e:
            print(f"❌ Final training failed: {str(e)}")
            self.vram_monitor.clear_cache()
            raise
        finally:
            self.vram_monitor.clear_cache()


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
    parser.add_argument("--hf-repo-name", type=str,
                       help="Hugging Face repository name (e.g., username/model-name)")
    parser.add_argument("--hf-token", type=str,
                       help="Hugging Face token (or set HF_TOKEN environment variable)")
    parser.add_argument("--disable-plots", action="store_true",
                       help="Disable generation of training plots and visualizations")
    
    # Model selection arguments
    parser.add_argument("--model-name", type=str, 
                       choices=["llama3.1-8b", "llama4-scout"],
                       default="llama3.1-8b",
                       help="Model to fine-tune")
    parser.add_argument("--model-path", type=str,
                       help="Custom model path (overrides --model-name)")
    
    return parser.parse_args()


def main():
    """Main training script with hyperparameter tuning."""
    try:
        args = parse_args()
        
        # Validate arguments
        if args.max_trials <= 0 and not args.skip_tuning:
            raise ValueError("max_trials must be greater than 0")
        if args.num_epochs <= 0:
            raise ValueError("num_epochs must be greater than 0")
        if args.hf_repo_name and not (args.hf_token or os.getenv("HF_TOKEN")):
            print("⚠️ Warning: HF_TOKEN not set for upload")
        
        print("🚀 Advanced Memory Fine-tuning with Hyperparameter Tuning")
        print("Based on Unsloth documentation and best practices")
        print(f"Dataset: {args.dataset_path}")
        print(f"Model: {args.model_name}")
        print(f"Export formats: {args.export_formats or 'None'}")
        
        # Get model path
        model_path = AdvancedMemoryTrainer.get_model_path(args.model_name, args.model_path)
        print(f"Model path: {model_path}")
        
        # Initialize trainer
        trainer = AdvancedMemoryTrainer(model_name=model_path, enable_plotting=not args.disable_plots)
        
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
            export_formats=args.export_formats,
            hf_repo_name=args.hf_repo_name,
            hf_token=args.hf_token
        )
        
        print("\n🎉 Training Pipeline Complete!")
        if args.export_formats:
            print(f"Model exported in formats: {args.export_formats}")
    
    except Exception as e:
        print(f"❌ Training pipeline failed: {str(e)}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
