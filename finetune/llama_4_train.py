"""
Llama 4 Scout Fine-tuning Script - Tutorial Implementation
=========================================================

LOGICAL STEPS OVERVIEW:
• Step 1: Environment setup and authentication with Hugging Face
• Step 2: Configuration of training parameters, LoRA settings, and model paths
• Step 3: Model loading with 4-bit quantization and memory optimization
• Step 4: Dataset preparation with prompt formatting and tokenization
• Step 5: LoRA adapter application for parameter-efficient fine-tuning
• Step 6: Training process with SFT trainer and evaluation
• Step 7: Model merging, deployment, and Hugging Face Hub upload

This script follows the exact tutorial approach with comprehensive comments
for educational purposes and production fine-tuning of Llama 4 Scout models.

USAGE:
1. Set your Hugging Face token: export HF_TOKEN=your_token_here
2. Ensure memory_dataset.jsonl is in the same directory
3. Run: python llama_4_train.py

The script will fine-tune Llama 4 Scout on the memory dataset for conversation
memory and context retention tasks.
"""

import os
import torch
import gc
import json
from pathlib import Path
from typing import Dict, Any, Tuple
from io import StringIO

# Core libraries
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
from trl import SFTTrainer
from huggingface_hub import login, upload_file

# =============================================================================
# STEP 1: ENVIRONMENT SETUP AND AUTHENTICATION
# =============================================================================

def setup_environment():
    """
    Setup environment and login to Hugging Face Hub
    
    This function handles authentication with Hugging Face to access
    the Llama 4 model and enable pushing fine-tuned versions.
    """
    print("🔧 Step 1: Setting up environment and authentication...")
    
    # Check for HF_TOKEN environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  Warning: HF_TOKEN environment variable not found")
        print("   Please set your Hugging Face token: export HF_TOKEN=your_token_here")
        return False
    else:
        print("✅ Found HF_TOKEN, logging in to Hugging Face Hub...")
        try:
            login(hf_token)
            print("✅ Successfully authenticated with Hugging Face")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False

# =============================================================================
# STEP 2: CONFIGURATION OVERVIEW
# =============================================================================

class Llama4TrainingConfig:
    """
    Configuration class containing all parameters for fine-tuning
    
    These settings determine everything from which model we'll use
    to how we'll train it, organized into logical groups.
    """
    
    def __init__(self):
        print("⚙️  Step 2: Configuring training parameters...")
        
        # Model and dataset configuration
        self.MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E"
        self.DATASET_PATH = "./memory_dataset.jsonl"  # Local memory dataset
        self.OUTPUT_DIR = "llama4_memory_finetuned"
        
        # Repository config for Hugging Face Hub
        self.REPO_ID = "kingJulio/llama4-scout-memory"  # Replace with your repo
        self.COMMIT_MESSAGE = "Upload fine-tuned Llama-4 Scout model for memory tasks"
        
        # Training parameters
        self.NUM_EPOCHS = 3
        self.BATCH_SIZE = 4
        self.GRADIENT_ACCUMULATION_STEPS = 8
        self.LEARNING_RATE = 3e-4
        self.WEIGHT_DECAY = 0.01
        self.WARMUP_RATIO = 0.1
        self.MAX_GRAD_NORM = 1.0
        self.MAX_SEQ_LENGTH = 512
        self.VALIDATION_SPLIT = 0.1
        
        # Precision settings
        self.BF16 = True
        self.FP16 = False
        self.GRADIENT_CHECKPOINTING = True
        
        # Logging and saving
        self.LOGGING_STEPS = 10
        self.SAVE_STEPS = 500
        
        # LoRA parameters for parameter-efficient fine-tuning
        self.LORA_ALPHA = 32
        self.LORA_DROPOUT = 0.1
        self.LORA_RANK = 64
        self.TARGET_MODULES = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj",     # MLP projections
        ]
        
        print(f"✅ Configuration set:")
        print(f"   Model: {self.MODEL_ID}")
        print(f"   Dataset: {self.DATASET_PATH}")
        print(f"   LoRA Rank: {self.LORA_RANK}")
        print(f"   Batch Size: {self.BATCH_SIZE}")
        print(f"   Learning Rate: {self.LEARNING_RATE}")

# =============================================================================
# STEP 3: LOADING AND QUANTIZING THE MODEL
# =============================================================================

def setup_model_and_tokenizer(config: Llama4TrainingConfig) -> Tuple[Any, Any]:
    """
    Load and configure the model and tokenizer with 4-bit quantization
    
    Loading a 17B parameter model requires careful memory management.
    We use 4-bit quantization to drastically reduce memory footprint
    while preserving model quality.
    """
    print("🦙 Step 3: Loading and quantizing the model...")
    
    # Configure 4-bit quantization for memory efficiency
    print("   Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # Enable 4-bit quantization
        bnb_4bit_use_double_quant=False,      # Disable double quantization
        bnb_4bit_quant_type="nf4",           # Use normal float 4 format
        bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computations
    )
    
    # Load model with quantization and optimization settings
    print(f"   Loading model: {config.MODEL_ID}")
    print("   This may take several minutes...")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        device_map="auto",                    # Automatically distribute across GPUs
        torch_dtype=torch.bfloat16,          # Use bfloat16 precision
        quantization_config=bnb_config,      # Apply 4-bit quantization
        trust_remote_code=True,              # Allow custom model code
    )
    
    # Configure model for training
    print("   Configuring model for training...")
    model.config.use_cache = False           # Disable KV cache during training
    model.config.pretraining_tp = 1          # Set tensor parallelism
    
    # Enable gradient checkpointing for memory efficiency
    if config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("   ✅ Gradient checkpointing enabled")
    
    # Load tokenizer
    print("   Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, 
        trust_remote_code=True
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model and tokenizer loaded successfully")
    
    # Display memory usage
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"   GPU Memory allocated: {allocated_memory:.2f} GB")
    
    return model, tokenizer

# =============================================================================
# STEP 4: PREPARING THE DATASET
# =============================================================================

def load_and_process_data(config: Llama4TrainingConfig, tokenizer) -> Tuple[Dataset, Dataset, Any, Dataset]:
    """
    Load and prepare dataset for fine-tuning
    
    This involves loading the local memory dataset JSONL file, formatting for training,
    tokenization, and creating train/validation splits.
    """
    print("📊 Step 4: Preparing the dataset...")
    
    # Load dataset from local JSONL file
    print(f"   Loading dataset: {config.DATASET_PATH}")
    try:
        # Check if file exists
        if not os.path.exists(config.DATASET_PATH):
            raise FileNotFoundError(f"Dataset file not found: {config.DATASET_PATH}")
        
        # Load JSONL data
        data_list = []
        with open(config.DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data_list.append(json.loads(line))
        
        dataset = Dataset.from_list(data_list)
        print(f"   ✅ Loaded {len(dataset)} examples from local JSONL file")
        
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        print("   📝 Creating a sample dataset for testing...")
        # Create a small sample dataset if loading fails
        sample_data = [
            {"text": "<|im_start|>user\nI love hiking in the mountains.<|im_end|>\n<|im_start|>assistant\nI'll remember your preference for outdoor activities.<|im_end|>"},
            {"text": "<|im_start|>user\nMy birthday is March 15th.<|im_end|>\n<|im_start|>assistant\nI've noted that important date for you.<|im_end|>"},
        ]
        dataset = Dataset.from_list(sample_data)
    
    # Function to format prompts for memory dataset
    def format_prompt(example):
        """
        Format the memory dataset examples for training
        
        The memory dataset already contains properly formatted conversations
        with <|im_start|> and <|im_end|> tags, so we use them directly.
        """
        # The memory dataset already has the full conversation text
        return {"formatted_text": example["text"]}
    
    # Apply formatting to all examples
    print("   Formatting prompts...")
    formatted_dataset = dataset.map(format_prompt)
    
    # Function to tokenize inputs
    def tokenize_function(examples):
        """
        Convert text to numerical tokens for model processing
        
        Sets labels equal to input_ids for causal language modeling,
        teaching the model to predict the next token in sequence.
        """
        model_inputs = tokenizer(
            examples["formatted_text"],
            truncation=True,                    # Truncate long sequences
            padding="max_length",               # Pad to consistent length
            max_length=config.MAX_SEQ_LENGTH,   # Maximum sequence length
            return_tensors="pt",                # Return PyTorch tensors
        )
        # Set labels for causal language modeling
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs
    
    # Apply tokenization with batching for efficiency
    print("   Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=formatted_dataset.column_names
    )
    
    # Split into train and validation sets
    print(f"   Creating train/validation split ({config.VALIDATION_SPLIT:.1%} validation)...")
    tokenized_dataset = tokenized_dataset.train_test_split(
        test_size=config.VALIDATION_SPLIT, 
        seed=42
    )
    
    # Create data collator for dynamic padding during training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # False for causal language modeling
    )
    
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["test"]
    
    print(f"   ✅ Dataset prepared:")
    print(f"      Training examples: {len(train_dataset)}")
    print(f"      Validation examples: {len(val_dataset)}")
    
    return train_dataset, val_dataset, data_collator, dataset

# =============================================================================
# STEP 5: APPLYING PARAMETER-EFFICIENT FINE-TUNING WITH LORA
# =============================================================================

def apply_lora(model, config: Llama4TrainingConfig) -> Tuple[Any, LoraConfig]:
    """
    Apply Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
    
    LoRA modifies only a small subset of parameters, making fine-tuning
    of large models much more efficient while maintaining quality.
    """
    print("🔧 Step 5: Applying LoRA for parameter-efficient fine-tuning...")
    
    # Define LoRA configuration
    print("   Configuring LoRA parameters...")
    peft_config = LoraConfig(
        lora_alpha=config.LORA_ALPHA,        # Scaling factor for LoRA updates
        lora_dropout=config.LORA_DROPOUT,    # Dropout for regularization
        r=config.LORA_RANK,                  # Rank of low-rank matrices
        bias="none",                         # Don't adapt bias parameters
        task_type=TaskType.CAUSAL_LM,        # Causal language modeling task
        target_modules=config.TARGET_MODULES, # Which layers to adapt
    )
    
    print(f"   LoRA Configuration:")
    print(f"      Rank: {config.LORA_RANK}")
    print(f"      Alpha: {config.LORA_ALPHA}")
    print(f"      Dropout: {config.LORA_DROPOUT}")
    print(f"      Target modules: {config.TARGET_MODULES}")
    
    # Apply LoRA to model
    print("   Applying LoRA adapters to model...")
    peft_model = get_peft_model(model, peft_config)
    
    # Calculate and display parameter efficiency
    print("   Analyzing parameter efficiency...")
    trainable_params = 0
    all_params = 0
    
    for name, param in peft_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    efficiency_percentage = (trainable_params / all_params) * 100
    
    print(f"   ✅ LoRA applied successfully:")
    print(f"      Total parameters: {all_params:,}")
    print(f"      Trainable parameters: {trainable_params:,}")
    print(f"      Efficiency: {efficiency_percentage:.2f}% of parameters trainable")
    print(f"      Memory reduction: ~{100 - efficiency_percentage:.1f}%")
    
    return peft_model, peft_config

# =============================================================================
# STEP 6: TRAINING PROCESS
# =============================================================================

def train_model(model, train_dataset, val_dataset, data_collator, peft_config, config: Llama4TrainingConfig):
    """
    Execute the training process using SFTTrainer
    
    Configures training arguments, initializes trainer, and runs
    the supervised fine-tuning with evaluation.
    """
    print("🚀 Step 6: Starting training process...")
    
    # Calculate training steps and warmup
    print("   Calculating training schedule...")
    num_update_steps_per_epoch = max(
        len(train_dataset) // (config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS), 1
    )
    max_train_steps = config.NUM_EPOCHS * num_update_steps_per_epoch
    warmup_steps = int(config.WARMUP_RATIO * max_train_steps)
    
    print(f"   Training schedule:")
    print(f"      Steps per epoch: {num_update_steps_per_epoch}")
    print(f"      Total training steps: {max_train_steps}")
    print(f"      Warmup steps: {warmup_steps}")
    print(f"      Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    
    # Define comprehensive training arguments
    print("   Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,                              # Output directory
        per_device_train_batch_size=config.BATCH_SIZE,             # Batch size per device
        per_device_eval_batch_size=config.BATCH_SIZE,              # Eval batch size
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS, # Gradient accumulation
        learning_rate=config.LEARNING_RATE,                       # Learning rate
        weight_decay=config.WEIGHT_DECAY,                         # L2 regularization
        max_grad_norm=config.MAX_GRAD_NORM,                       # Gradient clipping
        num_train_epochs=config.NUM_EPOCHS,                       # Number of epochs
        logging_steps=config.LOGGING_STEPS,                       # Logging frequency
        save_steps=config.SAVE_STEPS,                             # Checkpoint frequency
        save_total_limit=3,                                       # Keep only 3 checkpoints
        warmup_steps=warmup_steps,                                # Warmup schedule
        fp16=config.FP16,                                         # FP16 precision
        bf16=config.BF16,                                         # BF16 precision
        seed=42,                                                  # Random seed
        gradient_checkpointing=config.GRADIENT_CHECKPOINTING,     # Memory optimization
        evaluation_strategy="steps",                              # Evaluate during training
        eval_steps=config.SAVE_STEPS,                             # Evaluation frequency
        load_best_model_at_end=True,                              # Load best checkpoint
        metric_for_best_model="eval_loss",                        # Metric for best model
        greater_is_better=False,                                  # Lower loss is better
        report_to=None,                                           # Disable wandb/tensorboard
    )
    
    # Initialize SFTTrainer for supervised fine-tuning
    print("   Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,                          # Model with LoRA adapters
        args=training_args,                   # Training configuration
        train_dataset=train_dataset,          # Training data
        eval_dataset=val_dataset,             # Validation data
        peft_config=peft_config,              # LoRA configuration
        data_collator=data_collator,          # Data collation strategy
        max_seq_length=config.MAX_SEQ_LENGTH, # Maximum sequence length
    )
    
    # Execute training
    print("   🏃 Starting training...")
    print(f"   Training for {config.NUM_EPOCHS} epochs...")
    
    try:
        train_results = trainer.train()
        print("   ✅ Training completed successfully!")
        
        # Display training results
        if hasattr(train_results, 'training_loss'):
            print(f"      Final training loss: {train_results.training_loss:.4f}")
        
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        return None, None, None
    
    # Evaluate the trained model
    print("   📊 Evaluating model on validation set...")
    try:
        eval_results = trainer.evaluate()
        print("   ✅ Evaluation completed!")
        
        # Display evaluation results
        print("   📈 Evaluation results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.4f}")
            else:
                print(f"      {key}: {value}")
                
    except Exception as e:
        print(f"   ⚠️  Evaluation failed: {e}")
        eval_results = {}
    
    return trainer, train_results, eval_results

# =============================================================================
# STEP 7: MERGING AND DEPLOYING THE MODEL
# =============================================================================

def save_and_push_to_hub(trainer, tokenizer, config: Llama4TrainingConfig):
    """
    Merge LoRA adapters with base model and deploy to Hugging Face Hub
    
    This creates a single, optimized model ready for deployment and
    uploads it with proper documentation.
    """
    print("🚀 Step 7: Merging adapters and deploying model...")
    
    # Get the base model and prepare for merging
    print("   Preparing model for merging...")
    base_model = trainer.model.get_base_model()
    
    # Create merged model combining LoRA adapters with base model
    print("   🔄 Merging LoRA adapters with base model...")
    print("   This process combines the trained adapters with the original weights...")
    
    try:
        # Load the PEFT model for merging
        merged_model = PeftModel.from_pretrained(
            base_model,
            trainer.model.peft_config,
            is_trainable=False,  # Freeze for inference
        )
        
        # Perform the actual merge - combines LoRA weights with base model
        print("   Executing weight merge...")
        merged_model = merged_model.merge_and_unload()
        
        print("   ✅ Model merging completed!")
        print("   The model no longer requires PEFT library for inference")
        print("   Memory overhead from adapters has been eliminated")
        
    except Exception as e:
        print(f"   ❌ Model merging failed: {e}")
        return False
    
    # Save model locally first
    local_save_path = f"{config.OUTPUT_DIR}_merged"
    print(f"   💾 Saving merged model locally to: {local_save_path}")
    
    try:
        merged_model.save_pretrained(local_save_path)
        tokenizer.save_pretrained(local_save_path)
        print("   ✅ Local save completed")
    except Exception as e:
        print(f"   ⚠️  Local save failed: {e}")
    
    # Push to Hugging Face Hub
    print(f"   📤 Pushing to Hugging Face Hub: {config.REPO_ID}")
    print("   This may take several minutes depending on model size...")
    
    try:
        # Push model to hub
        merged_model.push_to_hub(
            repo_id=config.REPO_ID,
            commit_message=config.COMMIT_MESSAGE,
            use_auth_token=True
        )
        
        # Push tokenizer to hub
        tokenizer.push_to_hub(
            repo_id=config.REPO_ID,
            commit_message="Add tokenizer files",
            use_auth_token=True
        )
        
        print("   ✅ Model and tokenizer successfully uploaded!")
        
    except Exception as e:
        print(f"   ⚠️  Hub upload failed: {e}")
        print("   Model is saved locally and can be uploaded manually")
        return False
    
    # Create and upload model card with training information
    print("   📝 Creating model documentation...")
    
    try:
        # Get training metrics from trainer history
        final_train_loss = "N/A"
        final_eval_loss = "N/A"
        
        if hasattr(trainer, 'state') and trainer.state.log_history:
            for log_entry in reversed(trainer.state.log_history):
                if 'train_loss' in log_entry and final_train_loss == "N/A":
                    final_train_loss = f"{log_entry['train_loss']:.4f}"
                if 'eval_loss' in log_entry and final_eval_loss == "N/A":
                    final_eval_loss = f"{log_entry['eval_loss']:.4f}"
                if final_train_loss != "N/A" and final_eval_loss != "N/A":
                    break
        
        # Create comprehensive model card
        model_card = f"""# Llama 4 Scout Fine-tuned for Memory Tasks

This model is a fine-tuned version of [{config.MODEL_ID}](https://huggingface.co/{config.MODEL_ID}) on a memory dataset for conversation memory and context retention.

## Model Description

This model has been fine-tuned using LoRA (Low-Rank Adaptation) for parameter-efficient training, then merged back into the base model for optimal inference performance. It's specialized for remembering user preferences, facts, and maintaining conversational context.

## Training Details

### Training Data
- **Dataset**: Local memory dataset ({config.DATASET_PATH})
- **Task**: Conversational Memory and Context Retention
- **Format**: Chat conversations with memory responses

### Training Configuration
- **Base Model**: {config.MODEL_ID}
- **Training Method**: LoRA Fine-tuning with 4-bit quantization
- **LoRA Rank**: {config.LORA_RANK}
- **LoRA Alpha**: {config.LORA_ALPHA}
- **LoRA Dropout**: {config.LORA_DROPOUT}
- **Target Modules**: {', '.join(config.TARGET_MODULES)}

### Training Hyperparameters
- **Epochs**: {config.NUM_EPOCHS}
- **Batch Size**: {config.BATCH_SIZE}
- **Gradient Accumulation Steps**: {config.GRADIENT_ACCUMULATION_STEPS}
- **Learning Rate**: {config.LEARNING_RATE}
- **Weight Decay**: {config.WEIGHT_DECAY}
- **Warmup Ratio**: {config.WARMUP_RATIO}
- **Max Sequence Length**: {config.MAX_SEQ_LENGTH}

### Training Results
- **Final Training Loss**: {final_train_loss}
- **Final Validation Loss**: {final_eval_loss}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{config.REPO_ID}")
model = AutoModelForCausalLM.from_pretrained("{config.REPO_ID}")

# Example conversation with memory
prompt = "<|im_start|>user\\nI love pizza with extra cheese.<|im_end|>\\n<|im_start|>assistant\\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Model Architecture

This model maintains the same architecture as the base Llama 4 Scout model but with learned adaptations from the fine-tuning process.

## Limitations and Biases

Please refer to the base model's documentation for information about limitations and potential biases.

## Training Infrastructure

- **Precision**: Mixed precision (bfloat16)
- **Quantization**: 4-bit during training, full precision for inference
- **Memory Optimization**: Gradient checkpointing enabled

---

*This model was created using the step-by-step tutorial approach for Llama 4 Scout fine-tuning.*
"""
        
        # Upload model card to repository
        upload_file(
            path_or_fileobj=StringIO(model_card),
            path_in_repo="README.md",
            repo_id=config.REPO_ID,
            commit_message="Add comprehensive model card",
            token=os.environ.get("HF_TOKEN"),
        )
        
        print("   ✅ Model card uploaded successfully!")
        
    except Exception as e:
        print(f"   ⚠️  Model card upload failed: {e}")
    
    print(f"   🎉 Deployment completed!")
    print(f"   Model available at: https://huggingface.co/{config.REPO_ID}")
    
    return True

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main execution function that orchestrates the entire fine-tuning process
    
    Executes all 7 steps in sequence with comprehensive error handling
    and progress reporting.
    """
    print("🦙 Llama 4 Scout Fine-tuning - Tutorial Implementation")
    print("=" * 60)
    print("This script follows the exact tutorial steps for educational")
    print("and production fine-tuning of Llama 4 Scout models.")
    print("=" * 60)
    
    # Step 1: Environment Setup
    if not setup_environment():
        print("❌ Environment setup failed. Please check your HF_TOKEN.")
        return
    
    # Step 2: Configuration
    config = Llama4TrainingConfig()
    
    # Step 3: Model and Tokenizer Loading
    model, tokenizer = setup_model_and_tokenizer(config)
    if model is None or tokenizer is None:
        print("❌ Model loading failed.")
        return
    
    # Step 4: Dataset Preparation
    train_dataset, val_dataset, data_collator, original_dataset = load_and_process_data(config, tokenizer)
    if train_dataset is None:
        print("❌ Dataset preparation failed.")
        return
    
    # Step 5: LoRA Application
    model, peft_config = apply_lora(model, config)
    if model is None:
        print("❌ LoRA application failed.")
        return
    
    # Step 6: Training Process
    trainer, train_results, eval_results = train_model(
        model, train_dataset, val_dataset, data_collator, peft_config, config
    )
    if trainer is None:
        print("❌ Training failed.")
        return
    
    # Step 7: Model Merging and Deployment
    deployment_success = save_and_push_to_hub(trainer, tokenizer, config)
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎉 FINE-TUNING COMPLETED!")
    print("=" * 60)
    
    if deployment_success:
        print("✅ All steps completed successfully!")
        print(f"📍 Your fine-tuned model is available at:")
        print(f"   https://huggingface.co/{config.REPO_ID}")
    else:
        print("⚠️  Training completed but deployment had issues.")
        print("📁 Model is saved locally for manual upload.")
    
    print("\n📊 Training Summary:")
    if train_results and hasattr(train_results, 'training_loss'):
        print(f"   Final Training Loss: {train_results.training_loss:.4f}")
    
    if eval_results:
        if 'eval_loss' in eval_results:
            print(f"   Final Validation Loss: {eval_results['eval_loss']:.4f}")
    
    print("\n🚀 Next Steps:")
    print("   1. Test your model with sample prompts")
    print("   2. Evaluate on your specific use cases")
    print("   3. Deploy for production use")
    print("   4. Consider further fine-tuning if needed")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 GPU memory cleaned up")

if __name__ == "__main__":
    main()
