#!/usr/bin/env python3
"""
Model Downloader and Setup Script
Downloads and prepares models for benchmarking from various sources
Supports HuggingFace models, GGUF files, and local models
"""

import os
import argparse
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import json
import shutil

class ModelDownloader:
    def __init__(self, cache_dir="/home/ubuntu/mem0-assignment/model_cache"):
        """Initialize model downloader with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Model cache directory: {self.cache_dir}")
    
    def download_huggingface_model(self, repo_id, model_type="transformers", cache_subdir=None):
        """
        Download model from HuggingFace Hub
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "kingJulio/llama-3.1-8b-memory-finetune-gguf")
            model_type: "transformers" or "gguf"
            cache_subdir: Optional subdirectory name in cache
        """
        print(f"📥 Downloading model from HuggingFace: {repo_id}")
        
        # Determine cache subdirectory
        if cache_subdir is None:
            cache_subdir = repo_id.replace("/", "_")
        
        model_cache_path = self.cache_dir / cache_subdir
        
        # Check if model already exists
        if model_cache_path.exists():
            print(f"✅ Model already cached: {model_cache_path}")
            return str(model_cache_path)
        
        try:
            if model_type == "gguf":
                # For GGUF models, download specific files
                print("🔄 Downloading GGUF model files...")
                
                # Create model directory
                model_cache_path.mkdir(parents=True, exist_ok=True)
                
                # List of files to download for GGUF models
                files_to_download = [
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json", 
                    "special_tokens_map.json",
                    "generation_config.json"
                ]
                
                # Download metadata files first
                for filename in files_to_download:
                    try:
                        downloaded_file = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            cache_dir=str(model_cache_path)
                        )
                        # Copy to model directory for easier access
                        target_file = model_cache_path / filename
                        if not target_file.exists():
                            shutil.copy2(downloaded_file, target_file)
                    except Exception as e:
                        print(f"⚠️ Could not download {filename}: {e}")
                
                # Download GGUF files
                gguf_files = ["unsloth.BF16.gguf", "unsloth.Q4_K_M.gguf"]
                for gguf_file in gguf_files:
                    try:
                        print(f"📥 Downloading {gguf_file}...")
                        downloaded_file = hf_hub_download(
                            repo_id=repo_id,
                            filename=gguf_file,
                            cache_dir=str(model_cache_path)
                        )
                        # Copy to model directory
                        target_file = model_cache_path / gguf_file
                        if not target_file.exists():
                            shutil.copy2(downloaded_file, target_file)
                        print(f"✅ Downloaded: {gguf_file}")
                    except Exception as e:
                        print(f"⚠️ Could not download {gguf_file}: {e}")
                
            else:
                # For standard transformers models
                print("🔄 Downloading transformers model...")
                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(self.cache_dir),
                    local_dir=str(model_cache_path)
                )
                print(f"✅ Model downloaded to: {downloaded_path}")
            
            print(f"✅ Model cached at: {model_cache_path}")
            return str(model_cache_path)
            
        except Exception as e:
            print(f"❌ Failed to download model {repo_id}: {e}")
            # Clean up partial download
            if model_cache_path.exists():
                shutil.rmtree(model_cache_path)
            raise
    
    def setup_base_model(self):
        """Setup base model for benchmarking"""
        print("🔄 Setting up base model...")
        
        # Base model is already cached in the mem0-backend structure
        base_model_path = "/home/ubuntu/mem0-assignment/mem0-backend/model_cache/models--unsloth--llama-3.1-8b-bnb-4bit/snapshots/b80adf5d249b569469d0a19192ff36e88f133413"
        
        if Path(base_model_path).exists():
            print(f"✅ Base model found: {base_model_path}")
            return base_model_path
        else:
            print("⚠️ Base model not found in expected location. Downloading...")
            # Download base model using HuggingFace
            return self.download_huggingface_model(
                "unsloth/llama-3.1-8b-bnb-4bit",
                model_type="transformers",
                cache_subdir="base_model"
            )
    
    def setup_finetuned_models(self, repo_id="kingJulio/llama-3.1-8b-memory-finetune-gguf"):
        """Setup fine-tuned models from HuggingFace"""
        print(f"🔄 Setting up fine-tuned models from: {repo_id}")
        
        # Download GGUF models
        gguf_path = self.download_huggingface_model(
            repo_id,
            model_type="gguf",
            cache_subdir="finetuned_gguf"
        )
        
        return {
            "gguf_path": gguf_path,
            "bf16_file": str(Path(gguf_path) / "unsloth.BF16.gguf"),
            "q4km_file": str(Path(gguf_path) / "unsloth.Q4_K_M.gguf")
        }
    
    def install_dependencies(self):
        """Install required dependencies for different model types"""
        print("🔧 Installing model dependencies...")
        
        dependencies = [
            "transformers>=4.36.0",
            "torch>=2.0.0",
            "accelerate",
            "bitsandbytes",
            "peft",
            "huggingface_hub",
            "llama-cpp-python"
        ]
        
        for dep in dependencies:
            try:
                print(f"📦 Installing {dep}...")
                subprocess.run([
                    "pip", "install", dep
                ], check=True, capture_output=True)
                print(f"✅ Installed: {dep}")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Warning: Failed to install {dep}: {e}")
    
    def verify_model_setup(self, model_path, model_type="transformers"):
        """Verify that model setup is correct"""
        print(f"🔍 Verifying model setup: {model_path}")
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"❌ Model path does not exist: {model_path}")
            return False
        
        if model_type == "transformers":
            # Check for required transformers files
            required_files = ["config.json"]
            optional_files = ["tokenizer.json", "tokenizer_config.json"]
            
            for file in required_files:
                if not (model_path / file).exists():
                    print(f"❌ Missing required file: {file}")
                    return False
            
            for file in optional_files:
                if (model_path / file).exists():
                    print(f"✅ Found: {file}")
                else:
                    print(f"⚠️ Optional file missing: {file}")
        
        elif model_type == "gguf":
            # Check for GGUF files
            gguf_files = list(model_path.glob("*.gguf"))
            if not gguf_files:
                print(f"❌ No GGUF files found in: {model_path}")
                return False
            
            print(f"✅ Found GGUF files: {[f.name for f in gguf_files]}")
        
        print(f"✅ Model setup verification passed: {model_path}")
        return True
    
    def list_available_models(self):
        """List all cached models"""
        print("📋 Available cached models:")
        
        if not self.cache_dir.exists():
            print("  No models cached yet.")
            return []
        
        models = []
        for item in self.cache_dir.iterdir():
            if item.is_dir():
                # Check what type of model it is
                if list(item.glob("*.gguf")):
                    model_type = "GGUF"
                    files = [f.name for f in item.glob("*.gguf")]
                elif (item / "config.json").exists():
                    model_type = "Transformers"
                    files = ["config.json", "tokenizer files"]
                else:
                    model_type = "Unknown"
                    files = []
                
                models.append({
                    "name": item.name,
                    "path": str(item),
                    "type": model_type,
                    "files": files
                })
                
                print(f"  📁 {item.name} ({model_type})")
                if files:
                    print(f"     Files: {', '.join(files[:3])}{'...' if len(files) > 3 else ''}")
        
        return models
    
    def clean_cache(self, confirm=True):
        """Clean model cache"""
        if confirm:
            response = input(f"⚠️ This will delete all cached models in {self.cache_dir}. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Cache cleaning cancelled.")
                return
        
        print("🧹 Cleaning model cache...")
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Cache cleaned successfully.")
        else:
            print("✅ Cache directory was already empty.")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model Downloader and Setup")
    parser.add_argument("--action", type=str, 
                       choices=["download", "setup-all", "list", "verify", "clean", "install-deps"],
                       default="setup-all",
                       help="Action to perform")
    parser.add_argument("--repo-id", type=str,
                       default="kingJulio/llama-3.1-8b-memory-finetune-gguf",
                       help="HuggingFace repository ID")
    parser.add_argument("--model-type", type=str,
                       choices=["transformers", "gguf"],
                       default="gguf",
                       help="Type of model to download")
    parser.add_argument("--cache-dir", type=str,
                       default="/home/ubuntu/mem0-assignment/model_cache",
                       help="Model cache directory")
    parser.add_argument("--verify-path", type=str,
                       help="Path to verify model setup")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir)
    
    try:
        if args.action == "install-deps":
            downloader.install_dependencies()
            
        elif args.action == "download":
            downloader.download_huggingface_model(args.repo_id, args.model_type)
            
        elif args.action == "setup-all":
            print("🚀 Setting up all models for benchmarking...")
            
            # Install dependencies
            downloader.install_dependencies()
            
            # Setup base model
            base_path = downloader.setup_base_model()
            print(f"✅ Base model ready: {base_path}")
            
            # Setup fine-tuned models
            ft_models = downloader.setup_finetuned_models(args.repo_id)
            print(f"✅ Fine-tuned models ready:")
            print(f"   GGUF directory: {ft_models['gguf_path']}")
            print(f"   BF16 file: {ft_models['bf16_file']}")
            print(f"   Q4_K_M file: {ft_models['q4km_file']}")
            
            # Verify setups
            downloader.verify_model_setup(base_path, "transformers")
            downloader.verify_model_setup(ft_models['gguf_path'], "gguf")
            
            print("\n🎉 All models ready for benchmarking!")
            
        elif args.action == "list":
            downloader.list_available_models()
            
        elif args.action == "verify":
            if args.verify_path:
                downloader.verify_model_setup(args.verify_path, args.model_type)
            else:
                print("❌ --verify-path required for verify action")
                
        elif args.action == "clean":
            downloader.clean_cache()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
