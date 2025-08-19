"""
Model Acquisition Module for Llama 4 Scout and 8B Instruct Models
Phase 2A: Download and setup models (no GPU required for this step)
"""

import os
import torch
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from pathlib import Path

class ModelAcquisition:
    """Handle downloading and initial setup of Llama models"""
    
    def __init__(self, cache_dir="./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model configurations
        # Llama 4 Scout: 16 experts, ~109B total params, 17B active, 10M context length
        self.llama4_scout_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        self.instruct_8b_id = "unsloth/llama-3.1-8b-bnb-4bit"
        
        # GPU availability check
        self.gpu_available = torch.cuda.is_available()
        self.mps_available = torch.backends.mps.is_available()
        print(f"GPU Available: {self.gpu_available}")
        print(f"MPS Available: {self.mps_available}")
        if self.gpu_available:
            print(f"GPU Device: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif self.mps_available:
            print("Using Apple Silicon MPS backend")
    
    def setup_quantization_config(self, bits=4):
        """Setup quantization configuration for memory efficiency"""
        try:
            return BitsAndBytesConfig(
                load_in_4bit=(bits == 4),
                load_in_8bit=(bits == 8),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        except Exception as e:
            print(f"⚠️  BitsAndBytes not available: {e}")
            print("💡 Falling back to standard precision (no quantization)")
            return None
    
    def download_llama4_scout(self, download_only=True):
        """
        Download Llama 4 Scout model
        Args:
            download_only: If True, only download without loading (no GPU needed)
        """
        print("🔽 Starting Llama 4 Scout acquisition...")
        
        try:
            # Download tokenizer (always works, no GPU needed)
            print("📥 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.llama4_scout_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            print("✅ Tokenizer downloaded successfully")
            
            if download_only:
                # Just download model files without loading into memory
                print("📥 Downloading model files...")
                AutoModelForCausalLM.from_pretrained(
                    self.llama4_scout_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map=None,  # Don't load to device
                    low_cpu_mem_usage=True
                )
                print("✅ Llama 4 Scout model files downloaded")
                return {"status": "downloaded", "tokenizer": tokenizer}
            
            else:
                # Full loading (requires GPU)
                if not self.gpu_available:
                    raise RuntimeError("GPU required for model loading")
                
                quantization_config = self.setup_quantization_config(4)
                model = AutoModelForCausalLM.from_pretrained(
                    self.llama4_scout_id,
                    cache_dir=self.cache_dir,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=self._get_optimal_attention()  # Optimal attention
                )
                print("✅ Llama 4 Scout loaded into GPU memory")
                return {"status": "loaded", "model": model, "tokenizer": tokenizer}
                
        except Exception as e:
            print(f"❌ Error with Llama 4 Scout: {e}")
            return {"status": "error", "error": str(e)}
    
    def download_instruct_8b(self, download_only=True):
        """
        Download 8B Instruct model
        Args:
            download_only: If True, only download without loading (no GPU needed)
        """
        print("🔽 Starting 8B Instruct model acquisition...")
        
        try:
            # Download tokenizer
            print("📥 Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.instruct_8b_id,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            print("✅ Tokenizer downloaded successfully")
            
            if download_only:
                # Just download model files without quantization for compatibility
                print("📥 Downloading model files...")
                AutoModelForCausalLM.from_pretrained(
                    self.instruct_8b_id,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float32 if not self.gpu_available else torch.bfloat16,
                    device_map=None,  # Don't load to device
                    low_cpu_mem_usage=True
                )
                print("✅ 8B Instruct model files downloaded")
                return {"status": "downloaded", "tokenizer": tokenizer}
            
            else:
                # Full loading - support both GPU and MPS
                device = "auto" if self.gpu_available else ("mps" if self.mps_available else "cpu")
                print(f"Loading to device: {device}")
                
                # Configure for available hardware
                load_kwargs = {
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16 if (self.gpu_available or self.mps_available) else torch.float32,
                    "low_cpu_mem_usage": True
                }
                
                # Add device mapping
                if device != "cpu":
                    load_kwargs["device_map"] = device
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.instruct_8b_id,
                    **load_kwargs
                )
                print("✅ 8B Instruct loaded into GPU memory")
                return {"status": "loaded", "model": model, "tokenizer": tokenizer}
                
        except Exception as e:
            print(f"❌ Error with 8B Instruct: {e}")
            return {"status": "error", "error": str(e)}
    
    def download_selected_models(self, models=["both"], download_only=True):
        """Download selected models based on user choice"""
        print("🚀 Starting model acquisition process...")
        print(f"Mode: {'Download only' if download_only else 'Download and load'}")
        print(f"Models to download: {', '.join(models)}")
        
        results = {}
        
        # Determine which models to download
        download_llama4 = "both" in models or "llama4" in models
        download_8b = "both" in models or "8b" in models or "instruct" in models
        
        if download_llama4:
            print("\n📥 Downloading Llama 4 Scout...")
            results["llama4_scout"] = self.download_llama4_scout(download_only)
        else:
            print("\n⏭️  Skipping Llama 4 Scout")
            
        if download_8b:
            print("\n📥 Downloading 8B Instruct...")
            results["instruct_8b"] = self.download_instruct_8b(download_only)
        else:
            print("\n⏭️  Skipping 8B Instruct")
        
        # Summary
        print("\n📊 Acquisition Summary:")
        for model_name, result in results.items():
            status = result["status"]
            emoji = "✅" if status in ["downloaded", "loaded"] else "❌"
            print(f"{emoji} {model_name}: {status}")
        
        return results
    
    def download_all_models(self, download_only=True):
        """Download both models (legacy method)"""
        return self.download_selected_models(["both"], download_only)
    
    def get_model_info(self):
        """Get information about cached models"""
        cache_contents = list(self.cache_dir.rglob("*"))
        
        info = {
            "cache_directory": str(self.cache_dir),
            "total_files": len(cache_contents),
            "cache_size_gb": sum(f.stat().st_size for f in cache_contents if f.is_file()) / 1e9,
            "gpu_available": self.gpu_available
        }
        
        return info
    
    def _get_optimal_attention(self):
        """Get the best available attention implementation"""
        try:
            import flash_attn
            if self.gpu_available:
                return "flash_attention_2"
        except ImportError:
            pass
        
        # Fallback to SDPA (available in PyTorch 2.0+)
        return "sdpa"
    
    def install_flash_attention(self):
        """Helper to install flash attention with proper instructions"""
        if not self.gpu_available:
            print("⚠️  Flash Attention requires CUDA GPU. You're on CPU/MPS.")
            print("💡 For optimal performance on GPU, install flash-attn:")
            print("   pip install flash-attn --no-build-isolation")
            return False
        
        print("💡 To install Flash Attention for optimal performance:")
        print("   pip install flash-attn --no-build-isolation")
        print("   (Requires CUDA toolkit)")
        return True

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="🎯 Phase 2A: Model Acquisition for Mem0 Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_acquisition.py                          # Download both models (default)
  python model_acquisition.py --models 8b              # Download only 8B instruct model
  python model_acquisition.py --models llama4          # Download only Llama 4 Scout
  python model_acquisition.py --models 8b llama4       # Download both models explicitly
  python model_acquisition.py --load                   # Download and load models into memory
  python model_acquisition.py --models 8b --load       # Download and load only 8B model
        """
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        choices=["both", "8b", "instruct", "llama4"],
        default=["both"],
        help="Choose which models to download (default: both)"
    )
    
    parser.add_argument(
        "--load", 
        action="store_true",
        help="Load models into memory after download (requires GPU/MPS)"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="./model_cache",
        help="Directory to cache downloaded models (default: ./model_cache)"
    )
    
    args = parser.parse_args()
    
    print("🎯 Phase 2A: Model Acquisition")
    print(f"📋 Selected models: {', '.join(args.models)}")
    print(f"💾 Load into memory: {'Yes' if args.load else 'No (download only)'}")
    
    # Initialize acquisition system
    acquisition = ModelAcquisition(cache_dir=args.cache_dir)
    
    # Show system info
    info = acquisition.get_model_info()
    print(f"📁 Cache directory: {info['cache_directory']}")
    print(f"💾 Current cache size: {info['cache_size_gb']:.2f} GB")
    
    # Download selected models
    results = acquisition.download_selected_models(
        models=args.models, 
        download_only=not args.load
    )
    
    # Show final info
    final_info = acquisition.get_model_info()
    print(f"\n💾 Final cache size: {final_info['cache_size_gb']:.2f} GB")
    print(f"📁 Total files cached: {final_info['total_files']}")
    
    # Show flash attention recommendation
    if args.load and not acquisition.gpu_available:
        print("\n💡 Tip: For optimal performance with GPU, install flash attention:")
        print("   pip install flash-attn --no-build-isolation")
    
    return results

if __name__ == "__main__":
    main()
