#!/usr/bin/env python3
"""
Dedicated GGUF Export Script
Converts fine-tuned model to GGUF format for llama.cpp compatibility
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

from unsloth import FastLanguageModel


class GGUFExporter:
    """Handle GGUF model export with proper dependency management."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def install_dependencies(self):
        """Install required dependencies for GGUF export."""
        print("🔧 Installing GGUF export dependencies...")
        
        try:
            # Install curl for llama.cpp build
            subprocess.run(["sudo", "apt-get", "update"], check=True, capture_output=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "libcurl4-openssl-dev", "curl"], 
                         check=True, capture_output=True)
            print("✅ CURL dependencies installed")
            
            # Install cmake if not present
            subprocess.run(["sudo", "apt-get", "install", "-y", "cmake", "build-essential"], 
                         check=True, capture_output=True)
            print("✅ Build tools installed")
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Failed to install some dependencies: {e}")
            print("💡 Continuing with export attempt...")
    
    def load_model(self, model_path: str, base_model: str = "unsloth/llama-3.1-8b-bnb-4bit"):
        """Load the fine-tuned model."""
        print(f"🚀 Loading model from {model_path}")
        
        try:
            # Load the fine-tuned model in full precision for GGUF conversion
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=False,  # Load in full precision for proper merging
            )
            
            print("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            return False
    
    def merge_and_export_gguf(self, output_path: str, quantization: str = "q4_k_m"):
        """Merge LoRA weights and export to GGUF format."""
        if self.model is None:
            print("❌ No model loaded")
            return False
        
        print(f"🔄 Merging LoRA weights and exporting to GGUF...")
        print(f"📁 Output path: {output_path}")
        print(f"⚙️ Quantization: {quantization}")
        
        try:
            # Create output directory
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Method 1: Use Unsloth's built-in GGUF export
            try:
                print("🔄 Attempting Unsloth GGUF export...")
                self.model.save_pretrained_gguf(
                    str(output_path), 
                    self.tokenizer,
                    quantization_method=quantization
                )
                print("✅ GGUF export successful via Unsloth")
                return True
                
            except Exception as e:
                print(f"⚠️ Unsloth GGUF export failed: {e}")
                print("🔄 Trying alternative method...")
            
            # Method 2: Manual merge then convert
            try:
                print("🔄 Merging LoRA weights...")
                
                # Merge the LoRA adapter into the base model
                merged_model = self.model.merge_and_unload()
                
                # Save merged model in HF format first
                temp_merged_path = output_path / "temp_merged"
                temp_merged_path.mkdir(exist_ok=True)
                
                merged_model.save_pretrained(str(temp_merged_path))
                self.tokenizer.save_pretrained(str(temp_merged_path))
                
                print("✅ Model merged successfully")
                print(f"📁 Merged model saved to: {temp_merged_path}")
                
                # Convert to GGUF using llama.cpp convert script
                self.convert_to_gguf_manual(temp_merged_path, output_path, quantization)
                
                return True
                
            except Exception as e:
                print(f"❌ Manual merge and convert failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ GGUF export failed: {str(e)}")
            return False
    
    def convert_to_gguf_manual(self, merged_model_path: Path, output_path: Path, quantization: str):
        """Manual conversion using llama.cpp tools."""
        print("🔄 Converting to GGUF using llama.cpp...")
        
        # Check if llama.cpp is available
        llama_cpp_path = Path("./llama.cpp")
        if not llama_cpp_path.exists():
            print("📥 Cloning llama.cpp...")
            subprocess.run([
                "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
            ], check=True)
        
        # Build llama.cpp with CURL support
        os.chdir(llama_cpp_path)
        try:
            print("🔨 Building llama.cpp...")
            subprocess.run([
                "cmake", "-B", "build", 
                "-DLLAMA_CURL=ON",  # Enable CURL support
                "-DCMAKE_BUILD_TYPE=Release"
            ], check=True)
            
            subprocess.run([
                "cmake", "--build", "build", "--config", "Release", "-j", "4"
            ], check=True)
            
            print("✅ llama.cpp built successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build llama.cpp: {e}")
            raise
        finally:
            os.chdir("..")
        
        # Convert model to GGUF
        try:
            # Try different possible locations for the conversion script
            convert_script_options = [
                llama_cpp_path / "convert_hf_to_gguf.py",  # Correct name
                llama_cpp_path / "convert-hf-to-gguf.py",  # Incorrect name (legacy)
                llama_cpp_path / "build" / "bin" / "convert_hf_to_gguf.py",
            ]
            
            convert_script = None
            for script_path in convert_script_options:
                if script_path.exists():
                    convert_script = script_path
                    break
            
            if convert_script is None:
                raise FileNotFoundError("Could not find convert_hf_to_gguf.py script")
            
            gguf_file = output_path / f"model-{quantization}.gguf"
            
            print(f"🔄 Converting {merged_model_path} to GGUF...")
            subprocess.run([
                "python", str(convert_script),
                str(merged_model_path),
                "--outfile", str(gguf_file),
                "--outtype", quantization
            ], check=True)
            
            print(f"✅ GGUF conversion complete: {gguf_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ GGUF conversion failed: {e}")
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export fine-tuned model to GGUF format")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the fine-tuned model directory")
    parser.add_argument("--output-path", type=str, default="./model_gguf",
                       help="Output directory for GGUF files")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                       choices=["f16", "q4_0", "q4_1", "q4_k_m", "q4_k_s", "q5_0", "q5_1", "q5_k_m", "q5_k_s", "q8_0"],
                       help="Quantization method for GGUF")
    parser.add_argument("--base-model", type=str, default="unsloth/llama-3.1-8b-bnb-4bit",
                       help="Base model name")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install required dependencies")
    
    return parser.parse_args()


def main():
    """Main export function."""
    try:
        args = parse_args()
        
        exporter = GGUFExporter()
        
        # Install dependencies if requested
        if args.install_deps:
            exporter.install_dependencies()
        
        # Load model
        if not exporter.load_model(args.model_path, args.base_model):
            sys.exit(1)
        
        # Export to GGUF
        success = exporter.merge_and_export_gguf(args.output_path, args.quantization)
        
        if success:
            print(f"\n🎉 GGUF export completed successfully!")
            print(f"📁 Output directory: {args.output_path}")
            print(f"⚙️ Quantization: {args.quantization}")
        else:
            print(f"\n❌ GGUF export failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Export cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Export script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
