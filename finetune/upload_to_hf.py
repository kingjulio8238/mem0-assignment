#!/usr/bin/env python3
"""
Dedicated HuggingFace Model Upload Script
Handles model repository creation and upload with proper error handling
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


class HuggingFaceUploader:
    """Handle HuggingFace model uploads with proper repository management."""
    
    def __init__(self, token: Optional[str] = None):
        self.api = HfApi()
        self.token = token or os.getenv("HF_TOKEN")
        
        if not self.token:
            raise ValueError("HuggingFace token required. Set HF_TOKEN environment variable or provide --token")
        
        # Login to HuggingFace
        try:
            login(token=self.token)
            user_info = self.api.whoami(token=self.token)
            print(f"✅ Logged in as: {user_info['name']}")
        except Exception as e:
            raise ValueError(f"Failed to login to HuggingFace: {str(e)}")
    
    def create_repository_if_needed(self, repo_name: str, private: bool = False) -> bool:
        """Create repository if it doesn't exist."""
        try:
            # Check if repo exists
            self.api.repo_info(repo_id=repo_name, repo_type="model", token=self.token)
            print(f"📁 Repository {repo_name} already exists")
            return True
            
        except RepositoryNotFoundError:
            # Repository doesn't exist, create it
            print(f"🏗️ Creating new repository: {repo_name}")
            try:
                create_repo(
                    repo_id=repo_name,
                    token=self.token,
                    private=private,
                    repo_type="model",
                    exist_ok=True
                )
                print(f"✅ Repository {repo_name} created successfully")
                return True
                
            except Exception as e:
                print(f"❌ Failed to create repository: {str(e)}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking repository: {str(e)}")
            return False
    
    def upload_model(self, model_path: str, repo_name: str, 
                    commit_message: str = "Upload fine-tuned memory model",
                    private: bool = False) -> bool:
        """Upload model to HuggingFace Hub."""
        
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"❌ Model path does not exist: {model_path}")
            return False
        
        print(f"🚀 Starting upload to {repo_name}")
        print(f"📁 Model path: {model_path}")
        
        # Create repository if needed
        if not self.create_repository_if_needed(repo_name, private):
            return False
        
        try:
            # Upload the entire model directory
            self.api.upload_folder(
                folder_path=str(model_path),
                repo_id=repo_name,
                repo_type="model",
                commit_message=commit_message,
                token=self.token,
                ignore_patterns=["*.git*", "__pycache__", "*.pyc", "*.tmp", "optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin", "trainer_state.json"]
            )
            
            print(f"✅ Model uploaded successfully!")
            print(f"🔗 Model URL: https://huggingface.co/{repo_name}")
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {str(e)}")
            return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Upload fine-tuned model to HuggingFace Hub")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the fine-tuned model directory")
    parser.add_argument("--repo-name", type=str, required=True,
                       help="HuggingFace repository name (username/model-name)")
    parser.add_argument("--token", type=str,
                       help="HuggingFace token (or set HF_TOKEN environment variable)")
    parser.add_argument("--private", action="store_true",
                       help="Create private repository")
    parser.add_argument("--commit-message", type=str,
                       default="Upload fine-tuned memory model",
                       help="Commit message for the upload")
    
    return parser.parse_args()


def main():
    """Main upload function."""
    try:
        args = parse_args()
        
        # Create uploader and upload model
        uploader = HuggingFaceUploader(token=args.token)
        
        success = uploader.upload_model(
            model_path=args.model_path,
            repo_name=args.repo_name,
            commit_message=args.commit_message,
            private=args.private
        )
        
        if success:
            print(f"\n🎉 Upload completed successfully!")
        else:
            print(f"\n❌ Upload failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Upload script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
