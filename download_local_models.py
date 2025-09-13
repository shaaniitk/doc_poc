#!/usr/bin/env python3
"""
Local Model Download Script

Downloads and caches Hugging Face models locally for offline usage.
Supports different model profiles (lightweight, balanced, high_quality).

Usage:
    python download_local_models.py [--profile PROFILE] [--cache-dir DIR] [--force]
    
    --profile: Model profile to download (lightweight, balanced, high_quality)
    --cache-dir: Directory to cache models (default: ./models)
    --force: Force re-download even if models exist
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    print(f"✗ Missing required packages. Please install with:")
    print(f"  pip install transformers sentence-transformers torch")
    print(f"\nError: {e}")
    sys.exit(1)

# Model configurations for different profiles
MODEL_PROFILES = {
    "lightweight": {
        "embedding_models": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L3-v2"
        ],
        "llm_models": [
            "microsoft/DialoGPT-medium",
            "distilbert-base-uncased"
        ],
        "description": "Fast, lightweight models for basic tasks",
        "total_size_gb": 1.5
    },
    "balanced": {
        "embedding_models": [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2"
        ],
        "llm_models": [
            "Qwen/Qwen2.5-3B-Instruct",  # Primary model for GPU systems with high VRAM
            "microsoft/DialoGPT-large",  # Backup option
            "microsoft/DialoGPT-medium"   # Lightweight fallback
        ],
        "description": "Good balance of performance and resource usage",
        "total_size_gb": 4.2
    },
    "high_quality": {
        "embedding_models": [
            "sentence-transformers/all-distilroberta-v1",
            "sentence-transformers/all-mpnet-base-v2"
        ],
        "llm_models": [
            "huggingface/CodeBERTa-small-v1",
            "roberta-base",
            "microsoft/codebert-base"
        ],
        "description": "High-quality models for best performance",
        "total_size_gb": 8.5
    }
}

class ModelDownloader:
    def __init__(self, cache_dir: str = "./models", force_download: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_download = force_download
        self.download_log = []
        
    def check_disk_space(self, required_gb: float) -> bool:
        """Check if there's enough disk space for downloads."""
        try:
            import shutil
            free_bytes = shutil.disk_usage(self.cache_dir).free
            free_gb = free_bytes / (1024**3)
            
            if free_gb < required_gb:
                print(f"✗ Insufficient disk space. Required: {required_gb:.1f}GB, Available: {free_gb:.1f}GB")
                return False
            
            print(f"✓ Disk space check passed. Available: {free_gb:.1f}GB, Required: {required_gb:.1f}GB")
            return True
        except Exception as e:
            print(f"⚠ Could not check disk space: {e}")
            return True  # Proceed anyway
    
    def check_gpu_availability(self) -> Dict[str, any]:
        """Check GPU availability and memory."""
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": None,
            "memory_gb": 0
        }
        
        if gpu_info["available"]:
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ GPU detected: {gpu_info['device_name']} ({gpu_info['memory_gb']:.1f}GB)")
        else:
            print("ℹ No GPU detected, will use CPU")
        
        return gpu_info
    
    def download_embedding_model(self, model_name: str) -> bool:
        """Download a sentence transformer embedding model."""
        try:
            model_path = self.cache_dir / "embeddings" / model_name.replace("/", "_")
            
            if model_path.exists() and not self.force_download:
                print(f"✓ Embedding model already cached: {model_name}")
                return True
            
            print(f"📥 Downloading embedding model: {model_name}")
            
            # Download using sentence-transformers (handles caching automatically)
            model = SentenceTransformer(model_name, cache_folder=str(self.cache_dir / "embeddings"))
            
            # Test the model with a simple sentence
            test_embedding = model.encode(["This is a test sentence."])
            
            self.download_log.append({
                "model": model_name,
                "type": "embedding",
                "status": "success",
                "path": str(model_path),
                "embedding_dim": len(test_embedding[0])
            })
            
            print(f"✓ Successfully downloaded: {model_name} (dim: {len(test_embedding[0])})")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            self.download_log.append({
                "model": model_name,
                "type": "embedding",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    def download_llm_model(self, model_name: str) -> bool:
        """Download a language model."""
        try:
            model_path = self.cache_dir / "llm" / model_name.replace("/", "_")
            
            if model_path.exists() and not self.force_download:
                print(f"✓ LLM model already cached: {model_name}")
                return True
            
            print(f"📥 Downloading LLM model: {model_name}")
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=str(self.cache_dir / "llm")
            )
            model = AutoModel.from_pretrained(
                model_name, 
                cache_dir=str(self.cache_dir / "llm")
            )
            
            # Test the model with a simple input
            test_input = tokenizer("This is a test.", return_tensors="pt")
            with torch.no_grad():
                test_output = model(**test_input)
            
            self.download_log.append({
                "model": model_name,
                "type": "llm",
                "status": "success",
                "path": str(model_path),
                "vocab_size": tokenizer.vocab_size,
                "hidden_size": test_output.last_hidden_state.shape[-1]
            })
            
            print(f"✓ Successfully downloaded: {model_name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            self.download_log.append({
                "model": model_name,
                "type": "llm",
                "status": "failed",
                "error": str(e)
            })
            return False
    
    def download_profile(self, profile: str) -> bool:
        """Download all models for a specific profile."""
        if profile not in MODEL_PROFILES:
            print(f"✗ Unknown profile: {profile}. Available: {list(MODEL_PROFILES.keys())}")
            return False
        
        profile_config = MODEL_PROFILES[profile]
        print(f"\n🚀 Downloading models for profile: {profile}")
        print(f"📝 Description: {profile_config['description']}")
        print(f"💾 Estimated size: {profile_config['total_size_gb']}GB")
        
        # Check disk space
        if not self.check_disk_space(profile_config['total_size_gb'] * 1.2):  # 20% buffer
            return False
        
        # Check GPU
        gpu_info = self.check_gpu_availability()
        
        success_count = 0
        total_models = len(profile_config['embedding_models']) + len(profile_config['llm_models'])
        
        # Download embedding models
        print(f"\n📊 Downloading embedding models...")
        for model_name in profile_config['embedding_models']:
            if self.download_embedding_model(model_name):
                success_count += 1
        
        # Download LLM models
        print(f"\n🤖 Downloading LLM models...")
        for model_name in profile_config['llm_models']:
            if self.download_llm_model(model_name):
                success_count += 1
        
        # Save download log
        log_file = self.cache_dir / f"download_log_{profile}.json"
        with open(log_file, 'w') as f:
            json.dump({
                "profile": profile,
                "timestamp": str(Path().cwd()),
                "gpu_info": gpu_info,
                "models": self.download_log
            }, f, indent=2)
        
        print(f"\n📋 Download Summary:")
        print(f"✓ Successfully downloaded: {success_count}/{total_models} models")
        print(f"📄 Log saved to: {log_file}")
        
        if success_count == total_models:
            print(f"\n🎉 All models for '{profile}' profile downloaded successfully!")
            return True
        else:
            print(f"\n⚠ Some models failed to download. Check the log for details.")
            return False
    
    def list_cached_models(self) -> Dict[str, List[str]]:
        """List all cached models."""
        cached = {"embedding": [], "llm": []}
        
        embedding_dir = self.cache_dir / "embeddings"
        if embedding_dir.exists():
            cached["embedding"] = [d.name for d in embedding_dir.iterdir() if d.is_dir()]
        
        llm_dir = self.cache_dir / "llm"
        if llm_dir.exists():
            cached["llm"] = [d.name for d in llm_dir.iterdir() if d.is_dir()]
        
        return cached
    
    def cleanup_cache(self, confirm: bool = False) -> bool:
        """Clean up the model cache directory."""
        if not confirm:
            response = input(f"⚠ This will delete all cached models in {self.cache_dir}. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Cleanup cancelled.")
                return False
        
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                print(f"✓ Cache directory cleaned: {self.cache_dir}")
            return True
        except Exception as e:
            print(f"✗ Failed to clean cache: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Download local Hugging Face models')
    parser.add_argument('--profile', choices=list(MODEL_PROFILES.keys()), 
                       default='balanced', help='Model profile to download')
    parser.add_argument('--cache-dir', default='./models', help='Model cache directory')
    parser.add_argument('--force', action='store_true', help='Force re-download existing models')
    parser.add_argument('--list', action='store_true', help='List cached models')
    parser.add_argument('--cleanup', action='store_true', help='Clean up cache directory')
    parser.add_argument('--info', action='store_true', help='Show profile information')
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.cache_dir, args.force)
    
    if args.info:
        print("\n📊 Available Model Profiles:")
        for profile, config in MODEL_PROFILES.items():
            print(f"\n{profile.upper()}:")
            print(f"  Description: {config['description']}")
            print(f"  Size: ~{config['total_size_gb']}GB")
            print(f"  Embedding models: {', '.join(config['embedding_models'])}")
            print(f"  LLM models: {', '.join(config['llm_models'])}")
        return 0
    
    if args.list:
        cached = downloader.list_cached_models()
        print(f"\n📁 Cached models in {args.cache_dir}:")
        print(f"Embedding models: {len(cached['embedding'])}")
        for model in cached['embedding']:
            print(f"  - {model}")
        print(f"LLM models: {len(cached['llm'])}")
        for model in cached['llm']:
            print(f"  - {model}")
        return 0
    
    if args.cleanup:
        if downloader.cleanup_cache():
            return 0
        else:
            return 1
    
    # Download models
    try:
        if downloader.download_profile(args.profile):
            print("\n✅ Model download completed successfully!")
            print("\nNext steps:")
            print("1. Update config: python local_config_updater.py --backup")
            print("2. Test setup: python test_local_setup.py")
            return 0
        else:
            print("\n❌ Model download failed. Check the logs for details.")
            return 1
    except KeyboardInterrupt:
        print("\n\n⏹ Download interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())