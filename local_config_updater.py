#!/usr/bin/env python3
"""
Local Configuration Updater for Document Processing Pipeline

This script updates the main config.py to use local Hugging Face models
instead of external API services like OpenAI or Mistral.

Usage:
    python local_config_updater.py [--backup] [--restore]
    
    --backup: Create a backup of current config before updating
    --restore: Restore from backup (config.py.backup)
"""

import os
import shutil
import argparse
from pathlib import Path

# Local model configurations optimized for different use cases
LOCAL_CONFIGS = {
    "lightweight": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "microsoft/DialoGPT-medium",
        "device": "cpu",
        "batch_size": 16
    },
    "balanced": {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "llm_model": "microsoft/DialoGPT-large",
        "device": "auto",  # Will use GPU if available
        "batch_size": 32
    },
    "high_quality": {
        "embedding_model": "sentence-transformers/all-distilroberta-v1",
        "llm_model": "huggingface/CodeBERTa-small-v1",
        "device": "auto",
        "batch_size": 8
    }
}

def backup_config(config_path):
    """Create a backup of the current config file."""
    backup_path = config_path.with_suffix('.py.backup')
    shutil.copy2(config_path, backup_path)
    print(f"[OK] Backup created: {backup_path}")
    return backup_path

def restore_config(config_path):
    """Restore config from backup."""
    backup_path = config_path.with_suffix('.py.backup')
    if backup_path.exists():
        shutil.copy2(backup_path, config_path)
        print(f"[OK] Config restored from: {backup_path}")
        return True
    else:
        print(f"[ERROR] No backup found at: {backup_path}")
        return False

def update_config_for_local_models(config_path, profile="balanced"):
    """Update config.py to use local Hugging Face models."""
    
    if profile not in LOCAL_CONFIGS:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(LOCAL_CONFIGS.keys())}")
    
    local_config = LOCAL_CONFIGS[profile]
    
    # Read current config
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update SEMANTIC_MAPPING_CONFIG for local embeddings
    semantic_mapping_replacement = f'''# Option 2: Local SentenceTransformer Model (default - no API key required)
SEMANTIC_MAPPING_CONFIG = {{
    "model": "{local_config['embedding_model']}",
    "provider": "sentence_transformer",
    "similarity_threshold": 0.6,
    "device": "{local_config['device']}",
    "batch_size": {local_config['batch_size']},
    "top_k_candidates": 3,
    # Accept borderline matches within this margin below the threshold
    "soft_accept_margin": 0.05,
    # Also accept if the top-1 similarity exceeds top-2 by at least this gap
    "gap_accept_margin": 0.1,
    # Explicit alias used by output_manager (falls back to soft_accept_margin if absent)
    "low_confidence_margin": 0.05,
    # Boosting knobs for intelligent_mapper._run_graph_boost_pass
    "confidence_threshold": 0.6,
    "boost_amount": 0.3,
    "enable_neighbor_window_boost": True,
    "neighbor_window": 2,
    "neighbor_boost_amount": 0.05,
    "neighbor_max_boost": 0.2,
}}'''
    
    # Update LLM_CONFIG for local models
    llm_config_replacement = f'''LLM_CONFIG = {{
    "provider": "huggingface_local",  # Local Hugging Face models
    "model": "{local_config['llm_model']}",
    "api_key_env": None,  # No API key needed for local models
    "max_tokens": 2048,
    "temperature": 0.1,
    "timeout": 30,
    "device": "{local_config['device']}",
    "local_model_path": None  # Will download automatically if not cached
}}'''
    
    # Update CHUNKING_EMBEDDING for local embeddings
    chunking_embedding_replacement = f'''# --- NEW: Embedding-guided chunking configuration ---
CHUNKING_EMBEDDING = {{
    "enable": True,
    "cohesion_threshold": 0.7,
    "max_tokens_per_chunk": 1500,
    "overlap_tokens": 200,
    "fallback_provider": "sentence_transformer",
    "boundary_detection_method": "cohesion_minima",
    "adaptive_sizing": True,
    "smart_overlap": True,
    "local_model": "{local_config['embedding_model']}",
    "device": "{local_config['device']}"
}}'''
    
    # Update MISTRAL_REFINEMENT to use local models
    mistral_refinement_replacement = f'''# --- NEW: Local LLM refinement configuration ---
LOCAL_LLM_REFINEMENT = {{
    "enable": True,
    "only_for_low_confidence": True,
    "confidence_threshold": 0.6,
    "max_cases_per_doc": 10,
    "refinement_model": "{local_config['llm_model']}",
    "temperature": 0.1,
    "max_tokens": 1024,
    "device": "{local_config['device']}"
}}'''
    
    # Perform replacements
    import re
    
    # Replace SEMANTIC_MAPPING_CONFIG
    content = re.sub(
        r'# Option 2: Local SentenceTransformer Model.*?}',
        semantic_mapping_replacement,
        content,
        flags=re.DOTALL
    )
    
    # Replace LLM_CONFIG
    content = re.sub(
        r'LLM_CONFIG = {.*?}',
        llm_config_replacement,
        content,
        flags=re.DOTALL
    )
    
    # Replace CHUNKING_EMBEDDING
    content = re.sub(
        r'# --- NEW: Embedding-guided chunking configuration ---.*?CHUNKING_EMBEDDING = {.*?}',
        chunking_embedding_replacement,
        content,
        flags=re.DOTALL
    )
    
    # Replace MISTRAL_REFINEMENT with LOCAL_LLM_REFINEMENT
    content = re.sub(
        r'# --- NEW: Mistral refinement configuration ---.*?MISTRAL_REFINEMENT = {.*?}',
        mistral_refinement_replacement,
        content,
        flags=re.DOTALL
    )
    
    # Add local model configuration section at the end
    local_model_section = f'''

# === LOCAL MODEL CONFIGURATION ===
# Configuration for running models locally without external APIs
LOCAL_MODEL_CONFIG = {{
    "profile": "{profile}",
    "embedding_model": "{local_config['embedding_model']}",
    "llm_model": "{local_config['llm_model']}",
    "device": "{local_config['device']}",
    "batch_size": {local_config['batch_size']},
    "cache_dir": "./models",  # Local model cache directory
    "download_timeout": 300,  # Timeout for model downloads (seconds)
    "enable_gpu_if_available": True,
    "memory_optimization": True
}}

# Hardware requirements for different profiles
HARDWARE_REQUIREMENTS = {{
    "lightweight": {{
        "min_ram_gb": 4,
        "min_disk_gb": 2,
        "gpu_required": False,
        "description": "Runs on most systems, basic performance"
    }},
    "balanced": {{
        "min_ram_gb": 8,
        "min_disk_gb": 5,
        "gpu_required": False,
        "gpu_recommended": True,
        "description": "Good balance of performance and resource usage"
    }},
    "high_quality": {{
        "min_ram_gb": 16,
        "min_disk_gb": 10,
        "gpu_required": True,
        "description": "Best performance, requires dedicated GPU"
    }}
}}
'''
    
    # Add the local model section before the last line
    if not "LOCAL_MODEL_CONFIG" in content:
        content = content.rstrip() + local_model_section
    
    # Write updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] Config updated for local models (profile: {profile})")
    print(f"[OK] Embedding model: {local_config['embedding_model']}")
    print(f"[OK] LLM model: {local_config['llm_model']}")
    print(f"[OK] Device: {local_config['device']}")

def main():
    parser = argparse.ArgumentParser(description='Update config for local model usage')
    parser.add_argument('--backup', action='store_true', help='Create backup before updating')
    parser.add_argument('--restore', action='store_true', help='Restore from backup')
    parser.add_argument('--profile', choices=list(LOCAL_CONFIGS.keys()), 
                       default='balanced', help='Model profile to use')
    
    args = parser.parse_args()
    
    config_path = Path('config.py')
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1
    
    if args.restore:
        if restore_config(config_path):
            print("[OK] Configuration restored successfully")
            return 0
        else:
            return 1
    
    if args.backup:
        backup_config(config_path)
    
    try:
        update_config_for_local_models(config_path, args.profile)
        print("\n[OK] Configuration updated successfully!")
        print("\nNext steps:")
        print("1. Install required packages: pip install -r requirements.txt")
        print("2. Run model download script: python download_local_models.py")
        print("3. Test with: python test_local_setup.py")
        return 0
    except Exception as e:
        print(f"[ERROR] Error updating config: {e}")
        return 1

if __name__ == '__main__':
    exit(main())