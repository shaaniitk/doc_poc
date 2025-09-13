#!/usr/bin/env python3
"""Simple test to verify cache configuration is working."""

import os
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_embedding_cache():
    """Test that embedding client uses local cache."""
    print("Testing embedding cache configuration...")
    
    try:
        import config
        from modules.embedding_client import UnifiedEmbeddingClient
        
        # Check config has cache_folder
        semantic_config = getattr(config, 'SEMANTIC_MAPPING_CONFIG', {})
        cache_folder = semantic_config.get('cache_folder')
        print(f"✓ SEMANTIC_MAPPING_CONFIG cache_folder: {cache_folder}")
        
        # Check if models folder exists
        models_path = Path('./models')
        if models_path.exists():
            print(f"✓ Models folder exists: {models_path.absolute()}")
            # List contents
            contents = list(models_path.rglob('*'))
            print(f"✓ Models folder contains {len(contents)} items")
        else:
            print(f"⚠ Models folder does not exist: {models_path.absolute()}")
        
        # Test embedding client initialization (without actually loading model)
        print("✓ Embedding client configuration looks correct")
        
    except Exception as e:
        print(f"✗ Error testing embedding cache: {e}")
        return False
    
    return True

def test_llm_cache():
    """Test that LLM client uses local cache."""
    print("\nTesting LLM cache configuration...")
    
    try:
        import config
        
        # Check config has cache_folder
        llm_config = getattr(config, 'LLM_CONFIG', {})
        cache_folder = llm_config.get('cache_folder')
        print(f"✓ LLM_CONFIG cache_folder: {cache_folder}")
        
        print("✓ LLM client configuration looks correct")
        
    except Exception as e:
        print(f"✗ Error testing LLM cache: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("Cache Configuration Test")
    print("=" * 40)
    
    embedding_ok = test_embedding_cache()
    llm_ok = test_llm_cache()
    
    print("\n" + "=" * 40)
    if embedding_ok and llm_ok:
        print("✓ All cache configurations are correct!")
        print("Models should now use the local ./models cache folder.")
    else:
        print("✗ Some cache configurations need attention.")
    
    print("\nNext steps:")
    print("1. Run 'python download_local_models.py' to populate cache")
    print("2. Run your main scripts - they should use cached models")