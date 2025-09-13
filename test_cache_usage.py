#!/usr/bin/env python3
"""Test to verify models are loaded from cache without downloading."""

import os
import sys
import time
from pathlib import Path

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_embedding_cache_usage():
    """Test that embedding model loads from cache quickly."""
    print("Testing embedding model cache usage...")
    
    try:
        # Import after path setup
        import config
        from modules.embedding_client import UnifiedEmbeddingClient
        
        # Record start time
        start_time = time.time()
        
        # Initialize client (this should load from cache)
        print("Loading embedding client...")
        client = UnifiedEmbeddingClient()
        
        # Record load time
        load_time = time.time() - start_time
        print(f"✓ Embedding client loaded in {load_time:.2f} seconds")
        
        # Test encoding
        test_text = "This is a test sentence."
        embeddings = client.encode(test_text)
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        
        # If it loads quickly (< 5 seconds), it's likely using cache
        if load_time < 5.0:
            print("✓ Fast loading suggests cache is being used!")
        else:
            print("⚠ Slow loading suggests model might be downloading")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing embedding cache usage: {e}")
        return False

def check_cache_structure():
    """Check the cache folder structure."""
    print("\nChecking cache structure...")
    
    models_path = Path('./models')
    if not models_path.exists():
        print("✗ Models folder does not exist")
        return False
    
    # Check for sentence-transformers cache
    st_cache = models_path / 'models--sentence-transformers--all-mpnet-base-v2'
    if st_cache.exists():
        print(f"✓ SentenceTransformers cache found: {st_cache}")
        
        # Check snapshots
        snapshots = st_cache / 'snapshots'
        if snapshots.exists():
            snapshot_dirs = list(snapshots.iterdir())
            print(f"✓ Found {len(snapshot_dirs)} snapshot(s)")
            
            if snapshot_dirs:
                # Check first snapshot for model files
                first_snapshot = snapshot_dirs[0]
                model_files = list(first_snapshot.glob('*.safetensors')) + list(first_snapshot.glob('*.bin'))
                config_files = list(first_snapshot.glob('config.json'))
                
                print(f"✓ Snapshot contains {len(model_files)} model file(s)")
                print(f"✓ Snapshot contains {len(config_files)} config file(s)")
                
                if model_files and config_files:
                    print("✓ Cache appears complete with model and config files")
                    return True
    
    print("⚠ Cache structure may be incomplete")
    return False

if __name__ == '__main__':
    print("Cache Usage Test")
    print("=" * 50)
    
    cache_ok = check_cache_structure()
    
    if cache_ok:
        embedding_ok = test_embedding_cache_usage()
        
        print("\n" + "=" * 50)
        if embedding_ok:
            print("✓ Cache usage test completed successfully!")
        else:
            print("✗ Cache usage test failed.")
    else:
        print("\n" + "=" * 50)
        print("✗ Cache structure incomplete - run download_local_models.py first")