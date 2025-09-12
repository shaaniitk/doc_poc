"""Local embedding configuration using Hugging Face models.

This module provides optimized configurations for running embeddings locally
without requiring OpenAI API calls.
"""

import os
from typing import Dict, Any

# Local embedding model configurations
LOCAL_EMBEDDING_MODELS = {
    'lightweight': {
        'model': 'sentence-transformers/all-MiniLM-L6-v2',
        'provider': 'sentence_transformer',
        'description': 'Fast and lightweight, good for development and testing',
        'size': '90MB',
        'dimensions': 384,
        'performance': 'Fast',
        'quality': 'Good'
    },
    'balanced': {
        'model': 'sentence-transformers/all-mpnet-base-v2',
        'provider': 'sentence_transformer',
        'description': 'Balanced speed and quality, recommended for most use cases',
        'size': '420MB',
        'dimensions': 768,
        'performance': 'Medium',
        'quality': 'Very Good'
    },
    'high_quality': {
        'model': 'BAAI/bge-large-en-v1.5',
        'provider': 'sentence_transformer',
        'description': 'High quality embeddings, best alternative to OpenAI',
        'size': '1.3GB',
        'dimensions': 1024,
        'performance': 'Slower',
        'quality': 'Excellent'
    },
    'e5_large': {
        'model': 'intfloat/e5-large-v2',
        'provider': 'sentence_transformer',
        'description': 'Excellent for document retrieval and semantic search',
        'size': '1.3GB',
        'dimensions': 1024,
        'performance': 'Slower',
        'quality': 'Excellent'
    }
}

# Default local configuration (balanced option)
DEFAULT_LOCAL_CONFIG = LOCAL_EMBEDDING_MODELS['balanced'].copy()

# Local semantic mapping configuration
LOCAL_SEMANTIC_MAPPING_CONFIG = {
    'provider': 'sentence_transformer',
    'model': DEFAULT_LOCAL_CONFIG['model'],
    'batch_size': 32,
    'max_length': 512,
    'normalize_embeddings': True,
    'device': 'auto',  # Will use GPU if available, otherwise CPU
    'cache_folder': './models/embeddings',  # Local cache for models
    'trust_remote_code': False,  # Security setting
}

# Local chunking embedding configuration
LOCAL_CHUNKING_EMBEDDING = {
    'enable': True,
    'provider': 'sentence_transformer',
    'model': DEFAULT_LOCAL_CONFIG['model'],
    'similarity_threshold': 0.7,
    'batch_size': 16,  # Smaller batch for chunking
    'cache_embeddings': True,
    'cache_folder': './cache/embeddings'
}

def get_local_config(model_type: str = 'balanced') -> Dict[str, Any]:
    """Get local embedding configuration for specified model type.
    
    Args:
        model_type: One of 'lightweight', 'balanced', 'high_quality', 'e5_large'
        
    Returns:
        Configuration dictionary for the specified model
    """
    if model_type not in LOCAL_EMBEDDING_MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(LOCAL_EMBEDDING_MODELS.keys())}")
    
    base_config = LOCAL_EMBEDDING_MODELS[model_type].copy()
    
    # Add common configuration
    base_config.update({
        'batch_size': 32,
        'max_length': 512,
        'normalize_embeddings': True,
        'device': 'auto',
        'cache_folder': './models/embeddings',
        'trust_remote_code': False,
    })
    
    return base_config

def get_hardware_recommendation() -> str:
    """Get hardware-based model recommendation.
    
    Returns:
        Recommended model type based on available hardware
    """
    try:
        import psutil
        import torch
        
        # Check available RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Check if CUDA is available
        has_gpu = torch.cuda.is_available()
        
        if ram_gb >= 16 and has_gpu:
            return 'high_quality'  # Can handle large models with GPU
        elif ram_gb >= 8:
            return 'balanced'      # Good balance for most systems
        else:
            return 'lightweight'   # Conservative for low-memory systems
            
    except ImportError:
        # If psutil or torch not available, default to balanced
        return 'balanced'

def create_local_env_file(model_type: str = None) -> str:
    """Create a .env file for local configuration.
    
    Args:
        model_type: Model type to use, or None for auto-detection
        
    Returns:
        Path to created .env file
    """
    if model_type is None:
        model_type = get_hardware_recommendation()
    
    config = get_local_config(model_type)
    
    env_content = f"""# Local Embedding Configuration
# Generated automatically for model type: {model_type}

# Disable OpenAI (we're running locally)
OPENAI_API_KEY=

# Local embedding settings
EMBEDDING_PROVIDER=sentence_transformer
EMBEDDING_MODEL={config['model']}
EMBEDDING_BATCH_SIZE={config['batch_size']}
EMBEDDING_CACHE_FOLDER={config['cache_folder']}

# Performance settings
EMBEDDING_DEVICE={config['device']}
EMBEDDING_MAX_LENGTH={config['max_length']}
EMBEDDING_NORMALIZE=true

# Model info (for reference)
# Model: {config['description']}
# Size: {config['size']}
# Dimensions: {config['dimensions']}
# Performance: {config['performance']}
# Quality: {config['quality']}
"""
    
    env_path = '.env.local'
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    return env_path

if __name__ == '__main__':
    # Demo the configuration
    print("Available local embedding models:")
    for name, config in LOCAL_EMBEDDING_MODELS.items():
        print(f"\n{name.upper()}:")
        print(f"  Model: {config['model']}")
        print(f"  Description: {config['description']}")
        print(f"  Size: {config['size']}")
        print(f"  Quality: {config['quality']}")
        print(f"  Performance: {config['performance']}")
    
    print(f"\nRecommended model for your hardware: {get_hardware_recommendation()}")
    
    # Create local env file
    env_file = create_local_env_file()
    print(f"\nCreated local configuration file: {env_file}")