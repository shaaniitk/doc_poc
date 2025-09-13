#!/usr/bin/env python3
"""Script to convert models to ONNX format using optimum."""

import os
import sys
from pathlib import Path

def convert_embedding_model_to_onnx():
    """Convert the sentence-transformers model to ONNX format."""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    output_dir = "./onnx_models/embedding_model"
    
    print(f"Converting {model_name} to ONNX...")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # The correct command using optimum-cli
    cmd = f'optimum-cli export onnx --model "{model_name}" --cache_dir "./models" "{output_dir}"'
    print(f"\nCommand to run:")
    print(cmd)
    
    return cmd

def convert_llm_model_to_onnx():
    """Convert the DialoGPT model to ONNX format."""
    model_name = "microsoft/DialoGPT-large"
    output_dir = "./onnx_models/llm_model"
    
    print(f"\nConverting {model_name} to ONNX...")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # The correct command using optimum-cli
    cmd = f'optimum-cli export onnx --model "{model_name}" --cache_dir "./models" "{output_dir}"'
    print(f"\nCommand to run:")
    print(cmd)
    
    return cmd

def show_usage_examples():
    """Show usage examples for ONNX conversion."""
    print("\n" + "=" * 60)
    print("ONNX Conversion Examples")
    print("=" * 60)
    
    print("\n1. Basic conversion:")
    print("   optimum-cli export onnx --model model_name output_directory")
    
    print("\n2. With cache directory (recommended):")
    print("   optimum-cli export onnx --model model_name --cache_dir ./models output_directory")
    
    print("\n3. With specific input shapes:")
    print("   optimum-cli export onnx --model model_name --sequence_length 512 --batch_size 1 output_directory")
    
    print("\n4. With optimization:")
    print("   optimum-cli export onnx --model model_name --optimize O2 output_directory")
    
    print("\nNote: Replace 'your_model_name' with actual model names like:")
    print("- sentence-transformers/all-mpnet-base-v2")
    print("- microsoft/DialoGPT-large")
    print("- Or any HuggingFace model identifier")

if __name__ == '__main__':
    print("ONNX Model Conversion Script")
    print("=" * 40)
    
    # Show the correct commands for your project models
    embedding_cmd = convert_embedding_model_to_onnx()
    llm_cmd = convert_llm_model_to_onnx()
    
    show_usage_examples()
    
    print("\n" + "=" * 60)
    print("Ready to Convert Your Models")
    print("=" * 60)
    
    print("\nTo convert your project models, run these commands:")
    print(f"\n1. Embedding model:")
    print(f"   {embedding_cmd}")
    
    print(f"\n2. LLM model:")
    print(f"   {llm_cmd}")
    
    print("\nNote: The conversion will use your existing ./models cache folder.")
    print("This prevents re-downloading models during conversion.")