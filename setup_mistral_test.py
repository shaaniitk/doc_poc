#!/usr/bin/env python3
"""Setup script for Mistral API integration testing."""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3.7):
        print("❌ Python 3.7+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies for Mistral testing."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Check if aiohttp is already installed
        import aiohttp
        print("✅ aiohttp already installed")
        return True
    except ImportError:
        pass
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "aiohttp>=3.8.0"
        ])
        print("✅ aiohttp installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install aiohttp: {e}")
        print("💡 Try running: pip install aiohttp")
        return False

def check_files():
    """Check if required files exist."""
    print("\n📁 Checking required files...")
    
    required_files = [
        "mistral_llm_handler.py",
        "test_mistral_integration.py",
        "run_mistral_test.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_core_dependencies():
    """Check if core application files exist."""
    print("\n🔍 Checking core dependencies...")
    
    core_files = [
        "core/models.py",
        "core/template_processor_node.py",
        "core/semantic_mapper.py"
    ]
    
    available_files = []
    for file_path in core_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
            available_files.append(file_path)
        else:
            print(f"⚠️  {file_path} - Not found (may limit testing)")
    
    return len(available_files) > 0

def main():
    print("🔧 Mistral API Integration Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed - dependency installation error")
        return
    
    # Check required files
    if not check_files():
        print("\n❌ Setup incomplete - missing required files")
        print("💡 Make sure you're in the correct directory")
        return
    
    # Check core dependencies
    core_available = check_core_dependencies()
    
    print("\n🎉 Setup Complete!")
    print("=" * 20)
    
    if core_available:
        print("✅ Full integration testing available")
        print("\n🚀 Next steps:")
        print("   1. Set your MISTRAL_API_KEY environment variable")
        print("   2. Run: python run_mistral_test.py")
    else:
        print("⚠️  Limited testing available (core files missing)")
        print("\n🚀 Next steps:")
        print("   1. Set your MISTRAL_API_KEY environment variable")
        print("   2. Run: python test_mistral_integration.py (basic test only)")
    
    print("\n💰 Cost: < $0.005 USD for full test suite")
    print("📊 API calls: Maximum 2 calls")

if __name__ == "__main__":
    main()