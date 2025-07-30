#!/bin/bash

# Setup script for Hugging Face Model Caller
echo "🚀 Setting up Hugging Face Model Caller..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Check if Hugging Face token is set
if [ -z "$HUGGING_FACE_TOKEN" ]; then
    echo ""
    echo "⚠️  HUGGING_FACE_TOKEN environment variable is not set."
    echo "   You can still use the script, but may encounter rate limits."
    echo ""
    echo "   To set your token:"
    echo "   export HUGGING_FACE_TOKEN='your_token_here'"
    echo ""
    echo "   Get your token from: https://huggingface.co/settings/tokens"
    echo ""
fi

echo "✅ Setup complete!"
echo ""
echo "📋 Usage:"
echo "   source venv/bin/activate  # Activate environment"
echo "   python3 hf_mcp_caller.py  # Run demo"
echo "   python3 hf_mcp_caller.py --chat  # Interactive chat"
echo "   python3 hf_model_caller.py  # Full local model runner"
echo ""
