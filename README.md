# Hugging Face Model Caller

A Python toolkit for calling GPT-4o-like models and Mistral 7B from Hugging Face using both API-based and local approaches.

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

2. **Set Hugging Face Token** (Optional but recommended):
   ```bash
   export HUGGING_FACE_TOKEN="your_token_here"
   ```
   Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)

3. **Run Demo**:
   ```bash
   python3 hf_mcp_caller.py
   ```

4. **Interactive Chat**:
   ```bash
   python3 hf_mcp_caller.py --chat
   ```

## 📋 Available Scripts

### 1. `hf_mcp_caller.py` - MCP-Style API Caller
- Uses Hugging Face Inference API
- Lightweight and fast
- No local model storage required
- Perfect for quick testing and demos

**Features**:
- Direct API calls to Hugging Face models
- Multiple model support
- Interactive chat mode
- Error handling and retries

### 2. `hf_model_caller.py` - Local Model Runner
- Downloads and runs models locally
- More control over generation
- Works offline after initial download
- Better for production use

**Features**:
- Local model loading with transformers
- Pipeline and manual inference modes
- GPU acceleration support
- Advanced generation parameters

## 🤖 Available Models

| Model Name | Model ID | Description |
|------------|----------|-------------|
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | Latest Mistral 7B instruction model |
| `openhermes` | `teknium/OpenHermes-2.5-Mistral-7B` | Fine-tuned Mistral for helpful responses |
| `gpt4o-like` | `TommyZQ/GPT-4o` | GPT-4o alternative model |
| `mistral-base` | `mistralai/Mistral-7B-v0.1` | Original Mistral 7B base model |

## 📚 Usage Examples

### Basic API Call
```python
from hf_mcp_caller import HuggingFaceMCPCaller

caller = HuggingFaceMCPCaller()
response = caller.chat(
    "mistral-7b", 
    "What are the benefits of renewable energy?",
    "You are an environmental expert."
)
print(response)
```

### Local Model Usage
```python
from hf_model_caller import HuggingFaceModelCaller

caller = HuggingFaceModelCaller()
caller.load_model("mistral-7b")
response = caller.chat_with_model(
    "mistral-7b",
    "Explain quantum computing simply."
)
print(response)
```

### Custom Parameters
```python
response = caller.generate_text(
    "mistral-7b",
    "Write a short story about AI:",
    temperature=0.8,
    max_new_tokens=200,
    top_p=0.9
)
```

## ⚙️ Configuration

Edit `config.py` to:
- Add new models
- Change default parameters
- Modify API settings

```python
MODELS = {
    "your-model": {
        "model_id": "username/model-name",
        "description": "Your custom model",
        "recommended_params": {
            "temperature": 0.7,
            "max_new_tokens": 150
        }
    }
}
```

## 🛠️ Requirements

- Python 3.8+
- Internet connection (for API mode)
- GPU recommended (for local mode)
- Hugging Face account (for higher rate limits)

**Dependencies**:
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for model inference
- `requests` - HTTP requests for API calls
- `accelerate` - Model acceleration
- `safetensors` - Safe tensor format

## 🔧 Troubleshooting

### Model Loading Issues
- **503 Error**: Model is loading, wait a few minutes
- **Rate Limit**: Set `HUGGING_FACE_TOKEN` environment variable
- **Memory Error**: Try smaller models or reduce batch size

### GPU Issues
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-compatible PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### API Issues
- Verify internet connection
- Check Hugging Face status: https://status.huggingface.co/
- Ensure model exists and is publicly accessible

## 📁 File Structure

```
doc_poc/
├── hf_mcp_caller.py      # MCP-style API caller
├── hf_model_caller.py    # Local model runner
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── setup.sh             # Setup script
└── README.md            # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different models
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🔗 Links

- [Hugging Face Hub](https://huggingface.co/models)
- [Mistral AI](https://mistral.ai/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## 💡 Tips

- Use API mode for quick testing and demos
- Use local mode for production and offline use
- Set temperature lower (0.3-0.5) for factual responses
- Set temperature higher (0.8-1.0) for creative responses
- Use system messages to guide model behavior
- Monitor token usage to stay within limits

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review the Hugging Face model page
3. Check model compatibility with transformers
4. Verify your token permissions

---
*Built with ❤️ using Hugging Face and the Model Context Protocol*
