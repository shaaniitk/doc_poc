# ONNX Model Conversion Guide

## Problem Solved

The original command you tried:
```bash
python -m optimum.onnxruntime.utils.convert --model your_model_name --output model.onnx
```

Failed with: `ModuleNotFoundError: __path__ attribute not found on 'optimum.onnxruntime.utils'`

## Solution: Use optimum-cli

The correct way to convert models to ONNX format is using the `optimum-cli` command:

### Basic Syntax
```bash
optimum-cli export onnx --model MODEL_NAME OUTPUT_DIRECTORY
```

### Recommended Usage (with cache)
```bash
optimum-cli export onnx --model MODEL_NAME --cache_dir ./models OUTPUT_DIRECTORY
```

## Your Project Models

### 1. Embedding Model Conversion
```bash
optimum-cli export onnx --model "sentence-transformers/all-mpnet-base-v2" --cache_dir "./models" "./onnx_models/embedding_model"
```

**Result**: Successfully created ONNX files in `./onnx_models/embedding_model/`:
- `model.onnx` (436MB) - The main ONNX model
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer configuration
- `vocab.txt` - Vocabulary file
- `special_tokens_map.json` - Special tokens mapping
- `tokenizer_config.json` - Tokenizer settings

### 2. LLM Model Conversion
```bash
optimum-cli export onnx --model "microsoft/DialoGPT-large" --cache_dir "./models" "./onnx_models/llm_model"
```

## Advanced Options

### With Optimization
```bash
optimum-cli export onnx --model MODEL_NAME --optimize O2 OUTPUT_DIRECTORY
```

### With Custom Input Shapes
```bash
optimum-cli export onnx --model MODEL_NAME --sequence_length 512 --batch_size 1 OUTPUT_DIRECTORY
```

### With Quantization
```bash
optimum-cli export onnx --model MODEL_NAME --quantize OUTPUT_DIRECTORY
```

## Benefits of Using Cache Directory

- **Faster conversion**: Uses already downloaded models from `./models`
- **No re-downloading**: Saves bandwidth and time
- **Consistent versions**: Uses the same model versions as your runtime code

## Warning Explanation

The warning you see:
```
OnnxExporterWarning: Symbolic function 'aten::scaled_dot_product_attention' already registered for opset 14
```

This is a known PyTorch ONNX export warning and doesn't affect the conversion quality. It's safe to ignore.

## Using ONNX Models

Once converted, you can use ONNX models with:

### Python Code
```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

# Load ONNX model
model = ORTModelForFeatureExtraction.from_pretrained("./onnx_models/embedding_model")
tokenizer = AutoTokenizer.from_pretrained("./onnx_models/embedding_model")

# Use the model
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

### Performance Benefits
- **Faster inference**: ONNX Runtime optimizations
- **Lower memory usage**: Optimized model format
- **Cross-platform**: Works on different hardware
- **GPU acceleration**: Better GPU utilization

## Files Created

- `convert_to_onnx.py` - Helper script with conversion commands
- `./onnx_models/embedding_model/` - ONNX version of sentence-transformers model
- This guide - `ONNX_CONVERSION_GUIDE.md`

## Next Steps

1. Convert your LLM model if needed:
   ```bash
   optimum-cli export onnx --model "microsoft/DialoGPT-large" --cache_dir "./models" "./onnx_models/llm_model"
   ```

2. Update your code to use ONNX models for better performance

3. Test the ONNX models with your existing pipeline

4. Consider quantization for even better performance:
   ```bash
   optimum-cli export onnx --model MODEL_NAME --quantize OUTPUT_DIRECTORY
   ```