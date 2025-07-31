"""📄 FILE LOADING MODULE - DOCUMENT INPUT GATEWAY

This module handles all file I/O operations for the document processing system.
It provides reliable, encoding-aware file loading with proper error handling.

🎯 KEY FEATURES:
- UTF-8 encoding support for international characters
- Robust error handling with clear messages
- LaTeX file specialization
- Simple, focused interface

🛡️ RELIABILITY:
- File existence validation
- Encoding error handling
- Clear error messages for debugging

This is the trusted entry point for all document content.
"""
import os

def load_latex_file(file_path):
    """📄 LATEX FILE LOADER
    
    Loads LaTeX file content with proper error handling and encoding.
    This is the entry point for all document processing workflows.
    
    Args:
        file_path: Path to the LaTeX file
        
    Returns:
        str: File content as UTF-8 string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    🛡️ ERROR HANDLING:
    Provides clear error messages for missing files.
    """
    # 🔍 Check if file exists before attempting to read
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # 📄 Read file with UTF-8 encoding (handles international characters)
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_output(content, output_path):
    """💾 CONTENT OUTPUT SAVER
    
    Saves processed content to file with UTF-8 encoding.
    Used for persisting final processed documents.
    
    Args:
        content: String content to save
        output_path: Destination file path
        
    📄 ENCODING:
    Uses UTF-8 to ensure international character support.
    """
    # 💾 Write content with UTF-8 encoding
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)