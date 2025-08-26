"""
Universal File Content Loader.

This module is responsible for reading the content from various file formats,
returning the most appropriate data structure for parsing by the chunker.
"""
import os
import docx # Requires pip install python-docx

def load_file_content(file_path):
    """
    Loads the content from a file, returning a string for text-based formats
    and a Document object for .docx files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at path: {file_path}")

    _, extension = os.path.splitext(file_path)
    
    if extension in ['.txt', '.tex', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif extension == '.docx':
        return docx.Document(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")

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