"""File loading module"""
import os

def load_latex_file(file_path):
    """Load LaTeX file content"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_output(content, output_path):
    """Save content to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)