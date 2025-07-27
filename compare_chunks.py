from modules.chunker import extract_latex_sections
import re

# Read original tex file
with open('bitcoin_whitepaper.tex', 'r') as f:
    original_content = f.read()

# Extract chunks
chunks = extract_latex_sections(original_content)

# Combine all chunk content
chunk_content = '\n'.join([chunk['content'] for chunk in chunks])

# Remove LaTeX structure from original for comparison
original_clean = re.sub(r'\\documentclass.*?\\begin\{document\}', '', original_content, flags=re.DOTALL)
original_clean = re.sub(r'\\end\{document\}', '', original_clean)
original_clean = re.sub(r'\\title\{.*?\}', '', original_clean)
original_clean = re.sub(r'\\author\{.*?\}', '', original_clean)
original_clean = re.sub(r'\\date\{.*?\}', '', original_clean)
original_clean = re.sub(r'\\maketitle', '', original_clean)
original_clean = re.sub(r'% --- .+? ---', '', original_clean)
original_clean = re.sub(r'\n\s*\n', '\n', original_clean).strip()

# Compare lengths
print(f"Original content length: {len(original_clean)}")
print(f"Chunk content length: {len(chunk_content)}")
print(f"Difference: {len(original_clean) - len(chunk_content)} characters")

# Find missing content
original_words = set(original_clean.split())
chunk_words = set(chunk_content.split())

missing_words = original_words - chunk_words
extra_words = chunk_words - original_words

print(f"\nMissing words: {len(missing_words)}")
if missing_words:
    print("Sample missing:", list(missing_words)[:10])

print(f"Extra words: {len(extra_words)}")
if extra_words:
    print("Sample extra:", list(extra_words)[:10])

# Check for missing sections
original_sections = re.findall(r'% --- (.+?) ---', original_content)
chunk_sections = list(set([chunk['parent_section'] for chunk in chunks]))

print(f"\nOriginal sections: {original_sections}")
print(f"Chunk sections: {chunk_sections}")

missing_sections = set(original_sections) - set(chunk_sections)
if missing_sections:
    print(f"Missing sections: {missing_sections}")
else:
    print("All sections captured")