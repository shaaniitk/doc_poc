import re

# Read original and final documents
with open('bitcoin_whitepaper.tex', 'r') as f:
    original = f.read()

with open('outputs/20250726_110558/final_document.tex', 'r') as f:
    final = f.read()

# Clean both for comparison
def clean_content(text):
    # Remove LaTeX structure
    text = re.sub(r'\\documentclass.*?\\begin\{document\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\end\{document\}', '', text)
    text = re.sub(r'\\title\{.*?\}', '', text)
    text = re.sub(r'\\author\{.*?\}', '', text)
    text = re.sub(r'\\date\{.*?\}', '', text)
    text = re.sub(r'\\maketitle', '', text)
    text = re.sub(r'% --- .+? ---', '', text)
    text = re.sub(r'% .+', '', text)  # Remove comments
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    return text

original_clean = clean_content(original)
final_clean = clean_content(final)

print(f"Original length: {len(original_clean)} chars")
print(f"Final length: {len(final_clean)} chars")
print(f"Loss: {len(original_clean) - len(final_clean)} chars ({((len(original_clean) - len(final_clean))/len(original_clean)*100):.1f}%)")

# Check specific content types
equations_orig = len(re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', original, re.DOTALL))
equations_final = len(re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', final, re.DOTALL))

verbatim_orig = len(re.findall(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', original, re.DOTALL))
verbatim_final = len(re.findall(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', final, re.DOTALL))

enumerate_orig = len(re.findall(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', original, re.DOTALL))
enumerate_final = len(re.findall(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', final, re.DOTALL))

print(f"\nContent Analysis:")
print(f"Equations: {equations_orig} → {equations_final} (lost: {equations_orig - equations_final})")
print(f"Code blocks: {verbatim_orig} → {verbatim_final} (lost: {verbatim_orig - verbatim_final})")
print(f"Lists: {enumerate_orig} → {enumerate_final} (lost: {enumerate_orig - enumerate_final})")

# Check for specific missing content
missing_content = []

# Check for mathematical formulas
if "q_z = \\begin{cases}" not in final:
    missing_content.append("Probability equations with cases")

if "AttackerSuccessProbability" not in final:
    missing_content.append("C code implementation")

if "\\lambda = z \\frac{q}{p}" not in final:
    missing_content.append("Lambda equation")

if "Poisson distribution" not in final:
    missing_content.append("Poisson distribution discussion")

# Check sections
sections_orig = re.findall(r'% --- (.+?) ---', original)
sections_final = re.findall(r'% (.+)', final)

print(f"\nSection Analysis:")
print(f"Original sections: {sections_orig}")
print(f"Final sections: {sections_final}")

if missing_content:
    print(f"\nMissing Content:")
    for item in missing_content:
        print(f"- {item}")
else:
    print(f"\nNo major content missing detected")