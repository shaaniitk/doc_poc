import os
import re
import sys

# --- CONFIGURATION ---
DOCUMENT_SKELETON = [
    {"section": "Title", "content": "", "prompt": "Create a professional, academic title for a document about a new signal processing technique called DAWF."},
    {"section": "Author", "content": "", "prompt": "Provide placeholder author names and affiliations, like 'Dr. Evelyn Reed, Quantum Signal Labs'."},
    {"section": "Abstract", "content_type": "prose", "prompt": "Rewrite the following text into a concise, professional abstract of about 150 words. It should summarize the problem, the proposed method (DAWF), and the key results."},
    {"section": "1. Introduction", "content_type": "prose", "prompt": "Rewrite the following text into a compelling introduction. It should cover the motivation for the work, the core problem statement, and the scope and objectives of the document."},
    {"section": "2. Theoretical Foundations", "content_type": "prose_and_equations", "prompt": "Using the following text and equations, write a clear section on the theoretical foundations of signal processing. Explain concepts like DTFT, Z-Transform, and stochastic processes in a logical flow, ensuring all provided LaTeX equations are preserved exactly as they are."},
    {"section": "3. The Proposed Framework: DAWF", "content_type": "prose_and_equations", "prompt": "From the following text, describe the proposed Dynamic Adaptive Wavelet Filter (DAWF). Detail its architecture, mathematical model, and key innovations. Preserve all LaTeX equations perfectly."},
    {"section": "4. Input Data and Database", "content_type": "prose_and_tables", "prompt": "Write a section describing the data sources, pre-processing steps, and database structure using the text below. Preserve any LaTeX tables exactly and integrate them smoothly into the text."},
    {"section": "5. Implementation Details", "content_type": "prose_and_code", "prompt": "Detail the implementation of the DAWF using the text provided. Mention the language, libraries, and codebase structure. Preserve the verbatim code block and any included image figures exactly."},
    {"section": "6. Testing and Verification", "content_type": "prose_tables_and_figures", "prompt": "Create a section on testing and verification from the following text. It should cover sanity, unit, and integration tests. Preserve all LaTeX tables and included image figures exactly as provided."},
    {"section": "7. Experimental Results and Discussion", "content_type": "prose_and_plots", "prompt": "Analyze and structure the provided text and plots into a results and discussion section. It is critical that you preserve the LaTeX pgfplots figures exactly as they are. You can add text to introduce and discuss each plot."},
    {"section": "8. Conclusion", "content_type": "prose", "prompt": "Rewrite the following text into a strong concluding section. It should summarize the key contributions, limitations, and future work."},
    {"section": "References", "content_type": "bibliography", "prompt": "Format the following content as a standard LaTeX 'thebibliography' environment. Preserve all bibitems and their content perfectly."}
]

# --- IMPROVED CHUNKING ---
def extract_latex_sections(file_path):
    """Extract sections from LaTeX document using regex patterns"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = []
    
    # Extract comment-based sections (% --- Section Name ---)
    section_pattern = r'% --- (.+?) ---\n(.*?)(?=% ---|\\begin\{thebibliography\}|\\end\{document\}|$)'
    sections = re.findall(section_pattern, content, re.DOTALL)
    
    for section_name, section_content in sections:
        section_content = section_content.strip()
        if not section_content:
            continue
        
        # Extract environments and text blocks
        parts = extract_content_parts(section_content, section_name)
        chunks.extend(parts)
    
    # Extract bibliography separately
    bib_pattern = r'(\\begin\{thebibliography\}.*?\\end\{thebibliography\})'
    bib_match = re.search(bib_pattern, content, re.DOTALL)
    if bib_match:
        chunks.append({
            'type': 'bibliography',
            'content': bib_match.group(1),
            'parent_section': 'References'
        })
    
    return chunks

def extract_content_parts(content, section_name):
    """Extract text paragraphs and LaTeX environments from content"""
    # First, find and extract complete table blocks (including labels)
    table_pattern = r'(\\begin\{table\}.*?\\end\{table\}(?:\s*\\label\{[^}]+\})?)'    
    tables = re.findall(table_pattern, content, re.DOTALL)
    
    # Replace tables with placeholders
    temp_content = content
    for i, table in enumerate(tables):
        temp_content = temp_content.replace(table, f"__TABLE_PLACEHOLDER_{i}__")
    
    # Now extract other environments from the modified content
    env_pattern = r'(\\begin\{[^}]+\}.*?\\end\{[^}]+\})'    
    segments = re.split(env_pattern, temp_content, flags=re.DOTALL)
    
    parts = []
    current_text = ""
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # Check for table placeholder
        if "__TABLE_PLACEHOLDER_" in segment:
            if current_text.strip():
                parts.append({
                    'type': 'paragraph',
                    'content': current_text.strip(),
                    'parent_section': section_name
                })
                current_text = ""
            
            # Extract table index and add complete table
            match = re.search(r'__TABLE_PLACEHOLDER_(\d+)__', segment)
            if match:
                table_idx = int(match.group(1))
                parts.append({
                    'type': 'table',
                    'content': tables[table_idx],
                    'parent_section': section_name
                })
        elif re.match(r'\\begin\{', segment):
            if current_text.strip():
                parts.append({
                    'type': 'paragraph',
                    'content': current_text.strip(),
                    'parent_section': section_name
                })
                current_text = ""
            
            env_match = re.match(r'\\begin\{([^}]+)\}', segment)
            env_type = env_match.group(1) if env_match else 'environment'
            
            parts.append({
                'type': env_type,
                'content': segment,
                'parent_section': section_name
            })
        else:
            if "__TABLE_PLACEHOLDER_" not in segment:
                current_text += " " + segment
    
    if current_text.strip():
        parts.append({
            'type': 'paragraph',
            'content': current_text.strip(),
            'parent_section': section_name
        })
    
    return parts

def group_chunks_by_section(chunks):
    """Group chunks by their parent section"""
    grouped = {}
    for chunk in chunks:
        section = chunk['parent_section']
        if section not in grouped:
            grouped[section] = []
        grouped[section].append(chunk)
    return grouped

def assign_chunks_to_skeleton(grouped_chunks, skeleton):
    """Assign grouped chunks to skeleton sections using simple mapping"""
    assignments = {s['section']: [] for s in skeleton}
    
    # Simple mapping based on section names
    section_mapping = {
        'Abstract': 'Abstract',
        'Introduction': '1. Introduction', 
        'Theoretical Foundations': '2. Theoretical Foundations',
        'The Proposed Framework: DAWF': '3. The Proposed Framework: DAWF',
        'Input Data and Database': '4. Input Data and Database',
        'Implementation Details': '5. Implementation Details',
        'Testing and Verification': '6. Testing and Verification',
        'Experimental Results and Discussion': '7. Experimental Results and Discussion',
        'Conclusion': '8. Conclusion',
        'References': 'References'
    }
    
    for source_section, target_section in section_mapping.items():
        if source_section in grouped_chunks and target_section in assignments:
            # Combine all chunks from this section into one content block
            combined_content = []
            for chunk in grouped_chunks[source_section]:
                combined_content.append(chunk['content'])
            
            assignments[target_section].append({
                'content': '\n\n'.join(combined_content),
                'parent_section': source_section
            })
    
    return assignments

def log(msg):
    with open("chunking_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log file at start
with open("chunking_log.txt", "w", encoding="utf-8") as f:
    f.write("")

def main():
    log("=== FIXED CHUNKING: Improved regex-based parsing ===")
    source_file = "unstructured_document.tex"
    log(f"Reading source file: {source_file}")
    
    # Extract chunks using improved method
    chunks = extract_latex_sections(source_file)
    log(f"[DEBUG] Extracted {len(chunks)} chunks from LaTeX document.")
    
    for i, c in enumerate(chunks[:10]):
        log(f"[DEBUG] Chunk {i}: type={c.get('type', '')}, parent_section={c.get('parent_section', '')}")
    
    if not chunks:
        log("[WARNING] No chunks extracted from LaTeX document!")
        return
    
    # Group chunks by section
    grouped_chunks = group_chunks_by_section(chunks)
    log(f"[DEBUG] Grouped into {len(grouped_chunks)} sections: {list(grouped_chunks.keys())}")
    
    # Assign to skeleton
    assignments = assign_chunks_to_skeleton(grouped_chunks, DOCUMENT_SKELETON)
    
    # Write results
    md_lines = ["# Fixed Chunking Results\n"]
    for section_name, section_chunks in assignments.items():
        md_lines.append(f"## {section_name}")
        if section_chunks:
            for i, chunk in enumerate(section_chunks):
                md_lines.append(f"### Chunk {i}")
                content = chunk.get('content', '')
                if len(content) > 1000:
                    preview = content[:1000] + "..."
                else:
                    preview = content
                md_lines.append(f"**Content:**\n```\n{preview}\n```\n")
        else:
            md_lines.append("*No content assigned to this section*\n")
        md_lines.append("---\n")
    
    with open("fixed_chunked_output.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    
    log("[INFO] Fixed chunked output written to fixed_chunked_output.md")
    print("Fixed chunking complete! Check fixed_chunked_output.md for results.")

if __name__ == "__main__":
    main()