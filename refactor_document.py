import os
import re
from dotenv import load_dotenv
from plasTeX.TeX import TeX
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer
import requests
import sys

# --- CONFIGURATION ---
load_dotenv()

MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    #"llama-3": "meta-llama/Meta-Llama-3-8B-Instruct"
    # "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
    # "falcon-7b": "tiiuae/falcon-7b-instruct",
    # "gemma-2b": "google/gemma-2b-it"
}

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

# --- ADVANCED CHUNKING ---
def parse_latex_ast(file_path):
    tex = TeX()
    with open(file_path, 'r', encoding='utf-8') as f:
        tex.input(f.read())
    doc = tex.parse()
    # Debug: print top-level nodes/tags in AST
    print("[DEBUG] Top-level nodes in parsed AST:")
    if hasattr(doc, 'childNodes'):
        for i, node in enumerate(doc.childNodes):
            print(f"  Node {i}: tagname={getattr(node, 'tagname', None)}, type={type(node)}")
    else:
        print("  [WARNING] doc has no childNodes attribute!")
    return doc

def extract_chunks_from_ast(doc):
    # Extract sections, subsections, environments, and paragraphs
    # Returns a list of dicts: {type, title, content, parent_section}
    chunks = []
    def walk(node, parent_section=None):
        # If this is the document node, traverse its children
        if hasattr(node, 'tagname') and node.tagname == 'document':
            for child in node.childNodes:
                walk(child, parent_section='document')
        # Sectioning commands (if present)
        elif hasattr(node, 'tagname') and node.tagname in ['section', 'subsection', 'subsubsection']:
            title = str(node.attributes.get('title', ''))
            content = node.source
            chunks.append({'type': node.tagname, 'title': title, 'content': content, 'parent_section': parent_section})
            parent_section = title
        # Environments (expanded to include more types)
        elif hasattr(node, 'tagname') and node.tagname in [
            'equation', 'align', 'verbatim', 'table', 'figure', 'pgfplots', 'lstlisting', 'tikzpicture', 'tabular', 'algorithm', 'code', 'minted']:
            content = node.source
            print(f"[DEBUG] Extracted environment: {node.tagname} (parent_section={parent_section})")
            chunks.append({'type': node.tagname, 'title': '', 'content': content, 'parent_section': parent_section})
        # Text blocks (paragraphs)
        elif hasattr(node, 'textContent') and node.textContent.strip():
            # Only add non-empty text
            content = node.textContent.strip()
            if content:
                chunks.append({'type': 'paragraph', 'title': '', 'content': content, 'parent_section': parent_section})
        # Recurse into children
        if hasattr(node, 'childNodes'):
            for child in node.childNodes:
                walk(child, parent_section)
    walk(doc)
    return chunks

def group_paragraphs_and_environments(chunks):
    # Group paragraphs with their following environments
    grouped = []
    buffer = []
    for chunk in chunks:
        if chunk['type'] in ['section', 'subsection', 'subsubsection']:
            if buffer:
                grouped.append({'content': '\n'.join([c['content'] for c in buffer]), 'parent_section': buffer[0]['parent_section']})
                buffer = []
            grouped.append({'content': chunk['content'], 'parent_section': chunk['parent_section']})
        else:
            buffer.append(chunk)
    if buffer:
        grouped.append({'content': '\n'.join([c['content'] for c in buffer]), 'parent_section': buffer[0]['parent_section']})
    return grouped

def char_sliding_window_chunk(group, max_chars=4000, overlap=500):
    content = group['content']
    windows = []
    start = 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        chunk_text = content[start:end]
        windows.append({'content': chunk_text, 'parent_section': group['parent_section']})
        if end == len(content):
            break
        start += max_chars - overlap
    return windows

def embed_and_assign_chunks(groups, skeleton, embedder):
    # Embed each group and each skeleton prompt, assign by cosine similarity
    group_embs = embedder.encode([g['content'] for g in groups])
    section_embs = embedder.encode([s['prompt'] for s in skeleton])
    assignments = {s['section']: [] for s in skeleton}
    for i, g_emb in enumerate(group_embs):
        sims = util.cos_sim(g_emb, section_embs)[0]
        best_idx = int(sims.argmax())
        assignments[skeleton[best_idx]['section']].append(groups[i])
    # Hybrid fallback: if any section is empty or all content is routed to one section, use round-robin
    empty_sections = [k for k, v in assignments.items() if len(v) == 0]
    all_in_one = any(len(v) == len(groups) for v in assignments.values())
    if empty_sections or all_in_one:
        print("[DEBUG] Hybrid fallback: using round-robin assignment to ensure all sections are populated.")
        assignments = {s['section']: [] for s in skeleton}
        n_sections = len(skeleton)
        for i, group in enumerate(groups):
            section = skeleton[i % n_sections]['section']
            assignments[section].append(group)
    return assignments

def token_count(text, tokenizer):
    return len(tokenizer.encode(text))

def split_chunk(chunk, model_limit, tokenizer):
    # Split chunk['content'] into pieces that fit model_limit tokens
    tokens = tokenizer.encode(chunk['content'])
    splits = []
    start = 0
    while start < len(tokens):
        end = min(start + model_limit, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        splits.append({'content': chunk_text, 'parent_section': chunk['parent_section']})
        start = end
    return splits

def call_mistral_api(prompt, model="mistral-small-latest", max_tokens=2048):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set in environment or .env file.")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"[ERROR] Mistral API call failed: {response.status_code} {response.text}")
        return f"% Mistral API call failed: {response.status_code} {response.text}"

# --- LLM INTERACTION WITH RUNNING SUMMARY ---
def get_llm_response(pipe, prompt, content, section_title, doc_summary=None, section_summary=None):
    # Compose context
    context = ""
    if doc_summary:
        context += f"DOCUMENT SUMMARY SO FAR:\n{doc_summary}\n\n"
    if section_summary:
        context += f"SECTION SUMMARY SO FAR:\n{section_summary}\n\n"
    # Updated prompt: explicitly preserve all LaTeX environments
    full_prompt = context + f"TASK: {prompt} Preserve all LaTeX tables, figures, code, and plot environments (such as table, figure, tabular, pgfplots, tikzpicture, algorithm, minted, verbatim, etc.) exactly as they are. Do not modify or omit them.\n\nCONTENT TO REWRITE:\n---\n{content}\n---"
    system_prompt = ("You are a professional technical writer and LaTeX expert. "
        "Your output should be only the raw LaTeX content for the requested section, without any additional explanations, comments, or markdown formatting. "
        "Your output must be in a highly scientific and academic tone, suitable for publication in a peer-reviewed journal.")
    print(f"\n[DEBUG] LLM INPUT for section '{section_title}':\n{'='*40}\n{full_prompt[:1000]}\n{'='*40}")
    try:
        # Comment out Hugging Face pipeline call
        # result = pipe(f"{system_prompt}\n\n{full_prompt}", truncation=True)
        # output = result[0]['generated_text']
        # Use Mistral API instead
        output = call_mistral_api(f"{system_prompt}\n\n{full_prompt}")
        print(f"[DEBUG] LLM OUTPUT for section '{section_title}':\n{'-'*40}\n{output[:1000]}\n{'-'*40}")
        return output
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return "% LLM call failed: {e}"

def update_summary(summary, new_content, pipe, max_tokens=256):
    # Summarize new_content and append to summary
    try:
        result = pipe(f"Summarize the following for context (max {max_tokens} tokens):\n{new_content}", max_new_tokens=max_tokens, truncation=True)
        return summary + "\n" + result[0]['generated_text']
    except Exception:
        return summary

try:
    import tiktoken
    def count_tokens(text, model_name="gpt-3.5-turbo"):
        enc = tiktoken.encoding_for_model(model_name)
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text, model_name=None):
        return len(text.split())  # fallback: word count

def hybrid_chunking(chunks, max_tokens=1800, overlap=200):
    final_chunks = []
    for chunk in chunks:
        token_estimate = count_tokens(chunk['content'])
        if token_estimate > max_tokens:
            # Sliding window by tokens (approximate)
            words = chunk['content'].split()
            start = 0
            while start < len(words):
                end = min(start + max_tokens, len(words))
                chunk_text = ' '.join(words[start:end])
                final_chunks.append({'content': chunk_text, 'parent_section': chunk['parent_section']})
                if end == len(words):
                    break
                start += max_tokens - overlap
        else:
            final_chunks.append(chunk)
    return final_chunks

def llm_classify_section(chunk_content, section_names, call_llm_fn):
    prompt = (
        "Given the following text from a scientific/technical document, "
        "which section does it best belong to? Choose the single best match from this list: "
        f"{', '.join(section_names)}.\n"
        "Return only the section name, nothing else.\n"
        "\nTEXT:\n" + chunk_content[:1000] + "\n"
    )
    section = call_llm_fn(prompt)
    section = section.strip().split("\n")[0]
    return section

def llm_split_and_assign_sections(chunk_content, section_names, call_llm_fn):
    prompt = (
        "Given the following text from a scientific/technical document, split it into parts and assign each part to the most appropriate section from this list: "
        f"{', '.join(section_names)}.\n"
        "For each part, output the section name on a line by itself, followed by the corresponding text.\n"
        "Format:\nSection: <section name>\n<text>\nSection: <section name>\n<text>\n...\n"
        "If a section is not relevant, do not include it.\n"
        "\nTEXT:\n" + chunk_content[:2000] + "\n"
    )
    response = call_llm_fn(prompt)
    # Parse output: look for 'Section: <section name>' lines
    assignments = {s: [] for s in section_names}
    current_section = None
    current_lines = []
    for line in response.splitlines():
        if line.strip().startswith('Section:'):
            if current_section and current_lines:
                assignments[current_section].append('\n'.join(current_lines).strip())
            section_candidate = line.strip().replace('Section:', '').strip()
            current_section = section_candidate if section_candidate in assignments else None
            current_lines = []
        elif current_section:
            current_lines.append(line)
    if current_section and current_lines:
        assignments[current_section].append('\n'.join(current_lines).strip())
    return assignments

def embed_and_assign_chunks(groups, skeleton, embedder=None, call_llm_fn=None, multi_section=False):
    section_names = [s['section'] for s in skeleton]
    assignments = {s: [] for s in section_names}
    if call_llm_fn is not None and multi_section:
        print("[DEBUG] Using LLM-assisted multi-section splitting and assignment.")
        for i, group in enumerate(groups):
            split_assignments = llm_split_and_assign_sections(group['content'], section_names, call_llm_fn)
            for section, texts in split_assignments.items():
                for text in texts:
                    if text.strip():
                        print(f"[DEBUG] LLM assigned part of chunk {i} to section: {section}")
                        assignments[section].append({'content': text, 'parent_section': group.get('parent_section')})
        return assignments
    # If call_llm_fn is provided, use LLM classification for assignment
    section_names = [s['section'] for s in skeleton]
    assignments = {s: [] for s in section_names}
    if call_llm_fn is not None:
        print("[DEBUG] Using LLM-assisted section classification for chunk assignment.")
        for i, group in enumerate(groups):
            section = llm_classify_section(group['content'], section_names, call_llm_fn)
            if section not in assignments:
                print(f"[WARNING] LLM returned unknown section '{section}' for chunk {i}, assigning to '1. Introduction'.")
                section = '1. Introduction'
            print(f"[DEBUG] LLM assigned chunk {i} to section: {section}")
            assignments[section].append(group)
        return assignments
    # Fallback: original semantic assignment
    group_embs = embedder.encode([g['content'] for g in groups])
    section_embs = embedder.encode([s['prompt'] for s in skeleton])
    for i, g_emb in enumerate(group_embs):
        sims = util.cos_sim(g_emb, section_embs)[0]
        best_idx = int(sims.argmax())
        assignments[skeleton[best_idx]['section']].append(groups[i])
    empty_sections = [k for k, v in assignments.items() if len(v) == 0]
    all_in_one = any(len(v) == len(groups) for v in assignments.values())
    if empty_sections or all_in_one:
        print("[DEBUG] Hybrid fallback: using round-robin assignment to ensure all sections are populated.")
        assignments = {s['section']: [] for s in skeleton}
        n_sections = len(skeleton)
        for i, group in enumerate(groups):
            section = skeleton[i % n_sections]['section']
            assignments[section].append(group)
    return assignments

# --- MAIN PIPELINE ---
def clean_latex_output(text):
    # Remove LaTeX preamble and document environment lines
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if re.match(r'\\documentclass', line):
            continue
        if re.match(r'\\usepackage', line):
            continue
        if re.match(r'\\begin\{document\}', line):
            continue
        if re.match(r'\\end\{document\}', line):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)

LATEX_ENV_TYPES = [
    'equation', 'align', 'verbatim', 'table', 'figure', 'pgfplots', 'lstlisting', 'tikzpicture', 'tabular', 'algorithm', 'code', 'minted'
]

def log(msg):
    with open("chunking_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log file at start
with open("chunking_log.txt", "w", encoding="utf-8") as f:
    f.write("")

def main():
    log("=== MCP DEBUG: Chunking and grouping only ===")
    source_file = "unstructured_document.tex"
    log(f"Reading source file: {source_file}")
    doc = parse_latex_ast(source_file)
    chunks = extract_chunks_from_ast(doc)
    log(f"[DEBUG] Extracted {len(chunks)} chunks from AST.")
    for i, c in enumerate(chunks[:5]):
        log(f"[DEBUG] Chunk {i}: type={c.get('type', '')}, title={c.get('title', '')}, parent_section={c.get('parent_section', '')}, content preview: {c.get('content','')[:200]}")
    if not chunks:
        log("[WARNING] No chunks extracted from LaTeX AST!")
    # Dump all raw chunks to a Markdown file
    raw_md_lines = ["# Raw Extracted Chunks\n"]
    for i, chunk in enumerate(chunks):
        raw_md_lines.append(f"## Chunk {i}")
        raw_md_lines.append(f"**Type:** {chunk.get('type', 'paragraph')}")
        raw_md_lines.append(f"**Parent Section:** {chunk.get('parent_section', '')}")
        preview = chunk.get('content', '')
        if len(preview) > 1000:
            preview = preview[:1000] + "..."
        raw_md_lines.append(f"**Content Preview:**\n\n{preview}\n")
        raw_md_lines.append("---\n")
    with open("chunks_raw.md", "w", encoding="utf-8") as f:
        f.write("\n".join(raw_md_lines))
    log("[INFO] Raw extracted chunks written to chunks_raw.md")
    groups = group_paragraphs_and_environments(chunks)
    log(f"[DEBUG] Grouped into {len(groups)} groups.")
    for i, g in enumerate(groups[:5]):
        log(f"[DEBUG] Group {i}: parent_section={g.get('parent_section', '')}, content preview: {g.get('content','')[:200]}")
    if not groups:
        log("[WARNING] No groups formed from chunks!")
    # Write all groups to a Markdown file for analysis
    md_lines = ["# Chunked and Grouped Output\n"]
    for i, group in enumerate(groups):
        md_lines.append(f"## Group {i}")
        md_lines.append(f"**Type:** {group.get('type', 'paragraph')}")
        md_lines.append(f"**Parent Section:** {group.get('parent_section', '')}")
        preview = group.get('content', '')
        if len(preview) > 1000:
            preview = preview[:1000] + "..."
        md_lines.append(f"**Content Preview:**\n\n{preview}\n")
        md_lines.append("---\n")
    with open("chunked_output.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    log("[INFO] Chunked and grouped output written to chunked_output.md")

if __name__ == "__main__":
    main()