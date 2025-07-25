"""Document chunking module with LLM-enhanced chunking"""
import re
import os
import requests

def extract_latex_sections(content):
    """Extract sections from LaTeX document using regex patterns"""
    chunks = []
    
    # Extract comment-based sections
    section_pattern = r'% --- (.+?) ---\n(.*?)(?=% ---|\\begin\{thebibliography\}|\\end\{document\}|$)'
    sections = re.findall(section_pattern, content, re.DOTALL)
    
    for section_name, section_content in sections:
        if section_content.strip():
            parts = extract_content_parts(section_content.strip(), section_name)
            chunks.extend(parts)
    
    # Extract bibliography
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
    # Handle tables with labels as complete units
    table_pattern = r'(\\begin\{table\}.*?\\end\{table\}(?:\s*\\label\{[^}]+\})?)'    
    tables = re.findall(table_pattern, content, re.DOTALL)
    
    temp_content = content
    for i, table in enumerate(tables):
        temp_content = temp_content.replace(table, f"__TABLE_PLACEHOLDER_{i}__")
    
    # Extract other environments
    env_pattern = r'(\\begin\{[^}]+\}.*?\\end\{[^}]+\})'    
    segments = re.split(env_pattern, temp_content, flags=re.DOTALL)
    
    parts = []
    current_text = ""
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        if "__TABLE_PLACEHOLDER_" in segment:
            if current_text.strip():
                parts.append({
                    'type': 'paragraph',
                    'content': current_text.strip(),
                    'parent_section': section_name
                })
                current_text = ""
            
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
    
    # Apply semantic chunking to accumulated text
    if current_text.strip():
        semantic_chunks = semantic_chunk_boundaries(current_text.strip(), section_name)
        for chunk_content in semantic_chunks:
            if chunk_content.strip():
                parts.append({
                    'type': 'paragraph',
                    'content': chunk_content.strip(),
                    'parent_section': section_name
                })
    
    return parts

def dependency_aware_chunking(chunks):
    """Group chunks based on content dependencies"""
    try:
        from config import CHUNKING_PROMPTS
        enhanced_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                
                if has_dependency(current_chunk['content'], next_chunk['content']):
                    merged_chunk = {
                        'type': 'merged',
                        'content': current_chunk['content'] + '\n\n' + next_chunk['content'],
                        'parent_section': current_chunk['parent_section']
                    }
                    enhanced_chunks.append(merged_chunk)
                    i += 2
                    continue
            
            enhanced_chunks.append(current_chunk)
            i += 1
        
        return enhanced_chunks
    except:
        return chunks

def has_dependency(content1, content2):
    """Check if content2 depends on content1"""
    try:
        from .llm_client import UnifiedLLMClient
        client = UnifiedLLMClient()
        prompt = f"""Does the second content depend on or reference the first? Respond only 'YES' or 'NO'.

Content 1: {content1[:300]}
Content 2: {content2[:300]}"""
        
        result = client.call_llm(prompt, max_tokens=5, temperature=0.1)
        return "YES" in result.upper()
    except:
        return False

def content_type_classification(chunks):
    """Classify and group chunks by content type"""
    try:
        from .llm_client import UnifiedLLMClient
        client = UnifiedLLMClient()
        
        for chunk in chunks:
            if len(chunk['content']) > 100:
                prompt = f"""Classify this content type. Respond only: 'equation', 'table', 'figure', 'code', or 'text'.

Content: {chunk['content'][:500]}"""
                
                result = client.call_llm(prompt, max_tokens=10, temperature=0.1)
                chunk['content_type'] = result.strip().lower()
            else:
                chunk['content_type'] = 'text'
        
        return chunks
    except:
        return chunks

def llm_enhance_chunking(chunks, use_llm=True):
    """Use LLM to improve chunk boundaries and merge related content"""
    if not use_llm or not os.getenv("MISTRAL_API_KEY"):
        return chunks
    
    if not use_llm or not os.getenv("MISTRAL_API_KEY"):
        return chunks
    
    # Apply enhancement strategies
    enhanced_chunks = content_type_classification(chunks)
    enhanced_chunks = dependency_aware_chunking(enhanced_chunks)
    
    # Traditional merge logic
    final_chunks = []
    for i, chunk in enumerate(enhanced_chunks):
        if chunk['type'] == 'paragraph' and len(chunk['content']) < 200:
            next_chunk = enhanced_chunks[i+1] if i+1 < len(enhanced_chunks) else None
            if (next_chunk and next_chunk['type'] == 'paragraph' and 
                next_chunk['parent_section'] == chunk['parent_section']):
                
                merge_decision = should_merge_chunks(chunk['content'], next_chunk['content'])
                if merge_decision:
                    merged_chunk = {
                        'type': 'paragraph',
                        'content': chunk['content'] + '\n\n' + next_chunk['content'],
                        'parent_section': chunk['parent_section']
                    }
                    final_chunks.append(merged_chunk)
                    enhanced_chunks[i+1] = None
                    continue
        
        if chunk is not None:
            final_chunks.append(chunk)
    
    return [c for c in final_chunks if c is not None]

def should_merge_chunks(content1, content2):
    """Use LLM to decide if two text chunks should be merged"""
    prompt = f"""Should these two text segments be merged into one coherent paragraph? 
Respond with only 'YES' or 'NO'.

Segment 1: {content1[:300]}

Segment 2: {content2[:300]}"""
    
    try:
        from .llm_client import UnifiedLLMClient
        client = UnifiedLLMClient()
        result = client.call_llm(prompt, max_tokens=10, temperature=0.1)
        return "YES" in result.upper()
    except:
        pass
    
    return False

def semantic_chunk_boundaries(content, section_name):
    """Use LLM to identify optimal chunk boundaries"""
    if len(content) < 500 or not os.getenv("MISTRAL_API_KEY"):
        return [content]
    
    prompt = f"""Identify natural break points in this text where it should be split into coherent chunks.
Return only the first few words of each new chunk, separated by '|||'.

Text: {content[:2000]}"""
    
    try:
        api_key = os.getenv("MISTRAL_API_KEY")
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.1
            },
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()["choices"][0]["message"]["content"]
            break_points = [bp.strip() for bp in result.split('|||') if bp.strip()]
            
            # Split content based on break points
            chunks = []
            current_pos = 0
            
            for bp in break_points[1:]:  # Skip first as it's the beginning
                bp_pos = content.find(bp, current_pos)
                if bp_pos > current_pos:
                    chunks.append(content[current_pos:bp_pos].strip())
                    current_pos = bp_pos
            
            # Add remaining content
            if current_pos < len(content):
                chunks.append(content[current_pos:].strip())
            
            return [c for c in chunks if c.strip()]
    except:
        pass
    
    return [content]

def group_chunks_by_section(chunks):
    """Group chunks by their parent section"""
    grouped = {}
    for chunk in chunks:
        section = chunk['parent_section']
        if section not in grouped:
            grouped[section] = []
        grouped[section].append(chunk)
    return grouped