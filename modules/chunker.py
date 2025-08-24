"""Intelligent document chunking module

This module implements sophisticated document segmentation with LLM-enhanced
boundary detection and semantic understanding. It transforms raw documents
into intelligently structured chunks ready for processing.

INTELLIGENCE FEATURES:
- Semantic chunk boundary detection using LLM
- Content type classification (equations, tables, paragraphs)
- Dependency-aware chunking for related content
- LaTeX environment preservation
- Adaptive merging based on content relationships

KEY CAPABILITIES:
- Regex-based section extraction from LaTeX documents
- Granular content parsing (preserves tables, equations, figures)
- LLM-driven optimal chunk boundary identification
- Content dependency analysis for intelligent grouping
- Multi-strategy chunk enhancement and optimization

PROCESSING PIPELINE:
1. Extract sections using regex patterns
2. Parse content into granular parts (text, equations, tables)
3. Apply semantic chunking for optimal boundaries
4. Classify content types using LLM
5. Analyze dependencies between chunks
6. Merge related content intelligently

This creates the foundation for all downstream processing.
"""
import re
import os
import requests
from .error_handler import robust_llm_call, validate_chunk, safe_regex_extract, ChunkingError, LLMError

def extract_latex_sections(content):
    """Extract sections with robust error handling
    
    Extracts sections from LaTeX documents using sophisticated regex patterns.
    This is the entry point for document processing - it identifies and
    separates document sections for individual processing.
    
    Args:
        content: Raw LaTeX document content
        
    Returns:
        list: Structured chunks with metadata
    """
    if not content or len(content.strip()) < 10:
        raise ChunkingError("Invalid or empty content provided")
    
    try:
        chunks = []  # Collection of extracted chunks
        
        # Extract comment-based sections using safe regex
        section_pattern = r'% --- (.+?) ---\n(.*?)(?=% ---|\\begin\{thebibliography\}|\\end\{document\}|$)'
        sections = safe_regex_extract(section_pattern, content, re.DOTALL)
        
        # Process each identified section
        for section_name, section_content in sections:
            if section_content.strip():  # Only process non-empty sections
                # Extract granular content parts from section
                parts = extract_content_parts(section_content.strip(), section_name)
                chunks.extend(parts)
        
        # Special handling for bibliography section
        bib_pattern = r'(\\begin\{thebibliography\}.*?\\end\{thebibliography\})'
        bib_matches = safe_regex_extract(bib_pattern, content, re.DOTALL)
        bib_match = bib_matches[0] if bib_matches else None
        
        if bib_match:
            chunks.append({
                'type': 'bibliography',
                'content': bib_match,
                'parent_section': 'References'
            })
        
        return chunks
    except Exception as e:
        raise ChunkingError(f"Section extraction failed: {e}")

def extract_content_parts(content, section_name):
    """Extract content parts with validation"""
    if not content or not section_name:
        return []
    
    try:
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
    
        # Validate all chunks before returning
        validated_parts = []
        for part in parts:
            if validate_chunk(part):
                validated_parts.append(part)
        
        if not validated_parts:
            raise ChunkingError(f"No valid chunks extracted from {section_name}")
        
        return validated_parts
    except Exception as e:
        raise ChunkingError(f"Content extraction failed for {section_name}: {e}")

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

@robust_llm_call(max_retries=1)
def has_dependency(content1, content2):
    """Check if content2 depends on content1"""
    from .llm_client import UnifiedLLMClient
    client = UnifiedLLMClient()
    prompt = f"""Does the second content depend on or reference the first? Respond only 'YES' or 'NO'.

Content 1: {content1[:300]}
Content 2: {content2[:300]}"""
    
    result = client.call_llm(prompt, max_tokens=5, temperature=0.1)
    return "YES" in result.upper()

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
    """LLM-enhanced chunking optimization using configured LLM provider"""
    # Early exit if LLM disabled
    if not use_llm:
        return chunks
    
    # Check if any LLM provider is available
    from config import LLM_CONFIG
    provider = LLM_CONFIG.get("provider")
    api_key_env = LLM_CONFIG.get("api_key_env")
    
    if not provider or not os.getenv(api_key_env):
        print(f"  Warning: {provider} not available, skipping LLM enhancement")
        return chunks
    
    # Apply LLM-powered enhancement strategies
    print("  Applying LLM-enhanced chunking...")
    
    # Step 1: Classify content types using LLM
    enhanced_chunks = content_type_classification(chunks)
    
    # Step 2: Analyze and group dependent content
    enhanced_chunks = dependency_aware_chunking(enhanced_chunks)
    
    # Step 3: Apply intelligent merging logic
    final_chunks = []
    for i, chunk in enumerate(enhanced_chunks):
        # Check if small paragraph chunks should be merged
        if chunk['type'] == 'paragraph' and len(chunk['content']) < 200:
            next_chunk = enhanced_chunks[i+1] if i+1 < len(enhanced_chunks) else None
            
            # Only merge chunks from same section with same type
            if (next_chunk and next_chunk['type'] == 'paragraph' and 
                next_chunk['parent_section'] == chunk['parent_section']):
                
                # LLM decides if chunks should be merged
                merge_decision = should_merge_chunks(chunk['content'], next_chunk['content'])
                if merge_decision:
                    # Create merged chunk
                    merged_chunk = {
                        'type': 'paragraph',
                        'content': chunk['content'] + '\n\n' + next_chunk['content'],
                        'parent_section': chunk['parent_section']
                    }
                    final_chunks.append(merged_chunk)
                    enhanced_chunks[i+1] = None  # Mark for removal
                    continue
        
        # Keep chunk if not merged
        if chunk is not None:
            final_chunks.append(chunk)
    
    # Filter out None entries from merging
    return [c for c in final_chunks if c is not None]

@robust_llm_call(max_retries=1)
def should_merge_chunks(content1, content2):
    """LLM-powered chunk merge decision engine
    
    Uses LLM intelligence to determine if two text chunks should be merged
    based on semantic coherence and logical flow.
    
    Args:
        content1: First chunk content
        content2: Second chunk content
        
    Returns:
        bool: True if chunks should be merged
    """
    # Construct intelligent merge analysis prompt
    prompt = f"""CHUNK MERGE ANALYSIS
    
Should these two text segments be merged into one coherent paragraph?
Consider semantic relationship, logical flow, and natural boundaries.
Respond with only 'YES' or 'NO'.

SEGMENT 1: {content1[:300]}

SEGMENT 2: {content2[:300]}

DECISION:"""
    
    from .llm_client import UnifiedLLMClient
    client = UnifiedLLMClient()
    result = client.call_llm(prompt, max_tokens=10, temperature=0.1)
    return "YES" in result.upper()

@robust_llm_call(max_retries=1)
def semantic_chunk_boundaries(content, section_name):
    """Use LLM to identify optimal chunk boundaries"""
    if len(content) < 500:
        return [content]
    
    from .llm_client import UnifiedLLMClient
    client = UnifiedLLMClient()
    
    prompt = f"""Identify natural break points in this text where it should be split into coherent chunks.
Return only the first few words of each new chunk, separated by '|||'.

Text: {content[:2000]}"""
    
    result = client.call_llm(prompt, max_tokens=200, temperature=0.1)
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

def group_chunks_by_section(chunks):
    """Section-based chunk organization
    
    Groups processed chunks by their parent sections for organized processing.
    This creates the structure needed for section-by-section document processing.
    
    Args:
        chunks: List of processed chunks with parent_section metadata
        
    Returns:
        dict: Sections as keys, lists of chunks as values
    """
    grouped = {}  # Dictionary to hold grouped chunks
    
    # Process each chunk and group by parent section
    for chunk in chunks:
        section = chunk['parent_section']  # Get parent section name
        
        # Create section group if it doesn't exist
        if section not in grouped:
            grouped[section] = []
        
        # Add chunk to its parent section group
        grouped[section].append(chunk)
    
    return grouped