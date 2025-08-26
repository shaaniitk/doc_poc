"""
State-of-the-Art Document Chunker using an Abstract Syntax Tree (AST).

This module deconstructs LaTeX documents by first parsing them into a grammatical
tree structure (AST), which provides a deep understanding of the document's
hierarchy and components. This approach is vastly superior to regex-based methods
and robustly handles complex, nested structures found in academic papers.

Key Features:
- AST Parsing: Uses `pylatexenc` to convert LaTeX source into a traversable AST.
- Hierarchical Awareness: Understands the nesting of sections, subsections, etc.
- Metadata Extraction: Captures labels, citations, and references within each chunk.
- Semantic Grouping: Preserves the integrity of environments like figures, tables,
  and equations, keeping their content, captions, and labels together.
- Multi-File Support: Automatically resolves `\input` and `\include` commands to
  process multi-file projects as a single document.
"""

import os
import re
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexMacroNode, LatexEnvironmentNode
from pylatexenc.macrospec import LatexContextDb
from .llm_client import UnifiedLLMClient
from .error_handler import ChunkingError
from config import LLM_CHUNK_CONFIG, PROMPTS
# --- Main AST-based Chunking Logic ---

class ASTChunker:
    """
    Parses a LaTeX document into an AST and extracts structured, metadata-rich chunks.
    """
    def __init__(self, latex_content, source_directory=None):
        self.source_directory = source_directory or os.getcwd()
        self.full_content = self._resolve_inputs(latex_content, self.source_directory)
        
        # Use a standard LaTeX specification database for parsing
        default_latex_db = LatexContextDb()
        self.lw = LatexWalker(self.full_content, latex_context=default_latex_db)
        
        self.nodelist, _, _ = self.lw.get_latex_nodes()
        self.chunks = []
        self.current_section_hierarchy = []

    # --- AFTER ---
    def chunk_document(self):
        """
        (DEPRECATED) The main logic is now in the top-level 'extract_latex_sections'
        function to support the hybrid parsing strategy. This method is kept
        for potential direct instantiation if ever needed.
        """
        self._parse_node_list_into_chunks(self.nodelist)
        return self.chunks

    def _resolve_inputs(self, content, base_dir):
        """
        Recursively resolves `\input` and `\include` statements in the LaTeX source.
        """
        # A safer pattern that handles optional .tex extension and different quoting
        input_pattern = re.compile(r'\\(?:input|include)\s*\{([^}]+)\}')
        
        def replacer(match):
            filename = match.group(1)
            if not filename.endswith('.tex'):
                filename += '.tex'
            
            filepath = os.path.join(base_dir, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    # Recursively resolve inputs in the included file
                    return self._resolve_inputs(f.read(), os.path.dirname(filepath))
            else:
                # If file not found, leave the command as is to avoid breaking content
                return match.group(0)

        return input_pattern.sub(replacer, content)

    def _parse_node_list_into_chunks(self, nodelist):
        """
        Recursively traverses the AST node list to identify and create chunks.
        """
        buffer = []
        for node in nodelist:
            if node.isNodeType(LatexMacroNode) and node.macroname in ['section', 'subsection', 'subsubsection']:
                # When a new section starts, process the buffered paragraph content first
                self._process_buffer_as_paragraph(buffer)
                buffer = []
                
                # Update hierarchy
                level = {'section': 1, 'subsection': 2, 'subsubsection': 3}[node.macroname]
                section_title = self._get_node_text(node.nodeargs[0]).strip()
                
                # Trim hierarchy to the correct level
                self.current_section_hierarchy = self.current_section_hierarchy[:level-1]
                self.current_section_hierarchy.append(section_title)

            elif node.isNodeType(LatexEnvironmentNode):
                # An environment (figure, table, equation) is a self-contained chunk.
                # Process any preceding text buffer first.
                self._process_buffer_as_paragraph(buffer)
                buffer = []
                
                # Create a chunk for the environment itself
                self._create_chunk_from_environment(node)

            else:
                # It's a character node or another macro, add it to the paragraph buffer
                buffer.append(node)
        
        # Process any remaining buffer at the end of the list
        self._process_buffer_as_paragraph(buffer)

    def _process_buffer_as_paragraph(self, buffer):
        """
        Takes a buffer of nodes and creates a paragraph chunk if it's not empty.
        """
        if not buffer:
            return
            
        content = ''.join([n.latex_verbatim() for n in buffer]).strip()
        if not content:
            return

        metadata = self._extract_metadata(buffer)
        
        self.chunks.append({
            'type': 'paragraph',
            'content': content,
            'parent_section': ' -> '.join(self.current_section_hierarchy) or 'Preamble',
            'metadata': {
                'hierarchy_path': self.current_section_hierarchy.copy(),
                **metadata
            }
        })

    def _create_chunk_from_environment(self, node):
        """
        Creates a specialized chunk from an environment node (e.g., figure, table).
        """
        env_name = node.environmentname
        content = node.latex_verbatim()
        metadata = self._extract_metadata(node.nodelist)

        self.chunks.append({
            'type': env_name,
            'content': content,
            'parent_section': ' -> '.join(self.current_section_hierarchy) or 'Preamble',
            'metadata': {
                'hierarchy_path': self.current_section_hierarchy.copy(),
                'caption': self._find_caption(node.nodelist),
                **metadata
            }
        })

    def _extract_metadata(self, nodelist):
        """
        Extracts labels, citations, and references from a list of nodes.
        """
        labels = []
        citations = []
        refs = []
        
        for node in nodelist:
            if node.isNodeType(LatexMacroNode):
                if node.macroname == 'label':
                    labels.append(self._get_node_text(node.nodeargs[0]))
                elif node.macroname in ('cite', 'citep', 'citet'):
                    citations.extend(self._get_node_text(node.nodeargs[0]).split(','))
                elif node.macroname == 'ref':
                    refs.append(self._get_node_text(node.nodeargs[0]))
            # Recursively search in child nodes of environments
            elif node.isNodeType(LatexEnvironmentNode):
                child_metadata = self._extract_metadata(node.nodelist)
                labels.extend(child_metadata['labels'])
                citations.extend(child_metadata['citations'])
                refs.extend(child_metadata['refs'])
                
        return {
            'labels': [l.strip() for l in labels],
            'citations': [c.strip() for c in citations],
            'refs': [r.strip() for r in refs]
        }
        
    def _get_node_text(self, node):
        """
        Extracts the plain text content from a node or node list.
        """
        return node.latex_verbatim().strip('{}')
    
    def _find_caption(self, nodelist):
        """
        Finds the caption text within an environment's node list.
        """
        for node in nodelist:
            if node.isNodeType(LatexMacroNode) and node.macroname == 'caption':
                return self._get_node_text(node.nodeargs[0])
        return None

    

# --- Top-Level Functions ---

def extract_latex_sections(content, source_path=None):
    """
    Main entry point for the new HYBRID document chunking.
    This version includes a final LLM-enhancement pass for long paragraphs.
    """
    if not content or len(content.strip()) < 10:
        raise ChunkingError("Invalid or empty content provided for chunking.")

    # --- Step 1: Pre-process to resolve \input commands ---
    source_dir = os.path.dirname(source_path) if source_path else None
    temp_resolver = ASTChunker(content, source_directory=source_dir)
    full_content = temp_resolver.full_content
    
    # --- Step 2: Split the document by custom section comments ---
    section_pattern = r'% --- (.+?) ---\n(.*?)(?=% ---|\\end\{document\}|$)'
    sections = re.findall(section_pattern, full_content, re.DOTALL)
    if not sections:
        sections = [("Document", full_content)]

    initial_chunks = []
    for section_name, section_content in sections:
        if not section_content.strip(): continue
        try:
            chunker = ASTChunker(section_content, source_directory=source_dir)
            chunker.current_section_hierarchy = [section_name.strip()]
            chunker._parse_node_list_into_chunks(chunker.nodelist)
            initial_chunks.extend(chunker.chunks)
        except Exception as e:
            print(f"Warning: AST parsing failed for section '{section_name}'. Treating as raw text. Error: {e}")
            initial_chunks.append({
                'type': 'paragraph', 'content': section_content,
                'parent_section': section_name.strip(),
                'metadata': {'hierarchy_path': [section_name.strip()]}
            })

    # --- Step 3 (FINAL): LLM-Enhanced Semantic Splitting ---
    if not LLM_CHUNK_CONFIG['ENABLE_LLM_ENHANCEMENT']:
        return initial_chunks

    print("  -> Applying LLM-enhanced semantic splitting for long paragraphs...")
    final_chunks = []
    llm_client = UnifiedLLMClient() # Create a client for this task
    
    for chunk in initial_chunks:
        # Only apply to long paragraphs
        if (chunk['type'] == 'paragraph' and 
            len(chunk['content']) > LLM_CHUNK_CONFIG['SEMANTIC_SPLIT_THRESHOLD']):
            
            print(f"    -> Splitting a long paragraph from section '{chunk['parent_section']}'...")
            sub_chunks_content = llm_semantic_split(chunk['content'], llm_client)
            
            # Create new chunk objects for each split, inheriting metadata
            for sub_content in sub_chunks_content:
                new_chunk = chunk.copy() # Start with a copy of the original
                new_chunk['content'] = sub_content
                final_chunks.append(new_chunk)
        else:
            # If chunk is not a long paragraph, keep it as is
            final_chunks.append(chunk)
    
    return final_chunks


def group_chunks_by_section(chunks):
    """
    Groups processed chunks by their top-level parent section for organized processing.

    Args:
        chunks: List of processed chunks with parent_section metadata.

    Returns:
        dict: Sections as keys, lists of chunks as values.
    """
    grouped = {}
    for chunk in chunks:
        # Use only the top-level section for grouping to keep it simple
        top_level_section = chunk['metadata']['hierarchy_path'][0] if chunk['metadata']['hierarchy_path'] else 'Preamble'
        
        if top_level_section not in grouped:
            grouped[top_level_section] = []
        
        grouped[top_level_section].append(chunk)
    
    return grouped

def llm_semantic_split(content, llm_client):
        """
        Uses an LLM to find semantic break points in a long piece of text.
        """
        try:
            prompt = PROMPTS['semantic_split_paragraph'].format(text_content=content)
            response = llm_client.call_llm([{"role": "user", "content": prompt}])
            
            # Split the response into markers and clean them up
            break_markers = [m.strip() for m in response.split('|||---|||') if m.strip()]
            
            if not break_markers:
                return [content] # LLM found no good splits

            # Use the markers to split the original content string
            split_points = []
            for marker in break_markers:
                # Find the position of the marker in the original text
                pos = content.find(marker)
                if pos != -1:
                    split_points.append(pos)
            
            if not split_points:
                return [content]

            # Sort points and split the content
            split_points.sort()
            chunks = []
            last_pos = 0
            for pos in split_points:
                chunks.append(content[last_pos:pos].strip())
                last_pos = pos
            chunks.append(content[last_pos:].strip())
            
            return [c for c in chunks if c] # Return non-empty chunks

        except Exception as e:
            print(f"Warning: LLM semantic split failed. Returning original chunk. Error: {e}")
            return [content]
