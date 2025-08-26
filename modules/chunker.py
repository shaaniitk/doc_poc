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
from docx import Document
from docx.document import Document as DocxDocument
import logging

# Configure logging
log = logging.getLogger(__name__)

class DocxChunker:
    def chunk_document(self, doc: DocxDocument):
        chunks = []
        hierarchy = []
        current_paragraph_buffer = []

        def flush_buffer():
            if current_paragraph_buffer:
                content = "\n".join(current_paragraph_buffer).strip()
                if content:
                    chunks.append({
                        'type': 'paragraph', 'content': content,
                        'parent_section': ' -> '.join(hierarchy),
                        'metadata': {'hierarchy_path': hierarchy.copy()}
                    })
                current_paragraph_buffer.clear()

        for para in doc.paragraphs:
            style_name = para.style.name
            level = 0
            if style_name.startswith('Heading'):
                try:
                    level = int(style_name.split(' ')[-1])
                except:
                    level = 0
            
            if level > 0:
                flush_buffer()
                # Update hierarchy
                hierarchy = hierarchy[:level-1]
                hierarchy.append(para.text.strip())
            else:
                current_paragraph_buffer.append(para.text)
        
        flush_buffer() # Flush any remaining paragraph content at the end
        return chunks

# --- Specialized Chunker for .md ---
class MarkdownChunker:
    def chunk_document(self, content: str):
        chunks = []
        # Split by markdown headings (which are kept in the list)
        split_pattern = r'(^#{1,6}\s.*$)'
        parts = re.split(split_pattern, content, flags=re.MULTILINE)
        
        hierarchy = []
        # Process content before the first heading
        if parts[0] and parts[0].strip():
             chunks.append({'type': 'paragraph', 'content': parts[0].strip(), 'parent_section': 'Preamble', 'metadata': {'hierarchy_path': []}})

        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            heading_content = parts[i+1].strip()
            
            level = heading.find(' ')
            title = heading[level:].strip()

            hierarchy = hierarchy[:level-1]
            hierarchy.append(title)

            # Here you could add more regex to find code blocks, lists etc. within heading_content
            # For now, we treat the whole content as a single paragraph for simplicity.
            if heading_content:
                chunks.append({
                    'type': 'paragraph', 'content': heading_content,
                    'parent_section': ' -> '.join(hierarchy),
                    'metadata': {'hierarchy_path': hierarchy.copy()}
                })
        return chunks

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
        
        chunk_data = { 
        'type': 'paragraph',
        'content': content,
        'parent_section': ' -> '.join(self.current_section_hierarchy) or 'Preamble',
        'metadata': {
            'hierarchy_path': self.current_section_hierarchy.copy(),
            **metadata
        }
        }
        self.chunks.append(chunk_data)# <-- INCREMENT COUNTER

    def _create_chunk_from_environment(self, node):
        """
        Creates a specialized chunk from an environment node (e.g., figure, table).
        """
        env_name = node.environmentname
        content = node.latex_verbatim()
        metadata = self._extract_metadata(node.nodelist)

        chunk_data = {
        'type': env_name,
        'content': content,
        'parent_section': ' -> '.join(self.current_section_hierarchy) or 'Preamble',
        'metadata': {
            'hierarchy_path': self.current_section_hierarchy.copy(),
            'caption': self._find_caption(node.nodelist if node.nodelist else []),
            **metadata
        }
        }
        self.chunks.append(chunk_data)


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

    def chunk_document_with_custom_sections(self):
        """
        A specialized method for .tex/.txt files that use the '% ---' delimiter.
        It splits by the delimiter and then runs the core AST parser on each section.
        """
        initial_chunks = []
        split_pattern = r'(% --- .+? ---)'
        # Use self.full_content which has already resolved \input commands
        parts = re.split(split_pattern, self.full_content)
        
        if parts[0] and parts[0].strip():
            initial_chunks.append({
                'type': 'paragraph', 'content': parts[0].strip(),
                'parent_section': 'Preamble', 'metadata': {'hierarchy_path': ['Preamble']}
            })

        for i in range(1, len(parts), 2):
            title_match = re.search(r'% --- (.+?) ---', parts[i])
            if not title_match: continue
            section_name = title_match.group(1).strip()
            section_content = parts[i+1].strip()
            if not section_content: continue
            
            try:
                # Use a new instance of the parser for the sub-content. This is a clean way to isolate parsing.
                sub_chunker = ASTChunker(section_content, source_directory=self.source_directory)
                sub_chunker.current_section_hierarchy = [section_name]
                sub_chunker._parse_node_list_into_chunks(sub_chunker.nodelist)
                initial_chunks.extend(sub_chunker.chunks)
            except Exception as e:
                print(f"Warning: AST parsing failed for section '{section_name}'. Treating as raw text. Error: {e}")
                initial_chunks.append({
                    'type': 'paragraph', 'content': section_content,
                    'parent_section': section_name, 'metadata': {'hierarchy_path': [section_name]}
                })
        return initial_chunks

    

# --- Top-Level Functions ---

def extract_document_sections(content, source_path):
    """
    Main entry point for the new HYBRID, FORMAT-AWARE document chunking.
    It inspects the file extension and dispatches to the correct chunker class.
    """
    if not source_path:
        raise ValueError("source_path is required to determine file type.")
    
    _, extension = os.path.splitext(source_path)
    initial_chunks = []

    if extension == '.tex' or extension == '.txt':
        # Instantiate the ASTChunker and call its specialized method
        chunker = ASTChunker(content, source_directory=os.path.dirname(source_path))
        initial_chunks = chunker.chunk_document_with_custom_sections()

    elif extension == '.docx':
        chunker = DocxChunker()
        initial_chunks = chunker.chunk_document(content)

    elif extension == '.md':
        chunker = MarkdownChunker()
        initial_chunks = chunker.chunk_document(content)

    else:
        raise ChunkingError(f"Unsupported file format for chunking: {extension}")
    
    # --- SHARED FINAL PASSES FOR ALL FORMATS ---
    
    # Optional LLM Enhancement
    if not LLM_CHUNK_CONFIG['ENABLE_LLM_ENHANCEMENT']:
        processed_chunks = initial_chunks
    else:
        log.info("-> Applying LLM-enhanced semantic splitting for long paragraphs...")
        processed_chunks = []
        llm_client = UnifiedLLMClient()
        for chunk in initial_chunks:
            if (chunk['type'] == 'paragraph' and 
                len(chunk['content']) > LLM_CHUNK_CONFIG['SEMANTIC_SPLIT_THRESHOLD']):
                log.info(f"    -> Splitting a long paragraph from section '{chunk['parent_section']}'...")
                sub_chunks_content = _llm_semantic_split(chunk['content'], llm_client)
                for sub_content in sub_chunks_content:
                    new_chunk = chunk.copy()
                    new_chunk['content'] = sub_content
                    processed_chunks.append(new_chunk)
            else:
                processed_chunks.append(chunk)

    # FINAL STEP: Centralized ID Assignment
    for i, chunk in enumerate(processed_chunks):
        chunk['chunk_id'] = i
    
    return processed_chunks


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

def _llm_semantic_split(content, llm_client):
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
