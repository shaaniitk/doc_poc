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

from .error_handler import ChunkingError

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
        Processes the entire document AST and returns a list of enriched chunks.
        """
        self._traverse_nodes(self.nodelist)
        if not self.chunks:
            raise ChunkingError("AST parser failed to produce any chunks. The document might be empty or invalid.")
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

    def _traverse_nodes(self, nodelist):
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
    Main entry point for state-of-the-art document chunking.

    This function utilizes the ASTChunker to parse and chunk a LaTeX document,
    providing a structured, metadata-rich output.

    Args:
        content (str): The raw LaTeX document content.
        source_path (str, optional): The path to the source file. Required for
                                     resolving multi-file projects (`\input`).

    Returns:
        list: A list of structured chunk dictionaries.
    """
    if not content or len(content.strip()) < 10:
        raise ChunkingError("Invalid or empty content provided for chunking.")
    
    try:
        source_dir = os.path.dirname(source_path) if source_path else None
        chunker = ASTChunker(content, source_directory=source_dir)
        return chunker.chunk_document()
    except Exception as e:
        # Provide a more informative error message
        raise ChunkingError(f"AST-based chunking failed: {e}. Check for severe LaTeX syntax errors.")


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