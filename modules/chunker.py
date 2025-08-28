
import os
import re
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexMacroNode, LatexEnvironmentNode
from pylatexenc.macrospec import LatexContextDb
from .llm_client import UnifiedLLMClient, LangChainLLM
from .error_handler import ChunkingError
from config import LLM_CHUNK_CONFIG, PROMPTS,LANGCHAIN_CHUNK_CONFIG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List
from docx import Document
from docx.document import Document as DocxDocument
from pylatexenc.latexwalker import latex_group_delimiters
import logging
# Configure logging 
log = logging.getLogger(__name__)

class SemanticSplit(BaseModel):
    """A Pydantic model for the structured output of the semantic chunking process."""
    chunks: List[str] = Field(
        description="A list of text strings, where each string is a semantically complete paragraph or group of paragraphs."
    )

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

# --- NEW: LangChain-powered Markdown Chunker ---
class LangChainMarkdownChunker:
    def chunk_document(self, content: str):
        # LangChain's splitter is excellent at handling various markdown structures
        md_splitter = RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "], # Prioritize headings
            chunk_size=LANGCHAIN_CHUNK_CONFIG['md_chunk_size'], # A reasonable size for semantic meaning
            chunk_overlap=LANGCHAIN_CHUNK_CONFIG['md_chunk_overlap'],
            length_function=len
        )
        docs = md_splitter.create_documents([content])

        # Convert LangChain's Document objects into your project's chunk format
        chunks = []
        for i, doc in enumerate(docs):
            # A simple heuristic to find the parent section
            # This could be made more robust if needed
            lines = doc.page_content.split('\n')
            parent_section = "Preamble"
            for line in lines:
                if line.startswith('#'):
                    parent_section = line.strip()
                    break

            chunks.append({
                'type': 'paragraph',
                'content': doc.page_content,
                'parent_section': parent_section,
                'metadata': {'source_doc_id': i, 'hierarchy_path': [parent_section]}
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
        self.lw = LatexWalker(self.full_content, latex_context=default_latex_db,comment_delimiters=[('%', '\n')])
        
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
        input_pattern = re.compile(r'\\(?:input|include)\s*\{([^}]+)\}')
        
        def replacer(match):
            filename = match.group(1)
            if not filename.endswith('.tex'):
                filename += '.tex'
            
            filepath = os.path.join(base_dir, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return self._resolve_inputs(f.read(), os.path.dirname(filepath))
            else:
                return match.group(0)

        return input_pattern.sub(replacer, content)

    def _parse_node_list_into_chunks(self, nodelist):
        """
        Recursively traverses the AST node list to identify and create chunks.
        """
        buffer = []
        for node in nodelist:
            if node.isNodeType(LatexMacroNode) and node.macroname in ['section', 'subsection', 'subsubsection']:
                self._process_buffer_as_paragraph(buffer)
                buffer = []
                
                level = {'section': 1, 'subsection': 2, 'subsubsection': 3}[node.macroname]
                section_title = self._get_node_text(node.nodeargs[0]).strip()
                
                self.current_section_hierarchy = self.current_section_hierarchy[:level-1]
                self.current_section_hierarchy.append(section_title)

            elif node.isNodeType(LatexEnvironmentNode):
                self._process_buffer_as_paragraph(buffer)
                buffer = []
                self._create_chunk_from_environment(node)

            else:
                buffer.append(node)
        
        self._process_buffer_as_paragraph(buffer)

    def _process_buffer_as_paragraph(self, buffer):
        """
        Takes a buffer of nodes and creates a paragraph chunk if it's not empty.
        """
        if not buffer: return
        content = ''.join([n.latex_verbatim() for n in buffer]).strip()
        if not content: return

        metadata = self._extract_metadata(buffer)
        chunk_data = { 
            'type': 'paragraph', 'content': content,
            'parent_section': ' -> '.join(self.current_section_hierarchy) or 'Preamble',
            'metadata': {'hierarchy_path': self.current_section_hierarchy.copy(), **metadata}
        }
        self.chunks.append(chunk_data)

    def _create_chunk_from_environment(self, node):
        """
        Creates a specialized chunk from an environment node (e.g., figure, table).
        """
        env_name = node.environmentname
        content = node.latex_verbatim()
        metadata = self._extract_metadata(node.nodelist)
        chunk_data = {
            'type': env_name, 'content': content,
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
        labels, citations, refs = [], [], []
        for node in nodelist:
            if node.isNodeType(LatexMacroNode):
                if node.macroname == 'label':
                    labels.append(self._get_node_text(node.nodeargs[0]))
                elif node.macroname in ('cite', 'citep', 'citet'):
                    citations.extend(self._get_node_text(node.nodeargs[0]).split(','))
                elif node.macroname == 'ref':
                    refs.append(self._get_node_text(node.nodeargs[0]))
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
        return node.latex_verbatim().strip('{}')
    
    def _find_caption(self, nodelist):
        for node in nodelist:
            if node.isNodeType(LatexMacroNode) and node.macroname == 'caption':
                return self._get_node_text(node.nodeargs[0])
        return None

    def chunk_document_with_custom_sections(self):
        initial_chunks = []
        split_pattern = r'(% --- .+? ---)'
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
                sub_chunker = ASTChunker(section_content, source_directory=self.source_directory)
                sub_chunker.current_section_hierarchy = [section_name]
                sub_chunker._parse_node_list_into_chunks(sub_chunker.nodelist)
                initial_chunks.extend(sub_chunker.chunks)
            except Exception as e:
                log.warning(f"AST parsing failed for section '{section_name}'. Treating as raw text. Error: {e}")
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
        chunker = ASTChunker(content, source_directory=os.path.dirname(source_path))
        initial_chunks = chunker.chunk_document_with_custom_sections()
    elif extension == '.docx':
        chunker = DocxChunker()
        initial_chunks = chunker.chunk_document(content)
    elif extension == '.md':
        # --- USE NEW LANGCHAIN CHUNKER ---
        chunker = LangChainMarkdownChunker()
        initial_chunks = chunker.chunk_document(content)
    else:
        raise ChunkingError(f"Unsupported file format for chunking: {extension}")
    
    # --- SHARED FINAL PASSES FOR ALL FORMATS ---
    if not LLM_CHUNK_CONFIG['ENABLE_LLM_ENHANCEMENT']:
        processed_chunks = initial_chunks
    else:
        log.info("-> Applying LLM-enhanced semantic splitting for long paragraphs...")
        processed_chunks = []
        llm_client = UnifiedLLMClient()
        langchain_llm = LangChainLLM(client=llm_client)
        for chunk in initial_chunks:
            if (chunk['type'] == 'paragraph' and 
                len(chunk['content']) > LLM_CHUNK_CONFIG['SEMANTIC_SPLIT_THRESHOLD']):
                log.info(f"    -> Splitting a long paragraph from section '{chunk['parent_section']}'...")
                sub_chunks_content = _llm_semantic_split_langchain(chunk['content'], langchain_llm)
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
    grouped = {}
    for chunk in chunks:
        top_level_section = chunk['metadata']['hierarchy_path'][0] if chunk['metadata']['hierarchy_path'] else 'Preamble'
        if top_level_section not in grouped:
            grouped[top_level_section] = []
        grouped[top_level_section].append(chunk)
    return grouped

# --- REWRITTEN with LangChain Output Parsers for Robustness ---
def _llm_semantic_split_langchain(content: str, llm: LangChainLLM) -> List[str]:
    """
    Uses an LLM with a PydanticOutputParser to find semantic break points in a long piece of text.
    This is more robust than relying on string separators.
    """
    try:
        # 1. Set up the Pydantic Output Parser
        parser = PydanticOutputParser(pydantic_object=SemanticSplit)

        # 2. Create a prompt template that includes the format instructions from the parser
        prompt_template = PromptTemplate(
            template=PROMPTS['semantic_split_paragraph'],
            input_variables=["text_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

       # 3. Create the chain using the modern LCEL pipe syntax.
        #    This pipes the output of the prompt to the model, and the model's output to the parser.
        chain = prompt_template | llm | parser

        # 4. Invoke the chain. The input is a dictionary matching the prompt's input_variables.
        parsed_response = chain.invoke({"text_content": content})
        
        # 5. The result is now a Pydantic object directly, no need for a separate parse step.
        return [c for c in parsed_response.chunks if c]


    except Exception as e:
        log.warning(f"LangChain semantic split failed. Returning original chunk. Error: {e}")
        return [content]