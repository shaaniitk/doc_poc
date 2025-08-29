
import os
import re
#from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexMacroNode, LatexEnvironmentNode
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
from pylatexenc.macrospec import SpecialsSpec
import logging
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexMacroNode, LatexEnvironmentNode, LatexCommentNode # <-- Add LatexCommentNode
from pylatexenc.macrospec import LatexContextDb
#from pylatexenc.parsers import LatexVerbatimParser
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

class ASTChunker:
    """
    A state-of-the-art, robust chunker that leverages the pylatexenc AST.
    It cleanly separates the document preamble from the body and then recursively
    walks the body's structure to create chunks with precise contextual information.
    """
    def __init__(self, latex_content, source_directory=None):
        self.source_directory = source_directory or os.getcwd()
        self.full_content = self._resolve_inputs(latex_content, self.source_directory)
        
        db = LatexContextDb()
        db.add_context_category(
                        'comments',
                        specials=[
                            SpecialsSpec('%'),  # Recognize % as a special character
                        ]
                    )
        
        self.lw = LatexWalker(self.full_content, latex_context=db)
        self.nodelist, _, _ = self.lw.get_latex_nodes()
        
    def chunk(self):
        """
        Public entry point. Separates preamble from body, then chunks the body.
        Returns both content chunks and the preserved preamble string.
        """
        preamble_nodes = []
        body_nodelist = []
        # Find the document environment node in the top-level AST
        doc_env_node = None
        for node in self.nodelist:
            if node.isNodeType(LatexEnvironmentNode) and node.environmentname == 'document':
                doc_env_node = node
                break
            else:
                preamble_nodes.append(node)
        
        preserved_metadata = self._extract_preamble_metadata(preamble_nodes)
        
        # Preserve the preamble as a verbatim string
        #preserved_preamble = ''.join([n.latex_verbatim() for n in preamble_nodes]).strip()

        # If a document environment was found, get its content
        if doc_env_node:
            body_nodelist = doc_env_node.nodelist
        else:
            # If no \begin{document}, treat everything after preamble as body
            # This handles standalone content files
            start_index = len(preamble_nodes)
            body_nodelist = self.nodelist[start_index:]
            log.warning("No \\begin{document} environment found. Chunking all content after the preamble.")

        # Start the recursive chunking on ONLY the body nodes
        content_chunks = self._recursive_chunk_parser(body_nodelist, current_hierarchy=[])
        
        return content_chunks, preserved_metadata
    
    def _extract_preamble_metadata(self, preamble_nodes):
        """
        Finds specific commands like \\title, \\author, \\date in the preamble
        and extracts their string content.
        """
        metadata = {
            'title': 'Refactored Document', # Default values
            'author': 'ShantanuMisra',
            'date': r'\today'
        }
        for node in preamble_nodes:
            if node.isNodeType(LatexMacroNode) and node.macroname in ['title', 'author', 'date']:
                if node.nodeargs:
                    # Get the raw text content of the command's argument
                    metadata[node.macroname] = self._get_node_text(node.nodeargs[0])
        return metadata

    def _recursive_chunk_parser(self, nodelist, current_hierarchy):

        chunks = []
        buffer = []

        for node in nodelist:
            if node.isNodeType(LatexEnvironmentNode):
                chunks.extend(self._process_buffer(buffer, current_hierarchy))
                buffer = []
                chunks.append(self._create_environment_chunk(node, current_hierarchy))
                continue

            is_section_command = (
                node.isNodeType(LatexMacroNode) and
                node.macroname in ['section', 'subsection', 'subsubsection', 'paragraph']
            )
            if is_section_command:
                chunks.extend(self._process_buffer(buffer, current_hierarchy))
                buffer = []
                level = {'section': 1, 'subsection': 2, 'subsubsection': 3, 'paragraph': 4}[node.macroname]
                section_title = self._get_node_text(node.nodeargs[0]).strip() if node.nodeargs else f"Untitled {node.macroname}"
                current_hierarchy = current_hierarchy[:level - 1] + [section_title]
                continue

            if not node.isNodeType(LatexCommentNode):
                buffer.append(node)

        chunks.extend(self._process_buffer(buffer, current_hierarchy))
        return chunks

    def _process_buffer(self, buffer, hierarchy):
        if not buffer: return []
        content = ''.join([n.latex_verbatim() for n in buffer]).strip()
        if not content: return []
        parent_section_str = ' -> '.join(hierarchy) if hierarchy else 'Preamble'
        chunk_data = {
            'type': 'paragraph', 'content': content, 'parent_section': parent_section_str,
            'metadata': { 'hierarchy_path': hierarchy.copy() if hierarchy else ['Preamble'], **self._extract_metadata(buffer) }
        }
        return [chunk_data]

    def _create_environment_chunk(self, node, hierarchy):
        parent_section_str = ' -> '.join(hierarchy) if hierarchy else 'Preamble'
        return {
            'type': node.environmentname, 'content': node.latex_verbatim(), 'parent_section': parent_section_str,
            'metadata': { 'hierarchy_path': hierarchy.copy(), 'caption': self._find_caption(node.nodelist or []), **self._extract_metadata(node.nodelist or []) }
        }

    def _resolve_inputs(self, content, base_dir):
        input_pattern = re.compile(r'\\(?:input|include)\s*\{([^}]+)\}')
        def replacer(match):
            filename = match.group(1).strip()
            if not filename.endswith('.tex'): filename += '.tex'
            filepath = os.path.join(base_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return self._resolve_inputs(f.read(), os.path.dirname(filepath))
            else:
                log.warning(f"File specified in \\input not found: {filepath}")
                return match.group(0)
        return input_pattern.sub(replacer, content)

    def _extract_metadata(self, nodelist):
        labels, citations, refs = [], [], []
        for node in nodelist:
            if node.isNodeType(LatexMacroNode):
                if node.macroname == 'label' and node.nodeargs: labels.append(self._get_node_text(node.nodeargs[0]))
                elif node.macroname in ('cite', 'citep', 'citet') and node.nodeargs: citations.extend(self._get_node_text(node.nodeargs[0]).split(','))
                elif node.macroname == 'ref' and node.nodeargs: refs.append(self._get_node_text(node.nodeargs[0]))
            elif node.isNodeType(LatexEnvironmentNode) and node.nodelist:
                child_metadata = self._extract_metadata(node.nodelist)
                labels.extend(child_metadata['labels']); citations.extend(child_metadata['citations']); refs.extend(child_metadata['refs'])
        return {'labels': [l.strip() for l in labels], 'citations': [c.strip() for c in citations], 'refs': [r.strip() for r in refs]}

    def _get_node_text(self, node):
        return node.latex_verbatim().strip('{}')
    
    def _find_caption(self, nodelist):
        for node in nodelist:
            if node.isNodeType(LatexMacroNode) and node.macroname == 'caption' and node.nodeargs:
                return self._get_node_text(node.nodeargs[0])
        return None

# --- Top-Level Functions ---
def extract_document_sections(content, source_path):
    """
    Main entry point for ALL document chunking.
    It inspects the file extension, dispatches to the correct chunker,
    handles special cases like preamble preservation, runs shared post-processing,
    and returns both the final chunks and any preserved data.
    """
    if not source_path:
        raise ValueError("source_path is required to determine file type.")
    
    _, extension = os.path.splitext(source_path)
    initial_chunks = []
    preserved_data = {} # Initialize an empty dict for preserved data
     
    # --- 1. Format-Specific Initial Chunking ---
    if extension in ['.tex', '.txt']:
        log.info(f"-> Using ASTChunker for {extension} file.")
        chunker = ASTChunker(content, source_directory=os.path.dirname(source_path))
        # The ASTChunker is special: it returns chunks AND the preserved preamble
        initial_chunks, preserved_preamble = chunker.chunk()
        if preserved_preamble:
            preserved_data['latex_preamble'] = preserved_preamble
    
    elif extension == '.docx':
        log.info("-> Using DocxChunker for .docx file.")
        # We need to load the document object for the docx chunker
        from docx import Document
        doc = Document(io.BytesIO(content)) # Assume content is bytes, or load from path
        chunker = DocxChunker()
        initial_chunks = chunker.chunk_document(doc)

    elif extension == '.md':
        log.info("-> Using LangChainMarkdownChunker for .md file.")
        chunker = LangChainMarkdownChunker()
        initial_chunks = chunker.chunk_document(content)

    else:
        raise ChunkingError(f"Unsupported file format for chunking: {extension}")
    
    if not initial_chunks:
        log.warning("Initial chunking process resulted in zero chunks. Check input file.")

    # --- 2. Shared Post-Processing (Applied to ALL formats) ---
    
    # Optional LLM Enhancement
    if LLM_CHUNK_CONFIG['ENABLE_LLM_ENHANCEMENT']:
        log.info("-> Applying LLM-enhanced semantic splitting for long paragraphs...")
        processed_chunks = []
        llm_client = UnifiedLLMClient()
        langchain_llm = LangChainLLM(client=llm_client)
        for chunk in initial_chunks:
            if (chunk['type'] == 'paragraph' and 
                len(chunk.get('content', '')) > LLM_CHUNK_CONFIG['SEMANTIC_SPLIT_THRESHOLD']):
                log.info(f"    -> Splitting a long paragraph from section '{chunk.get('parent_section', 'N/A')}'...")
                sub_chunks_content = _llm_semantic_split_langchain(chunk['content'], langchain_llm)
                for sub_content in sub_chunks_content:
                    new_chunk = chunk.copy()
                    new_chunk['content'] = sub_content
                    processed_chunks.append(new_chunk)
            else:
                processed_chunks.append(chunk)
    else:
        processed_chunks = initial_chunks

    # Centralized ID Assignment
    for i, chunk in enumerate(processed_chunks):
        chunk['chunk_id'] = i
    
    log.info(f"-> Final chunk count after post-processing: {len(processed_chunks)}")
    
    # --- 3. Return both chunks and preserved data ---
    return processed_chunks, preserved_data

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