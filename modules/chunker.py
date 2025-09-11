
import os
import re
import uuid
import numpy as np
#from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexMacroNode, LatexEnvironmentNode
from pylatexenc.macrospec import LatexContextDb
from .llm_client import UnifiedLLMClient, LangChainLLM
from .embedding_client import UnifiedEmbeddingClient
from .error_handler import ChunkingError, EmbeddingAPIError
from config import LLM_CHUNK_CONFIG, PROMPTS,LANGCHAIN_CHUNK_CONFIG, CHUNKING_EMBEDDING
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Tuple, Dict, Any
# from docx import Document
from pylatexenc.macrospec import SpecialsSpec
import logging
from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexMacroNode, LatexEnvironmentNode, LatexCommentNode # <-- Add LatexCommentNode
from pylatexenc.macrospec import LatexContextDb
# Isolate transformers import to avoid jinja2 conflicts
def _get_tokenizer(tokenizer_name="bert-base-uncased"):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except ImportError:
        return None
    except Exception:
        return None
#from pylatexenc.parsers import LatexVerbatimParser
# Configure logging 
log = logging.getLogger(__name__)

class SemanticSplit(BaseModel):
    """A Pydantic model for the structured output of the semantic chunking process."""
    chunks: List[str] = Field(
        description="A list of text strings, where each string is a semantically complete paragraph or group of paragraphs."
    )

# class DocxChunker:
#     def chunk_document(self, doc: Document):
#         chunks = []
#         hierarchy = []
#         current_paragraph_buffer = []

#         def flush_buffer():
#             if current_paragraph_buffer:
#                 content = "\n".join(current_paragraph_buffer).strip()
#                 if content:
#                     chunks.append({
#                         'type': 'paragraph', 'content': content,
#                         'parent_section': ' -> '.join(hierarchy),
#                         'metadata': {'hierarchy_path': hierarchy.copy()}
#                     })
#                 current_paragraph_buffer.clear()

#         for para in doc.paragraphs:
#             style_name = para.style.name
#             level = 0
#             if style_name.startswith('Heading'):
#                 try:
#                     level = int(style_name.split(' ')[-1])
#                 except:
#                     level = 0
#             
#             if level > 0:
#                 flush_buffer()
#                 # Update hierarchy
#                 hierarchy = hierarchy[:level-1]
#                 hierarchy.append(para.text.strip())
#             else:
#                 current_paragraph_buffer.append(para.text)
#         
#         flush_buffer() # Flush any remaining paragraph content at the end
#         return chunks


    

    


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
    
    # elif extension == '.docx':
    #     log.info("-> Using DocxChunker for .docx file.")
    #     # We need to load the document object for the docx chunker
    #     from docx import Document
    #     doc = Document(io.BytesIO(content)) # Assume content is bytes, or load from path
    #     chunker = DocxChunker()
    #     initial_chunks = chunker.chunk_document(doc)

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
    
     # --- NEW: Centralized ID and Metadata Assignment ---
    total_chunks = len(processed_chunks)
    for i, chunk in enumerate(processed_chunks):
        chunk['chunk_id'] = i
        
        # --- NEW METADATA INJECTION ---
        chunk['metadata']['normalized_position'] = i / max(1, total_chunks - 1)
        chunk['metadata']['prev_chunk_id'] = i - 1 if i > 0 else None
        chunk['metadata']['next_chunk_id'] = i + 1 if i < total_chunks - 1 else None
    # Centralized ID Assignment
    
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


# --- New adaptive, token-aware chunker (non-breaking addition) ---


def _split_atomic_blocks(text: str, custom_patterns: tuple = None) -> List[Tuple[str, str]]:
    """
    Split text into segments, preserving atomic blocks like code blocks and LaTeX math environments.
    Returns a list of tuples: (segment_type, segment_text) where segment_type in {"atomic", "text"}.
    """
    # Default patterns for atomic blocks: fenced code, display math $$...$$, \[...\], \begin{...}...\end{...}
    default_patterns = [
        (r"```[\s\S]*?```", "fenced_code"),
        (r"\$\$[\s\S]*?\$\$", "display_math"),
        (r"\\\[[\s\S]*?\\\]", "display_math_bracket"),
        (r"\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}", "latex_env"),
    ]
    
    # Combine custom patterns with default patterns if provided
    if custom_patterns is not None:
        patterns = list(custom_patterns) + default_patterns
    else:
        patterns = default_patterns

    # Build a combined regex with capturing groups
    combined = "|".join(f"({p})" for p, _ in patterns)
    regex = re.compile(combined, re.MULTILINE)

    segments: List[Tuple[str, str]] = []
    last_idx = 0
    for m in regex.finditer(text):
        if m.start() > last_idx:
            segments.append(("text", text[last_idx:m.start()]))
        segments.append(("atomic", m.group(0)))
        last_idx = m.end()
    if last_idx < len(text):
        segments.append(("text", text[last_idx:]))
    return segments


def _simple_sentences(paragraph: str) -> List[str]:
    # Lightweight sentence splitter; avoids external deps
    # Split on ., !, ? followed by space or end, but keep delimiters
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _topic_boundaries(sentences: List[str], low_sim_threshold: float = 0.25) -> List[int]:
    """
    Heuristic topic-shift detection based on lexical Jaccard similarity between adjacent sentences.
    Returns list of indices after which a boundary should be placed.
    """
    boundaries: List[int] = []
    for i in range(len(sentences) - 1):
        s1, s2 = _tokens(sentences[i]), _tokens(sentences[i + 1])
        sim = _jaccard(s1, s2)
        if sim < low_sim_threshold:
            boundaries.append(i)
    return boundaries


class AdaptiveChunker:
    """
    Token-aware, topic-shift-aware, multi-granularity chunker.
    - Token-based windows with overlap
    - Preserves atomic blocks (code/math environments)
    - Produces fine/medium/coarse granularities with lineage tracking
    """

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_tokens_fine: int = 256,
        overlap_tokens: int = 32,
        granularity_multipliers: Tuple[int, int] = (2, 4),
    ) -> None:
        self.tokenizer = _get_tokenizer(tokenizer_name)
        self.max_tokens_fine = max_tokens_fine
        self.overlap_tokens = overlap_tokens
        self.granularity_multipliers = granularity_multipliers  # (medium_factor, coarse_factor)

    def _token_len(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Fallback: approximate token count as words * 1.3
            return int(len(text.split()) * 1.3)

    def _window_split(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        if self.tokenizer is None:
            # Fallback to simple word-based splitting when tokenizer is not available
            words = text.split()
            chunks = []
            start = 0
            while start < len(words):
                end = min(start + max_tokens, len(words))
                chunk_text = ' '.join(words[start:end])
                chunks.append(chunk_text)
                if end == len(words):
                    break
                start = end - overlap
            return chunks
        
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: List[List[int]] = []
        start = 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            chunks.append(ids[start:end])
            if end == len(ids):
                break
            start = max(0, end - overlap)
        return [self.tokenizer.decode(c, skip_special_tokens=True) for c in chunks]

    def _chunk_text_segment(self, text: str, max_tokens: int, overlap: int, enable_topic_shifts: bool = True, prefer_sweeping: bool = False) -> List[str]:
        # Break into paragraphs, then sentences within paragraphs
        paragraphs = [p for p in re.split(r"\n\n+", text) if p.strip()]
        out: List[str] = []
        for para in paragraphs:
            sents = _simple_sentences(para)
            if enable_topic_shifts and len(sents) > 1:
                boundaries = set(_topic_boundaries(sents, low_sim_threshold=0.4 if prefer_sweeping else 0.25))
                # Group sentences between boundaries
                current: List[str] = []
                for i, s in enumerate(sents):
                    current.append(s)
                    if i in boundaries:
                        block = " ".join(current).strip()
                        out.extend(self._window_split(block, max_tokens, overlap))
                        current = []
                if current:
                    block = " ".join(current).strip()
                    out.extend(self._window_split(block, max_tokens, overlap))
            else:
                out.extend(self._window_split(para, max_tokens, overlap))
        return [c for c in out if c.strip()]

    def _label_block_type(self, segment_text: str) -> str:
        if segment_text.strip().startswith("```"):
            return "code"
        if segment_text.strip().startswith("$$") or segment_text.strip().startswith("\\[") or segment_text.strip().startswith("\\begin"):
            return "math"
        return "text"

    def _attach_sequence_metadata(self, chunks: List[Dict[str, Any]]) -> None:
        for i, ch in enumerate(chunks):
            meta = ch.setdefault("metadata", {})
            meta["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
            meta["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i < len(chunks) - 1 else None

    def chunk(self, content: str, source_path: str = "", enable_topic_shifts: bool = True, doc_type: str = "text", atomic_blocks: tuple = None, max_tokens: int = None, prefer_sweeping: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns a dict with three granularities: {"fine": [...], "medium": [...], "coarse": [...]}.
        Each chunk contains: chunk_id, content, metadata with token_count, level, block_type, source_path.
        """
        max_fine = max_tokens if max_tokens is not None else self.max_tokens_fine
        max_med = max_fine * self.granularity_multipliers[0]
        max_coarse = max_fine * self.granularity_multipliers[1]

        # Convert atomic_blocks tuple to pattern format if provided
        custom_patterns = None
        if atomic_blocks is not None:
            custom_patterns = []
            for block_type in atomic_blocks:
                if block_type == 'equation':
                    custom_patterns.append((r"\\begin\{equation\}[\s\S]*?\\end\{equation\}", "equation"))
                elif block_type == 'align':
                    custom_patterns.append((r"\\begin\{align\}[\s\S]*?\\end\{align\}", "align"))
                # Add more block types as needed
        
        segments = _split_atomic_blocks(content, custom_patterns)

        outputs = {"fine": [], "medium": [], "coarse": []}
        for seg_type, seg_text in segments:
            if seg_type == "atomic":
                block_type = self._label_block_type(seg_text)
                token_count = self._token_len(seg_text)
                cid = str(uuid.uuid4())
                base = {
                    "chunk_id": cid,
                    "content": seg_text,
                    "metadata": {
                        "token_count": token_count,
                        "level": "atomic",
                        "block_type": block_type,
                        "source_path": source_path,
                    },
                }
                # Atomic blocks are duplicated across granularities (not split)
                outputs["fine"].append(base.copy())
                outputs["medium"].append(base.copy())
                outputs["coarse"].append(base.copy())
            else:
                # Non-atomic text -> split by windows at multiple granularities
                for level, max_tokens in [("fine", max_fine), ("medium", max_med), ("coarse", max_coarse)]:
                    parts = self._chunk_text_segment(seg_text, max_tokens, self.overlap_tokens, enable_topic_shifts, prefer_sweeping)
                    for p in parts:
                        cid = str(uuid.uuid4())
                        outputs[level].append({
                            "chunk_id": cid,
                            "content": p,
                            "metadata": {
                                "token_count": self._token_len(p),
                                "level": level,
                                "block_type": "text",
                                "source_path": source_path,
                            },
                        })

        # Attach sequence metadata within each level
        for level in outputs:
            self._attach_sequence_metadata(outputs[level])
        # Return only fine granularity for compatibility with existing tests
        return outputs["fine"]


class EmbeddingGuidedChunker:
    """
    Advanced chunker that uses embedding-based cohesion analysis for boundary detection.
    Implements adaptive chunk sizing based on semantic coherence.
    """
    
    def __init__(self, config=None):
        self.config = config or CHUNKING_EMBEDDING
        self.embedding_client = UnifiedEmbeddingClient() if self.config.get('enable', False) else None
        self.cohesion_threshold = self.config.get('cohesion_threshold', 0.7)
        self.max_tokens = self.config.get('max_tokens_per_chunk', 1500)
        self.overlap_tokens = self.config.get('overlap_tokens', 200)
        self.adaptive_sizing = self.config.get('adaptive_sizing', True)
        self.smart_overlap = self.config.get('smart_overlap', True)
        self.logger = log
        self.cache = {}  # Cache for embeddings
        
        # Initialize tokenizer for token counting
        self.tokenizer = _get_tokenizer("bert-base-uncased")
    
    def _token_len(self, text: str) -> int:
        """Calculate token length using the tokenizer."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            # Fallback: approximate token count as words * 1.3
            return int(len(text.split()) * 1.3)
    
    def _calculate_cohesion_scores(self, embeddings: List[List[float]]) -> List[float]:
        """
        Calculate cohesion scores between adjacent embeddings using cosine similarity.
        Returns list of cohesion scores where score[i] is cohesion between embedding[i] and embedding[i+1].
        """
        if not embeddings:
            return []
        
        if len(embeddings) == 1:
            return [1.0]
        
        if len(embeddings) < 2:
            return []

        try:
            import numpy as np
            cohesion_scores = [1.0]  # First embedding always gets 1.0
            
            # Calculate cosine similarity between adjacent embeddings
            for i in range(1, len(embeddings)):
                vec1 = np.array(embeddings[i-1])
                vec2 = np.array(embeddings[i])
                
                # Calculate cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm1 * norm2)
                
                cohesion_scores.append(float(similarity))

            return cohesion_scores
        except Exception as e:
            log.warning(f"Failed to calculate cohesion scores: {e}")
            return [1.0] + [0.5] * (len(embeddings) - 1)
    
    def _find_boundary_candidates(self, cohesion_scores: List[float]) -> List[int]:
        """
        Find potential chunk boundaries based on cohesion minima.
        Returns indices where cohesion drops below threshold or at local minima.
        """
        boundaries = []
        
        for i, score in enumerate(cohesion_scores):
            # Add boundary if cohesion drops below threshold
            if score < self.cohesion_threshold:
                boundaries.append(i)  # Boundary at sentence i
            
            # Add boundary at local minima (if significantly lower than neighbors)
            elif i > 0 and i < len(cohesion_scores) - 1:
                prev_score = cohesion_scores[i - 1]
                next_score = cohesion_scores[i + 1]
                if score < prev_score - 0.1 and score < next_score - 0.1:
                    boundaries.append(i)
        
        return sorted(set(boundaries))
    
    def _adaptive_chunk_sizing(self, sentences: List[str], boundaries: List[int]) -> List[List[str]]:
        """
        Create chunks with adaptive sizing based on content density and token limits.
        """
        chunks = []
        start_idx = 0
        
        for boundary in boundaries + [len(sentences)]:
            chunk_sentences = sentences[start_idx:boundary]
            
            if not chunk_sentences:
                continue
            
            # Check if chunk exceeds token limit
            chunk_text = " ".join(chunk_sentences)
            token_count = self._token_len(chunk_text)
            
            if token_count > self.max_tokens and len(chunk_sentences) > 1:
                # Split large chunk further
                mid_point = len(chunk_sentences) // 2
                chunks.append(chunk_sentences[:mid_point])
                chunks.append(chunk_sentences[mid_point:])
            else:
                chunks.append(chunk_sentences)
            
            start_idx = boundary
        
        return chunks
    
    def _apply_smart_overlap(self, chunks: List[List[str]]) -> List[str]:
        """
        Apply intelligent overlap between chunks based on semantic continuity.
        """
        if not self.smart_overlap or len(chunks) < 2:
            return [" ".join(chunk) for chunk in chunks]
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = " ".join(chunk)
            
            if i > 0 and self.overlap_tokens > 0:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_sentences = prev_chunk[-2:] if len(prev_chunk) >= 2 else prev_chunk
                overlap_text = " ".join(overlap_sentences)
                
                # Limit overlap by token count
                overlap_tokens = self._token_len(overlap_text)
                if overlap_tokens <= self.overlap_tokens:
                    chunk_text = overlap_text + " " + chunk_text
            
            overlapped_chunks.append(chunk_text)
        
        return overlapped_chunks
    
    def chunk_with_embeddings(self, content: str, source_path: str = "") -> List[Dict[str, Any]]:
        """
        Main chunking method using embedding-guided boundary detection.
        
        Args:
            content: Text content to chunk
            source_path: Source file path for metadata
        
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if not self.config.get('enable', False):
            # Fallback to simple chunking if embedding chunking is disabled
            return self._fallback_chunking(content, source_path)
        
        try:
            # Split content into sentences
            sentences = self._extract_sentences(content)
            
            if len(sentences) < 2:
                # Single sentence or empty content
                return [{
                    "content": content.strip(),
                    "metadata": {
                        "chunk_id": str(uuid.uuid4()),
                        "method": "embedding_guided",
                        "token_count": self._token_len(content),
                        "cohesion_score": 1.0,
                        "source_path": source_path,
                        "chunking_method": "embedding_guided"
                    }
                }]
            
            # Get embeddings for sentences and calculate cohesion scores
            try:
                embeddings = self.embedding_client.get_embeddings(sentences)
                cohesion_scores = self._calculate_cohesion_scores(embeddings)
            except EmbeddingAPIError as e:
                log.warning(f"Failed to get embeddings: {e}")
                return self._fallback_chunk(content, source_path)
            
            # Find boundary candidates based on cohesion
            boundaries = self._find_boundary_candidates(cohesion_scores)
            
            # Apply adaptive chunk sizing
            sentence_chunks = self._adaptive_chunk_sizing(sentences, boundaries)
            
            # Apply smart overlap
            final_chunks = self._apply_smart_overlap(sentence_chunks)
            
            # Create chunk objects with metadata
            result_chunks = []
            for i, chunk_content in enumerate(final_chunks):
                # Calculate average cohesion for this chunk
                chunk_start = sum(len(sentence_chunks[j]) for j in range(i))
                chunk_end = chunk_start + len(sentence_chunks[i]) - 1
                
                relevant_scores = cohesion_scores[max(0, chunk_start):min(len(cohesion_scores), chunk_end)]
                avg_cohesion = np.mean(relevant_scores) if relevant_scores else 1.0
                
                result_chunks.append({
                    "content": chunk_content.strip(),
                    "metadata": {
                        "chunk_id": str(uuid.uuid4()),
                        "method": "embedding_guided",
                        "token_count": self._token_len(chunk_content),
                        "cohesion_score": float(avg_cohesion),
                        "source_path": source_path,
                        "chunking_method": "embedding_guided",
                        "sentence_count": len(sentence_chunks[i]) if i < len(sentence_chunks) else 0
                    }
                })
            
            return result_chunks
            
        except Exception as e:
            log.warning(f"Embedding-guided chunking failed: {e}")
            return self._fallback_chunk(content, source_path)
    
    def _find_optimal_boundaries(self, sentences: List[str], cohesion_scores: List[float]) -> List[int]:
        """Find optimal chunk boundaries based on cohesion scores."""
        return self._find_boundary_candidates(cohesion_scores)
    
    def _create_chunks_with_overlap(self, sentences: List[str], boundaries: List[int]) -> List[Dict[str, Any]]:
        """Create chunks with overlap from sentence boundaries."""
        sentence_chunks = self._adaptive_chunk_sizing(sentences, boundaries)
        chunk_texts = self._apply_smart_overlap(sentence_chunks)
        
        # Convert to proper chunk structure
        chunks = []
        for chunk_text in chunk_texts:
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "chunk_id": str(uuid.uuid4()),
                    "method": "embedding_guided",
                    "token_count": self._token_len(chunk_text),
                    "cohesion_score": 0.8,  # Default high score for embedding-guided chunks
                    "source_path": "",
                    "chunking_method": "embedding_guided"
                }
            })
        return chunks
    
    def _fallback_chunk(self, content: str, source_path: str = "") -> List[Dict[str, Any]]:
        """Fallback chunking when embedding approach fails."""
        return self._fallback_chunking(content, source_path)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self._token_len(text)
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        return self._extract_sentences(content)
    
    def _extract_sentences(self, content: str) -> List[str]:
        """
        Extract sentences from content using simple sentence splitting.
        """
        # Simple sentence splitting - could be enhanced with more sophisticated NLP
        sentences = re.split(r'[.!?]+\s+', content.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _fallback_chunking(self, content: str, source_path: str = "") -> List[Dict[str, Any]]:
        """
        Fallback chunking method when embedding-guided chunking fails or is disabled.
        """
        # Simple token-based chunking
        words = content.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self._token_len(word)
            
            if current_tokens + word_tokens > self.max_tokens and current_chunk:
                # Create chunk
                chunk_content = " ".join(current_chunk)
                chunks.append({
                    "content": chunk_content,
                    "metadata": {
                        "chunk_id": str(uuid.uuid4()),
                        "method": "fallback",
                        "token_count": self._token_len(chunk_content),
                        "cohesion_score": 0.5,  # Default neutral score
                        "source_path": source_path,
                        "chunking_method": "fallback_token_based"
                    }
                })
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(word)
            current_tokens += word_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunks.append({
                "content": chunk_content,
                "metadata": {
                    "chunk_id": str(uuid.uuid4()),
                    "method": "fallback",
                    "token_count": self._token_len(chunk_content),
                    "cohesion_score": 0.5,
                    "source_path": source_path,
                    "chunking_method": "fallback_token_based"
                }
            })
        
        return chunks