# modules/llm_handler.py

from .llm_client import UnifiedLLMClient, LangChainLLM
from .error_handler import robust_llm_call
from .format_enforcer import FormatEnforcer
from config import PROMPTS,SEMANTIC_MAPPING_CONFIG
import re
import logging
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# The IntelligentMapper is needed for the dynamic subsection logic
from .intelligent_mapper import IntelligentMapper 
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from .embedding_client import UnifiedEmbeddingClient
import numpy as np
from langchain.memory import VectorStoreRetrieverMemory
# Try to import FAISS and related components; fall back gracefully if unavailable (e.g., Windows without faiss)
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_community.docstore.in_memory import InMemoryDocstore  # type: ignore
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception as _faiss_err:  # Broad by design: support environments without faiss wheels
    FAISS_AVAILABLE = False
    FAISS = None  # type: ignore
    InMemoryDocstore = None  # type: ignore
    faiss = None  # type: ignore
    logging.getLogger(__name__).warning(
        "FAISS is not available; falling back to a simple in-memory memory store. Error: %s",
        _faiss_err,
    )

log = logging.getLogger(__name__)

# Simple fallback memory to avoid hard dependency on FAISS
class SimpleMemory:
    def __init__(self, max_items: int = 10):
        self.history = []
        self.max_items = max_items

    def load_memory_variables(self, inputs):
        if not self.history:
            return {"history": "No relevant memories yet."}
        # Return a small concatenation of recent outputs as the "history"
        snippets = []
        for item in self.history[-3:]:
            out = item.get("output", "")
            if isinstance(out, str) and out:
                snippets.append(out[:250] + ("..." if len(out) > 250 else ""))
        return {"history": "\n".join(snippets) if snippets else "No relevant memories yet."}

    def save_context(self, inputs, outputs):
        self.history.append(outputs)
        # Trim history to cap size
        if len(self.history) > self.max_items:
            self.history = self.history[-self.max_items:]

class HierarchicalProcessingAgent:
    def __init__(self, llm_client: UnifiedLLMClient, output_format="latex", kg_processor=None):
        self.llm_client = llm_client
        self.langchain_llm = LangChainLLM(client=llm_client)
        self.full_tree = None
        self.global_context = ""
        self.format_enforcer = FormatEnforcer(output_format)
        self.semantic_graph = None # To store the graph
        self.all_chunks_map = {} # For quick lookup
        self.kg_processor = kg_processor
            # --- Initialize LangChain Memory ---
        # 1. Create unified embedding client from config
        embedding_client = UnifiedEmbeddingClient(SEMANTIC_MAPPING_CONFIG)
        
        if FAISS_AVAILABLE:
            # 2. Get embedding dimension from the client
            embedding_size = embedding_client.get_embedding_dimension()

            # 3. Use the dynamically determined size to create the FAISS index.
            index = faiss.IndexFlatL2(embedding_size)  # type: ignore
            
            # 4. Initialize the rest of the memory system with our wrapper.
            from .embedding_client import LangChainEmbeddingWrapper
            embedding_fn = LangChainEmbeddingWrapper(embedding_client)
            vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})  # type: ignore
            retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
            self.memory = VectorStoreRetrieverMemory(retriever=retriever)
        else:
            # Graceful fallback when FAISS isn't available
            self.memory = SimpleMemory()
        # --- END Memory Initialization ---

        

    def process_tree(self, document_tree, generative_context=None):
        self.full_tree = document_tree
         # --- BUILD SEMANTIC GRAPH AT THE START ---
        all_chunks = self._flatten_tree_to_chunks(document_tree)
        self.all_chunks_map = {c['chunk_id']: c['content'] for c in all_chunks if 'chunk_id' in c}
        # You'll need an instance of the embedding model here
        embedding_model = UnifiedEmbeddingClient(SEMANTIC_MAPPING_CONFIG) 
        self.semantic_graph = build_semantic_graph(all_chunks, embedding_model)
        if not self.global_context:
            self.global_context = document_tree.get("Abstract", {}).get('description', 
                                    "A peer-to-peer electronic cash system.")
        return self._recursive_process_node(document_tree, parent_context="", path=[], generative_context=generative_context)

    def _recursive_process_node(self, current_level_nodes, parent_context, path, generative_context=None):
        """
        The core recursive method. It traverses the document tree, dynamically creating
        subsections, gathering rich context from multiple sources, and applying a multi-pass
        LLM strategy (refactor + critique) to each node.
        """
        processed_level = {}
        for title, node_data in current_level_nodes.items():
            if not isinstance(node_data, dict): continue

            current_path = path + [title]

            # --- Feature: Dynamic Subsection Generation ---
            if node_data.get('dynamic') and not generative_context:
                self._dynamically_generate_subsections(node_data, title)

            # Determine the source content for this node
            if generative_context:
                # Truncate generative_context to fit within token limits for GPT-2
                max_tokens = 400  # Conservative limit for generative context (leave room for prompts)
                # Simple token estimation (rough approximation: 1 token ≈ 4 characters)
                estimated_tokens = len(generative_context) // 4
                
                if estimated_tokens > max_tokens:
                    # Truncate to fit within token limit
                    max_chars = max_tokens * 4
                    node_content = generative_context[:max_chars] + "..."
                    log.warning(f"Truncated generative context from {estimated_tokens} to ~{max_tokens} tokens")
                else:
                    node_content = generative_context
                    
                log.info(f"  Generatively processing node: {' -> '.join(current_path)}")
            else:
                log.info(f"  Refactoring node: {' -> '.join(current_path)}")
                # Combine chunks while respecting token limits
                chunks = node_data.get('chunks', [])
                if chunks:
                    # Start with first chunk and add others if they fit within token limit
                    node_content = chunks[0]['content']
                    max_tokens = 600  # More conservative limit for GPT-2 (1024 - 400 for prompts/context)
                    
                    # Simple token estimation (rough approximation: 1 token ≈ 4 characters)
                    current_tokens = len(node_content) // 4
                    
                    for chunk in chunks[1:]:
                        chunk_tokens = len(chunk['content']) // 4
                        if current_tokens + chunk_tokens + 10 <= max_tokens:  # +10 for separator
                            node_content += "\n\n" + chunk['content']
                            current_tokens += chunk_tokens + 10
                        else:
                            log.warning(f"Skipping chunk to stay within token limit. Current: {current_tokens}, would add: {chunk_tokens}")
                            break
                else:
                    node_content = ""

            if node_content:
                # --- Feature: Rich Context Gathering ---
                
                # 1. Semantic Context from Knowledge Graph
                semantic_context = "N/A"
                node_chunks = node_data.get('chunks', [])
                if node_chunks and self.kg_processor:
                    first_chunk_id = node_chunks[0].get('chunk_id')
                    if first_chunk_id:
                        neighbor_chunks = self.kg_processor.find_semantic_neighbors(first_chunk_id)
                        context_parts = [c['content'][:250] + "..." for c in neighbor_chunks]
                        semantic_context = "\n---\n".join(context_parts) if context_parts else "N/A"

                # 2. Long-Range Memory Context
                relevant_memories = self.memory.load_memory_variables({"prompt": node_content})
                memory_context = relevant_memories.get('history', "No relevant memories yet.")
                
                # Build the full context dictionary for the prompt
                context = {
                    "node_path": " -> ".join(current_path),
                    "global_context": self.global_context,
                    "parent_context": parent_context,
                    "semantic_context": semantic_context,
                    "memory_context": memory_context,
                    "node_content": node_content
                }

                # --- Feature: Multi-Pass LLM Processing ---
                
                # Pass 1: Initial Refactoring
                refactored_content = self._strategy_refactor_content(context, node_data)
                
                # Pass 2: Self-Critique and Refinement
                if 'self_critique_and_refine' in PROMPTS:
                    log.info(f"    -> Running self-critique pass for node: {' -> '.join(current_path)}")
                    refactored_content = self._llm_self_critique_pass(context, refactored_content)

                # --- Storing Result and Memory ---
                node_data['processed_content'] = refactored_content
                self.memory.save_context(
                    {"input": f"Refactor content for section: {title}"}, 
                    {"output": refactored_content}
                )
            else:
                node_data['processed_content'] = ""

            # Recurse into subsections
            if node_data.get('subsections'):
                processed_subsections = self._recursive_process_node(
                    node_data['subsections'], parent_context=node_data.get('processed_content', ''),
                    path=current_path, generative_context=generative_context
                )
                node_data['subsections'] = processed_subsections
            
            processed_level[title] = node_data
        
        return processed_level
        
    def _flatten_tree_to_chunks(self, node_level):
        chunks = []
        for node_data in node_level.values():
            if isinstance(node_data, dict):
                chunks.extend(node_data.get('chunks', []))
                if node_data.get('subsections'):
                    chunks.extend(self._flatten_tree_to_chunks(node_data['subsections']))
        return chunks

    def _get_semantic_context_for_node(self, node_chunks):
        if not self.semantic_graph or not node_chunks:
            return "N/A"
        
        neighbor_ids = set()
        for chunk in node_chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id is not None and self.semantic_graph.has_node(chunk_id):
                # Find all neighbors (chunks this chunk is related to)
                neighbors = list(self.semantic_graph.successors(chunk_id))
                neighbor_ids.update(neighbors)
        
        # Build the context string from the content of the neighbor chunks
        context_parts = []
        for neighbor_id in sorted(list(neighbor_ids)): # Sort for deterministic order
            content = self.all_chunks_map.get(neighbor_id)
            if content:
                context_parts.append(content[:250] + "...") # Add snippets
        
        return repr("\n---\n").strip("'").join(context_parts) if context_parts else "N/A"

    @robust_llm_call(max_retries=2)
    def _strategy_refactor_content(self, context, node_data):
        persona_prompts = node_data.get('persona_prompts', {})
        system_prompt = persona_prompts.get('default', node_data.get('prompt', 'You are a professional technical editor.')) 
        full_prompt_text = f"{system_prompt}\n\n{PROMPTS['hierarchical_refactor']}"
        prompt_template = PromptTemplate(input_variables=list(context.keys()), template=full_prompt_text)
        chain = LLMChain(llm=self.langchain_llm, prompt=prompt_template)
        result = chain.invoke(context)
        raw_output = result['text']
        clean_output, issues = self.format_enforcer.enforce_format(raw_output)
        if issues: log.warning(f"FormatEnforcer found issues (pass 1): {issues}")
        return clean_output

    # --- REINSTATED: Self-Critique Function ---
    @robust_llm_call(max_retries=2)
    def _llm_self_critique_pass(self, original_context, refactored_text):
        prompt_text = PROMPTS['self_critique_and_refine']
        prompt_template = PromptTemplate(
            input_variables=["node_path", "refactored_text"],
            template=prompt_text
        )
        chain = LLMChain(llm=self.langchain_llm, prompt=prompt_template)
        
        critique_context = {
            'node_path': original_context['node_path'],
            'refactored_text': refactored_text
        }
        
        result = chain.invoke(critique_context)
        raw_output = result['text']
        
        match = re.search(r"Final Polished Version:\s*(.*)", raw_output, re.DOTALL | re.IGNORECASE)
        
        if match:
            raw_final_version = match.group(1).strip()
            clean_final_version, issues = self.format_enforcer.enforce_format(raw_final_version)
            if issues: log.warning(f"FormatEnforcer found issues (pass 2): {issues}")
            return clean_final_version
        else:
            log.warning("Self-critique pass failed to find 'Final Polished Version' marker. Returning original refactored text.")
            return refactored_text
    
    # --- REINSTATED: Dynamic Subsection Function ---
    def _dynamically_generate_subsections(self, parent_node_data, parent_title):
        log.info(f"    -> Running dynamic subsection discovery for '{parent_title}'...")
        all_content = "\n\n".join([chunk['content'] for chunk in parent_node_data.get('chunks', [])])
        if not all_content: return

        try:
            prompt = PROMPTS['dynamic_subsection_identifier'].format(parent_section_title=parent_title, text_content=all_content)
            response = self.llm_client.call_llm([{"role": "user", "content": prompt}])
            subsection_titles = json.loads(response)
            if not isinstance(subsection_titles, list): raise ValueError("LLM did not return a valid JSON list.")
        except Exception as e:
            log.warning(f"    -> WARNING: Failed to dynamically generate subsections for '{parent_title}'. Error: {e}")
            return

        log.info(f"    -> Discovered {len(subsection_titles)} subsections to create.")
        parent_node_data['subsections'] = {}
        parent_path = parent_node_data.get('metadata', {}).get('hierarchy_path', [parent_title])
        for title in subsection_titles:
            parent_node_data['subsections'][title] = {
                'prompt': parent_node_data['prompt'],
                'description': f"Content related to {title}",
                'chunks': [],
                'subsections': {},
                'metadata': {'hierarchy_path': parent_path + [title]}
            }

        if parent_node_data.get('chunks'):
            # This is a clever re-use of our existing powerful module!
            # We create a temporary, one-off mapper for this specific task.
            temp_mapper = IntelligentMapper(template_name=None)
            temp_mapper.skeleton = parent_node_data['subsections']
            temp_mapper.flat_skeleton = temp_mapper._flatten_skeleton_recursive(temp_mapper.skeleton, [])
            temp_mapper.section_paths = [s['path'] for s in temp_mapper.flat_skeleton]
            temp_mapper.section_descriptions = [s['description'] for s in temp_mapper.flat_skeleton]
            temp_mapper.section_embeddings = temp_mapper.embedding_model.encode(temp_mapper.section_descriptions)
            
            # Re-map the parent's chunks into its newly created children
            # We don't need the full multi-pass mapping here, just the initial semantic one.
            mapped_tree, orphans = temp_mapper._run_semantic_pass(parent_node_data['chunks'])
            
            # Integrate the results back
            for section_title, section_data in mapped_tree.items():
                if section_title in parent_node_data['subsections']:
                    parent_node_data['subsections'][section_title]['chunks'].extend(section_data['chunks'])
            
            # Any chunks that couldn't be mapped to the new subsections can be left in the parent.
            parent_node_data['chunks'] = orphans

#TODO SHM : Move these helper to a new .py
def build_semantic_graph(all_chunks, embedding_model, top_k=3, threshold=0.75):
    """
    Builds a directed graph of the document's semantic relationships.
    An edge from chunk A to chunk B means B is one of the most semantically
    similar chunks to A in the entire document.
    """
    log.info("  -> Building document semantic knowledge graph...")
    G = nx.DiGraph()
    if len(all_chunks) < 2: return G

    chunk_ids = [c.get('chunk_id') for c in all_chunks]
    chunk_contents = [c.get('content', '') for c in all_chunks]

    embeddings = embedding_model.encode(chunk_contents, show_progress_bar=False)
    similarity_matrix = cosine_similarity(embeddings)

    for i in range(len(all_chunks)):
        # Get similarity scores for chunk i against all other chunks
        sim_scores = similarity_matrix[i]
        # Adjust top_k to not exceed available chunks (excluding self)
        effective_top_k = min(top_k, len(all_chunks) - 1)
        # Find the indices of the top_k most similar chunks (excluding itself)
        # We use argpartition for efficiency, as we don't need to fully sort
        if effective_top_k > 0:
            top_indices = np.argpartition(sim_scores, -effective_top_k-1)[-effective_top_k-1:]
        else:
            top_indices = []
        
        source_id = chunk_ids[i]
        if source_id is None: continue
        G.add_node(source_id)

        for j in top_indices:
            if i == j: continue # Skip self-reference
            
            score = sim_scores[j]
            if score >= threshold:
                target_id = chunk_ids[j]
                if target_id is not None:
                    G.add_edge(source_id, target_id, weight=score)

    log.info(f"  -> Semantic graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G