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
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore import InMemoryDocstore
import faiss

log = logging.getLogger(__name__)

class HierarchicalProcessingAgent:
    def __init__(self, llm_client: UnifiedLLMClient, output_format="latex"):
        self.llm_client = llm_client
        self.langchain_llm = LangChainLLM(client=llm_client)
        self.full_tree = None
        self.global_context = ""
        self.format_enforcer = FormatEnforcer(output_format)
        self.semantic_graph = None # To store the graph
        self.all_chunks_map = {} # For quick lookup
            # --- Initialize LangChain Memory ---
        # 1. Get the model name from the central config.
        model_name = SEMANTIC_MAPPING_CONFIG['model']
        
        # 2. Dynamically determine the embedding dimension from the model itself.
        #    This makes the code robust to any model you choose in the config.
        temp_model = SentenceTransformer(model_name)
        embedding_size = temp_model.get_sentence_embedding_dimension()
        del temp_model # Clean up the temporary model

        # 3. Use the dynamically determined size to create the FAISS index.
        index = faiss.IndexFlatL2(embedding_size)
        
        # 4. Initialize the rest of the memory system as before.
        embedding_fn = SentenceTransformerEmbeddings(model_name=model_name)
        vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
        self.memory = VectorStoreRetrieverMemory(retriever=retriever)
        # --- END Memory Initialization ---

        

    def process_tree(self, document_tree, generative_context=None):
        self.full_tree = document_tree
         # --- BUILD SEMANTIC GRAPH AT THE START ---
        all_chunks = self._flatten_tree_to_chunks(document_tree)
        self.all_chunks_map = {c['chunk_id']: c['content'] for c in all_chunks if 'chunk_id' in c}
        # You'll need an instance of the embedding model here
        embedding_model = SentenceTransformer(SEMANTIC_MAPPING_CONFIG['model']) 
        self.semantic_graph = build_semantic_graph(all_chunks, embedding_model)
        if not self.global_context:
            self.global_context = document_tree.get("Abstract", {}).get('description', 
                                    "A peer-to-peer electronic cash system.")
        return self._recursive_process_node(document_tree, parent_context="", path=[], generative_context=generative_context)

    def _recursive_process_node(self, current_level_nodes, parent_context, path, generative_context=None):
        processed_level = {}
        for title, node_data in current_level_nodes.items():
            if not isinstance(node_data, dict): continue

            current_path = path + [title]
            
            # --- REINSTATED: Dynamic Subsection Generation ---
            # If a node is marked as dynamic, first create its subsections
            if node_data.get('dynamic') and not generative_context:
                self._dynamically_generate_subsections(node_data, title)

            if generative_context:
                node_content = generative_context
                log.info(f"  Generatively processing node: {' -> '.join(current_path)}")
            else:
                log.info(f"  Refactoring node: {' -> '.join(current_path)}")
                node_content = "\n\n".join([chunk['content'] for chunk in node_data.get('chunks', [])])

            node_chunks = node_data.get('chunks', [])
            semantic_context = self._get_semantic_context_for_node(node_chunks)
            # --- END SEMANTIC CONTEXT ---
            # --- NEW: Load relevant memories before processing ---
          
                # --- END Memory Loading ---


            if node_content:

                relevant_memories = self.memory.load_memory_variables({"prompt": node_content})
                memory_context = relevant_memories.get('history', "No relevant memories yet.")
                context = {
                    "node_path": " -> ".join(current_path), "global_context": self.global_context,
                    "parent_context": parent_context, "node_content": node_content,
                    "semantic_context": semantic_context, "memory_context": memory_context
                }
                refactored_content = self._strategy_refactor_content(context, node_data.get('prompt', ''))
                
           
                if 'self_critique_and_refine' in PROMPTS:
                    log.info(f"    -> Running self-critique pass for node: {' -> '.join(current_path)}")
                    refactored_content = self._llm_self_critique_pass(context, refactored_content)

                #SAVE Memory
                self.memory.save_context(
                    {"input": f"Refactor content for section: {title}"}, 
                    {"output": refactored_content}
                )
                # --- END Memory Saving ---

                node_data['processed_content'] = refactored_content
            else:
                node_data['processed_content'] = ""

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
    def _strategy_refactor_content(self, context, node_prompt):
        system_prompt = node_prompt or "You are a professional technical editor."
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
        for title in subsection_titles:
            parent_node_data['subsections'][title] = {
                'prompt': parent_node_data['prompt'], 'description': f"Content related to {title}",
                'chunks': [], 'subsections': {}
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
        # Find the indices of the top_k most similar chunks (excluding itself)
        # We use argpartition for efficiency, as we don't need to fully sort
        top_indices = np.argpartition(sim_scores, -top_k-1)[-top_k-1:]
        
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