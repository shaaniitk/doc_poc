# modules/intelligent_mapper.py

"""
The Four-Pass Strategy:
1.  **Semantic Pass (Fast):** Uses sentence-transformers to create an initial similarity matrix.
2.  **Graph-Boost Pass (Structural):** Uses the document's internal reference graph to boost the
    scores of related chunks, adding structural context to the semantic scores.
3.  **LLM-Powered Remediation Pass (Smart):** For remaining unmapped chunks, it leverages a
    powerful LLM via a structured LangChain chain to make a nuanced decision.
4.  **Contextual Cohesion Pass (Deterministic):** Rescues any final orphans by analyzing
    their original neighbors and graph connections.
"""
import logging
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- LangChain Imports for the Remediation Pass ---
from .llm_client import UnifiedLLMClient, LangChainLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# --- End LangChain Imports ---

from config import DOCUMENT_TEMPLATES, SEMANTIC_MAPPING_CONFIG, PROMPTS

log = logging.getLogger(__name__)

class IntelligentMapper:
    def __init__(self, template_name="bitcoin_paper_hierarchical"):
        self.template_name = template_name
        self.skeleton = DOCUMENT_TEMPLATES.get(template_name)
        if not self.skeleton:
            raise ValueError(f"Template '{template_name}' not found.")
            
        self.config = SEMANTIC_MAPPING_CONFIG
        self.llm_client = UnifiedLLMClient()
        self.langchain_llm = LangChainLLM(client=self.llm_client) # For the LLM pass
        self.embedding_model = SentenceTransformer(self.config['model'])
        
        self.flat_skeleton = self._flatten_skeleton_recursive(self.skeleton, [])
        self.section_paths = [s['path'] for s in self.flat_skeleton]
        self.section_descriptions = [s['description'] for s in self.flat_skeleton]
        self.section_embeddings = self.embedding_model.encode(self.section_descriptions, show_progress_bar=False)

    def map_chunks(self, all_chunks_in_order, use_llm_pass=False):
        # --- Pre-computation ---
        reference_graph = self._build_reference_graph(all_chunks_in_order)
        chunk_contents = [c['content'] for c in all_chunks_in_order]
        chunk_embeddings = self.embedding_model.encode(chunk_contents, show_progress_bar=False)
        
        # --- Pass 1: Initial Semantic Similarity ---
        similarity_matrix = cosine_similarity(chunk_embeddings, self.section_embeddings)
        
        # --- Pass 2: Graph-Boost Pass ---
        boosted_similarity_matrix = self._run_graph_boost_pass(similarity_matrix, all_chunks_in_order, reference_graph)

        # --- Final Decision Making ---
        log.info("--- Final Mapping Decision based on Boosted Scores ---")
        mapped_tree = self._create_empty_skeleton(self.skeleton)
        unmapped_chunks = []
        
        best_match_indices = np.argmax(boosted_similarity_matrix, axis=1)
        best_match_scores = np.max(boosted_similarity_matrix, axis=1)
        threshold = self.config['similarity_threshold']

        for i, chunk in enumerate(all_chunks_in_order):
            best_path_index = best_match_indices[i]
            score = best_match_scores[i]
            chunk['metadata']['assignment_score'] = float(score)

            if score >= threshold:
                best_path = self.section_paths[best_path_index]
                parent_node = mapped_tree
                for part in best_path[:-1]:
                    parent_node = parent_node[part]['subsections']
                parent_node[best_path[-1]]['chunks'].append(chunk)
            else:
                unmapped_chunks.append(chunk)
        
        log.info(f"  -> Initial mapping complete. Mapped: {len(all_chunks_in_order) - len(unmapped_chunks)}. Unmapped: {len(unmapped_chunks)}.")

        # --- Pass 3 & 4: Orphan Handling (LLM and Cohesion) ---
        if unmapped_chunks and use_llm_pass:
            log.info(f"--- Mapper Pass 3: LLM-Powered Remediation for {len(unmapped_chunks)} Orphan(s) ---")
            mapped_tree, remaining_orphans = self._run_llm_pass_langchain(mapped_tree, unmapped_chunks)
        else:
            remaining_orphans = unmapped_chunks

        if remaining_orphans:
            log.info(f"--- Mapper Pass 4: Contextual Cohesion for {len(remaining_orphans)} Orphan(s) ---")
            mapped_tree, final_orphans = self._run_cohesion_pass(mapped_tree, remaining_orphans, all_chunks_in_order)
            if final_orphans:
                mapped_tree['Orphaned_Content'] = final_orphans
        
        return mapped_tree

    # --- UPGRADED WITH LANGCHAIN ---
    def _run_llm_pass_langchain(self, mapped_tree, orphans):
        remaining_orphans = []
        section_details = "\n".join([f"- Path: {' -> '.join(s['path'])}\n  Description: {s['description']}" for s in self.flat_skeleton])
        
        # Define the prompt template for the LangChain chain
        prompt_template = PromptTemplate(
            input_variables=["section_details", "chunk_content"],
            template=PROMPTS['llm_map_chunk_to_section']
        )
        chain = LLMChain(llm=self.langchain_llm, prompt=prompt_template)
        
        valid_paths = {" -> ".join(p) for p in self.section_paths}

        for orphan in orphans:
            try:
                # Use the chain to get the response
                response = chain.run(section_details=section_details, chunk_content=orphan['content']).strip()
                
                if response in valid_paths:
                    target_path = response.split(' -> ')
                    parent_node = mapped_tree
                    for part in target_path[:-1]:
                        parent_node = parent_node[part]['subsections']
                    parent_node[target_path[-1]]['chunks'].append(orphan)
                else: # Includes "UNCATEGORIZED" or hallucinated paths
                    remaining_orphans.append(orphan)
            except Exception as e:
                log.error(f"LangChain LLM mapping failed for chunk {orphan.get('chunk_id', 'N/A')}: {e}")
                remaining_orphans.append(orphan)
        
        return mapped_tree, remaining_orphans

    # The rest of the file remains the same as what you provided...
    # --- PASS 3: COHESION ---
    def _run_cohesion_pass(self, mapped_tree, orphans, all_chunks_in_order):
        # This function remains exactly as you provided it.
        log.info("--- Running Orphan Cohesion Pass ---")
        if not orphans: return mapped_tree, []

        reference_graph = self._build_reference_graph(all_chunks_in_order)
        chunk_id_to_path_map = {}
        def build_path_map(node_level, path):
            for title, node_data in node_level.items():
                if not isinstance(node_data, dict): continue
                current_path = path + [title]
                for chunk in node_data.get('chunks', []):
                    if 'chunk_id' in chunk:
                        chunk_id_to_path_map[chunk['chunk_id']] = current_path
                if node_data.get('subsections'):
                    build_path_map(node_data['subsections'], current_path)
        build_path_map(mapped_tree, [])

        rescue_missions, remaining_orphans_list, rescued_ids = [], [], set()
        for orphan in orphans:
            orphan_id = orphan.get('chunk_id')
            if orphan_id is None:
                remaining_orphans_list.append(orphan)
                continue

            prev_path = chunk_id_to_path_map.get(orphan_id - 1)
            next_path = chunk_id_to_path_map.get(orphan_id + 1)
            if prev_path and prev_path == next_path:
                rescue_missions.append({'orphan': orphan, 'target_path': prev_path, 'neighbor_id': orphan_id - 1})
                rescued_ids.add(orphan_id)
                log.info(f"  -> Planning rescue for orphan {orphan_id} via Neighbor Cohesion.")
                continue

            if reference_graph.has_node(orphan_id):
                for _, target_id in reference_graph.out_edges(orphan_id):
                    target_path = chunk_id_to_path_map.get(target_id)
                    if target_path:
                        rescue_missions.append({'orphan': orphan, 'target_path': target_path, 'neighbor_id': None})
                        rescued_ids.add(orphan_id)
                        log.info(f"  -> Planning rescue for orphan {orphan_id} via Graph Cohesion (references {target_id}).")
                        break
        
        remaining_orphans_list.extend([o for o in orphans if o.get('chunk_id') not in rescued_ids])

        for mission in rescue_missions:
            try:
                parent = mapped_tree
                for part in mission['target_path'][:-1]: parent = parent[part]['subsections']
                target_node = parent[mission['target_path'][-1]]
                if mission['neighbor_id'] is not None:
                    idx = next((i for i, c in enumerate(target_node['chunks']) if c.get('chunk_id') == mission['neighbor_id']), -1)
                    target_node['chunks'].insert(idx + 1, mission['orphan']) if idx != -1 else target_node['chunks'].append(mission['orphan'])
                else:
                    target_node['chunks'].append(mission['orphan'])
            except Exception as e:
                log.error(f"  -> Mission failed for chunk {mission['orphan'].get('chunk_id')}: {e}")
                remaining_orphans_list.append(mission['orphan'])
        
        return mapped_tree, remaining_orphans_list

    def _run_graph_boost_pass(self, similarity_matrix, all_chunks_in_order, reference_graph):
        # This function remains exactly as you provided it.
        log.info("--- Mapper Pass 2: Graph-Based Confidence Boosting ---")
        CONFIDENCE_THRESHOLD = self.config.get('confidence_threshold', 0.6)
        BOOST_AMOUNT = self.config.get('boost_amount', 0.3)

        best_match_indices = np.argmax(similarity_matrix, axis=1)
        best_match_scores = np.max(similarity_matrix, axis=1)
        confident_placements = {
            chunk.get('chunk_id'): best_match_indices[i] 
            for i, chunk in enumerate(all_chunks_in_order)
            if chunk.get('chunk_id') is not None and best_match_scores[i] >= CONFIDENCE_THRESHOLD
        }

        for source_id, target_id in reference_graph.edges():
            if target_id in confident_placements and source_id < similarity_matrix.shape[0]:
                target_section_index = confident_placements[target_id]
                original_score = similarity_matrix[source_id, target_section_index]
                similarity_matrix[source_id, target_section_index] = min(1.0, original_score + BOOST_AMOUNT)

        return similarity_matrix

    def _build_reference_graph(self, all_chunks_in_order):
        # This function remains exactly as you provided it.
        log.info("  -> Building document reference graph...")
        G = nx.DiGraph()
        label_to_chunk_id = {}
        for chunk in all_chunks_in_order:
            chunk_id = chunk.get('chunk_id')
            if chunk_id is None: continue
            G.add_node(chunk_id)
            for label in chunk.get('metadata', {}).get('labels', []):
                label_to_chunk_id[label] = chunk_id
        
        for chunk in all_chunks_in_order:
            chunk_id = chunk.get('chunk_id')
            if chunk_id is None: continue
            for ref in chunk.get('metadata', {}).get('refs', []):
                target_chunk_id = label_to_chunk_id.get(ref)
                if target_chunk_id and G.has_node(target_chunk_id):
                    G.add_edge(chunk_id, target_chunk_id)
        
        log.info(f"  -> Reference graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def _create_empty_skeleton(self, node_level):
        # This function remains exactly as you provided it.
        new_level = {}
        for title, data in node_level.items():
            new_node = data.copy()
            new_node['chunks'] = []
            new_node['subsections'] = self._create_empty_skeleton(data.get('subsections', {}))
            new_level[title] = new_node
        return new_level

    def _flatten_skeleton_recursive(self, node_level, current_path):
        # This function remains exactly as you provided it.
        flat_list = []
        for title, data in node_level.items():
            new_path = current_path + [title]
            flat_list.append({'path': new_path, 'description': data['description']})
            if data.get('subsections'):
                flat_list.extend(self._flatten_skeleton_recursive(data['subsections'], new_path))
        return flat_list