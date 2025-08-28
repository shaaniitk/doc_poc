# In modules/intelligent_mapper.py

import logging
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .llm_client import UnifiedLLMClient, LangChainLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config import DOCUMENT_TEMPLATES, SEMANTIC_MAPPING_CONFIG, PROMPTS

log = logging.getLogger(__name__)


class IntelligentMapper:
    """
    A state-of-the-art, multi-pass chunk mapping engine. It uses a combination of
    semantic similarity, graph-based structural analysis, and LLM-powered remediation
    to assign document chunks to a hierarchical template.
    """
    def __init__(self, template_name="bitcoin_paper_hierarchical", template_object=None):
        if template_object:
            self.template_name = template_name
            self.skeleton = template_object
        else:
            self.template_name = template_name
            self.skeleton = DOCUMENT_TEMPLATES.get(template_name)

        if not self.skeleton:
            raise ValueError(f"Template '{template_name}' not found.")
            
        self.config = SEMANTIC_MAPPING_CONFIG
        self.llm_client = UnifiedLLMClient()
        self.langchain_llm = LangChainLLM(client=self.llm_client)
        self.embedding_model = SentenceTransformer(self.config['model'])
        
        self.flat_skeleton = self._flatten_skeleton_recursive(self.skeleton, [])
        self.section_paths = [s['path'] for s in self.flat_skeleton]
        self.section_descriptions = [s['description'] for s in self.flat_skeleton]
        self.section_embeddings = self.embedding_model.encode(self.section_descriptions, show_progress_bar=False)

    # --- MAIN PUBLIC METHOD ---
    def map_chunks(self, all_chunks_in_order, use_llm_pass=False):
        """
        Main entry point to run the full, multi-pass mapping pipeline.
        """
        # --- Pre-computation ---
        reference_graph = self._build_reference_graph(all_chunks_in_order)
        chunk_contents = [c['content'] for c in all_chunks_in_order]
        chunk_embeddings = self.embedding_model.encode(chunk_contents, show_progress_bar=False)
        similarity_matrix = cosine_similarity(chunk_embeddings, self.section_embeddings)
        
        # --- Pass 1: Graph-Boost Pass ---
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
                if not self._assign_chunk_to_path(mapped_tree, chunk, best_path):
                    unmapped_chunks.append(chunk)
            else:
                unmapped_chunks.append(chunk)
        
        log.info(f"  -> Initial mapping complete. Mapped: {len(all_chunks_in_order) - len(unmapped_chunks)}. Unmapped: {len(unmapped_chunks)}.")

        # --- Pass 2 & 3: Orphan Handling (LLM and Cohesion) ---
        remaining_orphans = unmapped_chunks
        if unmapped_chunks and use_llm_pass:
            log.info(f"--- Mapper Pass 2: LLM-Powered Remediation for {len(unmapped_chunks)} Orphan(s) ---")
            mapped_tree, remaining_orphans = self._run_llm_pass_langchain(mapped_tree, unmapped_chunks)

        if remaining_orphans:
            log.info(f"--- Mapper Pass 3: Contextual Cohesion for {len(remaining_orphans)} Orphan(s) ---")
            mapped_tree, final_orphans = self._run_cohesion_pass(mapped_tree, remaining_orphans, all_chunks_in_order)
            if final_orphans:
                mapped_tree['Orphaned_Content'] = final_orphans
        
        return mapped_tree

    # --- HELPER PASSES AND METHODS ---

    def _assign_chunk_to_path(self, mapped_tree, chunk, path):
        """
        Navigates to the correct node in the tree and assigns the chunk,
        while also updating the chunk's internal metadata to reflect its new home.
        """
        try:
            parent_node = mapped_tree
            for part in path[:-1]:
                parent_node = parent_node[part]['subsections']
            target_node = parent_node[path[-1]]
            
            chunk['metadata']['hierarchy_path'] = path
            chunk['parent_section'] = ' -> '.join(path)
            
            target_node['chunks'].append(chunk)
            return True
        except KeyError:
            log.warning(f"Could not find path '{" -> ".join(path)}' in tree skeleton for chunk assignment.")
            return False

    def _run_semantic_pass(self, chunks):
        """
        Performs a simple, high-confidence semantic mapping. This is used by
        other modules like the LLMHandler for quick, internal re-mapping tasks.
        """
        mapped_tree = self._create_empty_skeleton(self.skeleton)
        orphans = []
        if not chunks:
            return mapped_tree, orphans

        chunk_contents = [c['content'] for c in chunks]
        chunk_embeddings = self.embedding_model.encode(chunk_contents, show_progress_bar=False)
        similarity_matrix = cosine_similarity(chunk_embeddings, self.section_embeddings)

        best_match_indices = np.argmax(similarity_matrix, axis=1)
        best_match_scores = np.max(similarity_matrix, axis=1)
        threshold = self.config['similarity_threshold']
        
        for i, chunk in enumerate(chunks):
            score = best_match_scores[i]
            chunk['metadata']['assignment_score'] = float(score)
            if score >= threshold:
                best_path = self.section_paths[best_match_indices[i]]
                if not self._assign_chunk_to_path(mapped_tree, chunk, best_path):
                    orphans.append(chunk)
            else:
                orphans.append(chunk)
        return mapped_tree, orphans

    def _run_llm_pass_langchain(self, mapped_tree, orphans):
        remaining_orphans = []
        section_details = "\n".join([f"- Path: {' -> '.join(s['path'])}\n  Description: {s['description']}" for s in self.flat_skeleton])
        prompt_template = PromptTemplate(input_variables=["section_details", "chunk_content"], template=PROMPTS['llm_map_chunk_to_section'])
        chain = LLMChain(llm=self.langchain_llm, prompt=prompt_template)
        valid_paths = {" -> ".join(p) for p in self.section_paths}

        for orphan in orphans:
            try:
                response = chain.invoke({"section_details": section_details, "chunk_content": orphan['content']})['text'].strip()
                if response in valid_paths:
                    target_path = response.split(' -> ')
                    if not self._assign_chunk_to_path(mapped_tree, orphan, target_path):
                        remaining_orphans.append(orphan)
                else:
                    remaining_orphans.append(orphan)
            except Exception as e:
                log.error(f"LangChain LLM mapping failed for chunk {orphan.get('chunk_id', 'N/A')}: {e}")
                remaining_orphans.append(orphan)
        return mapped_tree, remaining_orphans

    def _run_cohesion_pass(self, mapped_tree, orphans, all_chunks_in_order):
        # This method's logic is complex but self-contained and correct.
        if not orphans: return mapped_tree, []
        
        reference_graph = self._build_reference_graph(all_chunks_in_order)
        chunk_id_to_path_map = {}
        def build_path_map(node_level, path):
            for title, node_data in node_level.items():
                if not isinstance(node_data, dict): continue
                current_path = path + [title]
                for chunk in node_data.get('chunks', []):
                    if 'chunk_id' in chunk: chunk_id_to_path_map[chunk['chunk_id']] = current_path
                if node_data.get('subsections'): build_path_map(node_data['subsections'], current_path)
        build_path_map(mapped_tree, [])

        rescue_missions, remaining_orphans_list, rescued_ids = [], [], set()
        for orphan in orphans:
            orphan_id = orphan.get('chunk_id')
            if orphan_id is None:
                remaining_orphans_list.append(orphan); continue
            
            prev_path = chunk_id_to_path_map.get(orphan_id - 1)
            next_path = chunk_id_to_path_map.get(orphan_id + 1)
            if prev_path and prev_path == next_path:
                rescue_missions.append({'orphan': orphan, 'target_path': prev_path}); rescued_ids.add(orphan_id)
                log.info(f"  -> Planning rescue for orphan {orphan_id} via Neighbor Cohesion."); continue

            if reference_graph.has_node(orphan_id):
                for _, target_id in reference_graph.out_edges(orphan_id):
                    target_path = chunk_id_to_path_map.get(target_id)
                    if target_path:
                        rescue_missions.append({'orphan': orphan, 'target_path': target_path}); rescued_ids.add(orphan_id)
                        log.info(f"  -> Planning rescue for orphan {orphan_id} via Graph Cohesion (references {target_id})."); break
        
        remaining_orphans_list.extend([o for o in orphans if o.get('chunk_id') not in rescued_ids])

        for mission in rescue_missions:
            orphan_to_rescue, target_path = mission['orphan'], mission['target_path']
            if self._assign_chunk_to_path(mapped_tree, orphan_to_rescue, target_path):
                log.info(f"  -> Rescued orphan chunk {orphan_to_rescue.get('chunk_id')} to {' -> '.join(target_path)}")
            else:
                remaining_orphans_list.append(orphan_to_rescue)
        
        return mapped_tree, remaining_orphans_list

    def _run_graph_boost_pass(self, similarity_matrix, all_chunks_in_order, reference_graph):
        log.info("--- Mapper Pass 1: Graph-Based Confidence Boosting ---")
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
        new_level = {}
        for title, data in node_level.items():
            new_node = data.copy()
            new_node['chunks'] = []
            new_node['subsections'] = self._create_empty_skeleton(data.get('subsections', {}))
            new_level[title] = new_node
        return new_level

    def _flatten_skeleton_recursive(self, node_level, current_path):
        flat_list = []
        for title, data in node_level.items():
            new_path = current_path + [title]
            flat_list.append({'path': new_path, 'description': data['description']})
            if data.get('subsections'):
                flat_list.extend(self._flatten_skeleton_recursive(data['subsections'], new_path))
        return flat_list