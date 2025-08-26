
"""
The Three-Pass Strategy:
1.  **High-Confidence Semantic Pass (Fast):** Uses a sentence-transformer for
    quick, high-confidence assignments.
2.  **LLM-Powered Mapping Pass (Smart):** For ambiguous chunks, it leverages a
    powerful LLM to make a nuanced, context-aware decision.
3.  **Contextual Cohesion Pass (Deterministic):** Rescues any remaining orphans
    (like code blocks) by analyzing their original neighbors.
"""
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .llm_client import UnifiedLLMClient
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
        self.embedding_model = SentenceTransformer(self.config['model'])
        
        # Prepare the flattened skeleton for all passes
        self.flat_skeleton = self._flatten_skeleton_recursive(self.skeleton, [])
        self.section_paths = [s['path'] for s in self.flat_skeleton]
        # This creates the top-level section names needed for the initial tree structure
        self.section_names = [title for title in self.skeleton.keys()]
        self.section_descriptions = [s['description'] for s in self.flat_skeleton]
        self.section_embeddings = self.embedding_model.encode(self.section_descriptions, show_progress_bar=False)

    def map_chunks(self, all_chunks_in_order, use_llm_pass=False):
        """Main entry point to run the full, multi-pass mapping pipeline."""
        
        # --- Pass 1: High-Confidence Semantic Mapping ---
        log.info("--- Mapper Pass 1: High-Confidence Semantic Mapping ---")
        mapped_tree, pass1_orphans = self._run_semantic_pass(all_chunks_in_order)
        log.info(f"  -> Pass 1 complete. Mapped: {len(all_chunks_in_order) - len(pass1_orphans)}. Orphans remaining: {len(pass1_orphans)}.")

        # --- Pass 2: LLM-Powered Mapping ---
        if pass1_orphans and use_llm_pass: # <-- Use the flag here
            log.info("--- Mapper Pass 2: LLM-Powered Mapping for Ambiguous Chunks ---")
            mapped_tree, pass2_orphans = self._run_llm_pass(mapped_tree, pass1_orphans)
            log.info(f"  -> Pass 2 complete. Mapped: {len(pass1_orphans) - len(pass2_orphans)}. Orphans remaining: {len(pass2_orphans)}.")
        elif pass1_orphans:
            log.info("--- Mapper Pass 2: Skipped LLM-Powered Mapping (disabled by flag) ---")
            pass2_orphans = pass1_orphans # Pass the orphans directly to the next stag
            log.info(f"  -> Pass 2 complete. Mapped: {len(pass1_orphans) - len(pass2_orphans)}. Orphans remaining: {len(pass2_orphans)}.")
        else:
            pass2_orphans = []

        # --- Pass 3: Contextual Cohesion ---
        if pass2_orphans:
            log.info("--- Mapper Pass 3: Contextual Cohesion for Final Orphans ---")
            mapped_tree, final_orphans = self._run_cohesion_pass(mapped_tree, pass2_orphans, all_chunks_in_order)
            if final_orphans:
                mapped_tree['Orphaned_Content'] = final_orphans
        
        return mapped_tree

    # --- PASS 1: SEMANTIC ---
    def _run_semantic_pass(self, chunks):
        """
        Performs the initial, high-confidence semantic mapping.
        """
        # Create the initial hierarchical tree structure
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
            best_path_index = best_match_indices[i]
            score = best_match_scores[i]
            chunk['metadata']['assignment_score'] = float(score)

            if score >= threshold:
                best_path = self.section_paths[best_path_index]
                
                # Navigate and place the chunk
                target_parent = mapped_tree
                for part in best_path[:-1]:
                    target_parent = target_parent[part]['subsections']
                target_node = target_parent[best_path[-1]]
                target_node['chunks'].append(chunk)
            else:
                orphans.append(chunk)

        return mapped_tree, orphans

    # --- PASS 2: LLM ---
    def _run_llm_pass(self, mapped_tree, orphans):
        remaining_orphans = []
        # Prepare a detailed string of section details for the LLM prompt
        section_details = "\n".join([f"- Path: {' -> '.join(s['path'])}\n  Description: {s['description']}" for s in self.flat_skeleton])
        
        for orphan in orphans:
            prompt = PROMPTS['llm_map_chunk_to_section'].format(
                section_details=section_details,
                chunk_content=orphan['content']
            )
            try:
                response = self.llm_client.call_llm([{"role": "user", "content": prompt}]).strip()
                
                if response == "UNCATEGORIZED":
                    remaining_orphans.append(orphan)
                    continue

                # Find the path in our list of valid paths
                target_path = next((path for path in self.section_paths if " -> ".join(path) == response), None)

                if target_path:
                    # Graft the orphan into the tree
                    parent_node = mapped_tree
                    for part in target_path[:-1]:
                        parent_node = parent_node[part]['subsections']
                    parent_node[target_path[-1]]['chunks'].append(orphan)
                else:
                    remaining_orphans.append(orphan) # LLM hallucinated a path
            except Exception as e:
                log.error(f"LLM mapping failed for chunk {orphan.get('chunk_id', 'N/A')}: {e}")
                remaining_orphans.append(orphan)
        
        return mapped_tree, remaining_orphans

    # --- PASS 3: COHESION ---
    def _run_cohesion_pass(self, mapped_tree, orphans, all_chunks_in_order):
        """
        A deterministic pass to "rescue" orphans by checking their neighbors.
        If an orphan's preceding and succeeding chunks were both mapped to the
        same location, it's highly probable the orphan belongs there too.
        """
        log.info("--- Running Orphan Cohesion Pass ---")
    
        # Create a quick lookup map of chunk_id -> its final mapped path
        chunk_id_to_path_map = {}
        def build_path_map(node_level, path):
            for title, node_data in node_level.items():
                if not isinstance(node_data, dict): continue
                current_path = path + [title]
                for chunk in node_data.get('chunks', []):
                    chunk_id_to_path_map[chunk['chunk_id']] = current_path
                if node_data.get('subsections'):
                    build_path_map(node_data['subsections'], current_path)
        
        build_path_map(mapped_tree, [])

        rescued_orphans = []
        remaining_orphans = []

        for orphan in orphans:
            orphan_id = orphan['chunk_id']
            prev_chunk_id = orphan_id - 1
            next_chunk_id = orphan_id + 1

            prev_chunk_path = chunk_id_to_path_map.get(prev_chunk_id)
            next_chunk_path = chunk_id_to_path_map.get(next_chunk_id)

            # The cohesion rule: if the chunk before and the chunk after went to the same place...
            if prev_chunk_path and prev_chunk_path == next_chunk_path:
                # ...then this orphan belongs there too!
                log.info(f"  -> Rescuing orphan chunk {orphan_id} based on neighbor cohesion. Target: {' -> '.join(prev_chunk_path)}")
                
                # Navigate to the target node and insert the orphan
                target_node = mapped_tree
                for part in prev_chunk_path:
                    target_node = target_node[part]
                
                # We need to find the right place to insert it to maintain order
                neighbor_index = -1
                for i, chunk in enumerate(target_node['chunks']):
                    if chunk['chunk_id'] == prev_chunk_id:
                        neighbor_index = i
                        break
                
                target_node['chunks'].insert(neighbor_index + 1, orphan)
                rescued_orphans.append(orphan)
            else:
                remaining_orphans.append(orphan)
        
        log.info(f"  -> Cohesion Pass complete. Rescued: {len(rescued_orphans)}. Remaining orphans: {len(remaining_orphans)}.")
        return mapped_tree, remaining_orphans

    # --- Helper Methods ---
    def _create_empty_skeleton(self, node_level):
        """
        Recursively creates a deep copy of the skeleton, preserving all metadata
        (like 'generative' flags) and setting up empty 'chunks' lists.
        """
        new_level = {}
        for title, data in node_level.items():
            # Start by copying ALL keys from the template (prompt, description, generative, etc.)
            new_node = data.copy()
            # Then, specifically set the 'chunks' list to be empty.
            new_node['chunks'] = []
            # Finally, recurse to build the subsections.
            new_node['subsections'] = self._create_empty_skeleton(data.get('subsections', {}))
            new_level[title] = new_node
        return new_level

    def _flatten_skeleton_recursive(self, node_level, current_path):
        """
        A helper function to traverse the nested skeleton dictionary.
        """
        flat_list = []
        for title, data in node_level.items():
            new_path = current_path + [title]
            flat_list.append({
                'path': new_path,
                'description': data['description']
            })
            if data.get('subsections'):
                flat_list.extend(
                    self._flatten_skeleton_recursive(data['subsections'], new_path)
                )
        return flat_list