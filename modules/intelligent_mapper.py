
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
import logging
import numpy as np
import networkx as nx # <-- ADD THIS IMPORT
from sentence_transformers import SentenceTransformer

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
        
        # --- Pre-computation ---
        # Build the reference graph and semantic embeddings once at the beginning.
        reference_graph = self._build_reference_graph(all_chunks_in_order)
        chunk_contents = [c['content'] for c in all_chunks_in_order]
        chunk_embeddings = self.embedding_model.encode(chunk_contents, show_progress_bar=False)
        similarity_matrix = cosine_similarity(chunk_embeddings, self.section_embeddings)
        
        # --- Pass 1: Graph-Boost Pass ---
        # This is the new, powerful pass that uses structure to improve semantics.
        boosted_similarity_matrix = self._run_graph_boost_pass(similarity_matrix, all_chunks_in_order, reference_graph)

        # --- Final Decision Making (replaces the old Pass 1) ---
        log.info("--- Final Mapping Decision ---")
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

        # --- Pass 2 & 3: Orphan Handling (LLM and Cohesion) ---
        # These passes now run only on the truly unmapped chunks.
        if unmapped_chunks and use_llm_pass:
            mapped_tree, remaining_orphans = self._run_llm_pass(mapped_tree, unmapped_chunks)
        else:
            remaining_orphans = unmapped_chunks

        if remaining_orphans:
            mapped_tree, final_orphans = self._run_cohesion_pass(mapped_tree, remaining_orphans, all_chunks_in_order)
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
        Rescues orphans using two deterministic methods:
        1. Neighbor Cohesion: Checks if the preceding and succeeding chunks went to the same place.
        2. Graph Cohesion: Checks if an orphan references a chunk that was confidently mapped.
        """
        log.info("--- Running Orphan Cohesion Pass ---")
        if not orphans:
            return mapped_tree, []

        # --- Build the necessary data structures ---
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

        # --- Analyze and plan rescue missions ---
        rescue_missions = []
        remaining_orphans = []
        rescued_ids = set()

        for orphan in orphans:
            orphan_id = orphan.get('chunk_id')
            if orphan_id is None:
                remaining_orphans.append(orphan)
                continue

            # Method 1: Neighbor Cohesion
            prev_chunk_path = chunk_id_to_path_map.get(orphan_id - 1)
            next_chunk_path = chunk_id_to_path_map.get(orphan_id + 1)
            if prev_chunk_path and prev_chunk_path == next_chunk_path:
                mission = {'orphan': orphan, 'target_path': prev_chunk_path, 'neighbor_id': orphan_id - 1}
                rescue_missions.append(mission)
                rescued_ids.add(orphan_id)
                log.info(f"  -> Planning rescue for orphan {orphan_id} via Neighbor Cohesion.")
                continue

            # Method 2: Graph Cohesion
            if reference_graph.has_node(orphan_id):
                # Find chunks that this orphan references
                for _, target_id in reference_graph.out_edges(orphan_id):
                    target_path = chunk_id_to_path_map.get(target_id)
                    # If the referenced chunk was confidently mapped...
                    if target_path:
                        # ...rescue the orphan to the same location.
                        mission = {'orphan': orphan, 'target_path': target_path, 'neighbor_id': None}
                        rescue_missions.append(mission)
                        rescued_ids.add(orphan_id)
                        log.info(f"  -> Planning rescue for orphan {orphan_id} via Graph Cohesion (references chunk {target_id}).")
                        break # One successful rescue is enough for this orphan
        
        # Filter out the orphans that have been rescued
        remaining_orphans.extend([o for o in orphans if o.get('chunk_id') not in rescued_ids])

        log.info(f"  -> Cohesion Pass Analysis complete. Planned {len(rescue_missions)} rescue mission(s).")

        # --- Execute rescue missions ---
        for mission in rescue_missions:
            try:
                # ... (The mission execution logic is the same as before, no changes here) ...
                target_parent = mapped_tree
                for part in mission['target_path'][:-1]:
                    target_parent = target_parent[part]['subsections']
                target_node = target_parent[mission['target_path'][-1]]
                
                if mission['neighbor_id'] is not None:
                    # Insert after neighbor if possible
                    neighbor_index = next((i for i, c in enumerate(target_node['chunks']) if c.get('chunk_id') == mission['neighbor_id']), -1)
                    if neighbor_index != -1:
                        target_node['chunks'].insert(neighbor_index + 1, mission['orphan'])
                    else:
                        target_node['chunks'].append(mission['orphan'])
                else:
                    # If rescued by graph, just append
                    target_node['chunks'].append(mission['orphan'])
            except Exception as e:
                log.error(f"  -> Mission failed for chunk {mission['orphan'].get('chunk_id')}: {e}")
                remaining_orphans.append(mission['orphan'])
        
        return mapped_tree, remaining_orphans

    def _run_graph_boost_pass(self, similarity_matrix, all_chunks_in_order, reference_graph):
        """
        Adjusts the semantic similarity matrix by "boosting" the scores of
        chunks that have explicit references to other confidently placed chunks.
        """
        log.info("--- Mapper Pass 2: Graph-Based Confidence Boosting ---")
        
        # A "confident" mapping is one with a high initial score.
        CONFIDENCE_THRESHOLD = self.config['similarity_threshold'] + 0.3 # e.g., 0.6
        BOOST_AMOUNT = 0.5 # A significant boost

        # Find the initial, high-confidence placements
        best_match_indices = np.argmax(similarity_matrix, axis=1)
        best_match_scores = np.max(similarity_matrix, axis=1)

        confident_placements = {
            chunk_id: best_match_indices[i] 
            for i, chunk_id in enumerate(c.get('chunk_id') for c in all_chunks_in_order)
            if best_match_scores[i] >= CONFIDENCE_THRESHOLD
        }

        # Now, iterate through the graph edges to apply boosts
        for source_id, target_id in reference_graph.edges():
            # If the chunk we are referencing (target) has a confident placement...
            if target_id in confident_placements:
                target_section_index = confident_placements[target_id]
                
                # ...then significantly boost the score of the referencing chunk (source)
                # with respect to that same section.
                source_chunk_index = source_id # Since chunk_id is the index
                
                # Boost the score in the main matrix
                original_score = similarity_matrix[source_chunk_index, target_section_index]
                boosted_score = min(1.0, original_score + BOOST_AMOUNT) # Cap at 1.0
                similarity_matrix[source_chunk_index, target_section_index] = boosted_score

        return similarity_matrix

    def _build_reference_graph(self, all_chunks_in_order):
        """
        Builds a directed graph of the document's internal references.
        An edge from chunk A to chunk B means A contains a `\ref` to a `\label` in B.
        """
        log.info("  -> Building document reference graph...")
        G = nx.DiGraph()
        label_to_chunk_id = {}

        # First pass: build the graph nodes and find all labels
        for chunk in all_chunks_in_order:
            chunk_id = chunk.get('chunk_id')
            if chunk_id is None: continue
            G.add_node(chunk_id)
            for label in chunk.get('metadata', {}).get('labels', []):
                label_to_chunk_id[label] = chunk_id
        
        # Second pass: add edges based on references
        for chunk in all_chunks_in_order:
            chunk_id = chunk.get('chunk_id')
            if chunk_id is None: continue
            for ref in chunk.get('metadata', {}).get('refs', []):
                target_chunk_id = label_to_chunk_id.get(ref)
                if target_chunk_id and G.has_node(target_chunk_id):
                    # Add an edge from the referencing chunk to the labeled chunk
                    G.add_edge(chunk_id, target_chunk_id)
        
        log.info(f"  -> Reference graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

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