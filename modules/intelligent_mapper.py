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
    def __init__(self, template_name="bitcoin_paper_hierarchical", template_object=None, kg_processor=None):
        if template_object:
            self.template_name = template_name
            self.skeleton = template_object
        else:
            self.template_name = template_name
            self.skeleton = DOCUMENT_TEMPLATES.get(template_name)

        if not self.skeleton:
            raise ValueError(f"Template '{template_name}' not found.")
            
        self.config = SEMANTIC_MAPPING_CONFIG
        self.kg_processor = kg_processor
        self.embedding_model = self.kg_processor.embedding_model
        
        self.llm_client = UnifiedLLMClient()
        self.langchain_llm = LangChainLLM(client=self.llm_client)
        
        self.flat_skeleton = self._flatten_skeleton_recursive(self.skeleton, [])
        self.section_paths = [s['path'] for s in self.flat_skeleton]
        self.section_descriptions = [s['description'] for s in self.flat_skeleton]
        self.section_embeddings = self.embedding_model.encode(self.section_descriptions, show_progress_bar=False)

    # --- MAIN ORCHESTRATOR METHOD ---
    def map_chunks(self, all_chunks_in_order, use_llm_pass=False):
        """
        Main entry point that orchestrates the full, multi-pass mapping pipeline.
        """
        # Stage 1: Initial Calculations
        similarity_matrix = cosine_similarity(self.kg_processor.embeddings, self.section_embeddings)
        
        # Stage 2: Iterative Mapping
        chunk_assignments, unmapped_indices = self._run_high_confidence_pass(all_chunks_in_order, similarity_matrix)
        refined_matrix = self._run_contextual_refinement_pass(similarity_matrix, chunk_assignments, all_chunks_in_order, unmapped_indices)
        
        # Stage 3: Final Assignment
        mapped_tree, unmapped_chunks = self._run_final_assignment(all_chunks_in_order, refined_matrix)
        
        # Stage 4: Orphan Handling
        remaining_orphans = unmapped_chunks
        if unmapped_chunks and use_llm_pass:
            mapped_tree, remaining_orphans = self._run_llm_pass_langchain(mapped_tree, unmapped_chunks)

        if remaining_orphans:
            mapped_tree, final_orphans = self._run_cohesion_pass(mapped_tree, remaining_orphans, all_chunks_in_order)
            if final_orphans:
                mapped_tree['Orphaned_Content'] = final_orphans
        
        return mapped_tree

    # --- SPECIALIZED HELPER METHODS ---

    def _run_high_confidence_pass(self, all_chunks, similarity_matrix):
        """Round 1: Finds the initial, most obvious chunk assignments."""
        log.info("--- Iterative Mapping Round 1: High-Confidence Pass ---")
        threshold = self.config.get('high_confidence_threshold', 0.80)
        
        assignments = {}
        unmapped_indices = list(range(len(all_chunks)))
        best_matches = np.argmax(similarity_matrix, axis=1)
        
        for i in list(unmapped_indices):
            score = similarity_matrix[i, best_matches[i]]
            if score >= threshold:
                chunk_id = all_chunks[i]['chunk_id']
                assignments[chunk_id] = self.section_paths[best_matches[i]]
                unmapped_indices.remove(i)
        
        log.info(f"  -> Round 1 complete. Mapped: {len(assignments)}. Remaining: {len(unmapped_indices)}.")
        return assignments, unmapped_indices

    def _run_contextual_refinement_pass(self, matrix, assignments, all_chunks, unmapped_indices):
        """Round 2: Uses confident assignments to boost scores of their neighbors."""
        log.info("--- Iterative Mapping Round 2: Contextual Refinement Pass ---")
        boost = self.config.get('refinement_boost', 0.20)

        for i in unmapped_indices:
            chunk = all_chunks[i]
            prev_path = assignments.get(chunk['metadata'].get('prev_chunk_id'))
            next_path = assignments.get(chunk['metadata'].get('next_chunk_id'))

            if prev_path and prev_path == next_path:
                try:
                    target_index = self.section_paths.index(prev_path)
                    original_score = matrix[i, target_index]
                    matrix[i, target_index] = min(1.0, original_score + boost)
                    log.info(f"  -> Boosting chunk {chunk['chunk_id']} based on neighbor context.")
                except ValueError:
                    continue
        return matrix

    def _run_final_assignment(self, all_chunks, refined_matrix):
        """Makes the final assignment decisions based on the refined scores."""
        log.info("--- Final Assignment Pass ---")
        mapped_tree = self._create_empty_skeleton(self.skeleton)
        orphans = []
        threshold = self.config['similarity_threshold']
        best_matches = np.argmax(refined_matrix, axis=1)

        for i, chunk in enumerate(all_chunks):
            best_path_index = best_matches[i]
            score = refined_matrix[i, best_path_index]
            chunk['metadata']['assignment_score'] = float(score)

            if score >= threshold:
                if not self._assign_chunk_to_path(mapped_tree, chunk, self.section_paths[best_path_index]):
                    orphans.append(chunk)
            else:
                orphans.append(chunk)
        return mapped_tree, orphans

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
            log.warning(f"Could not find path {' -> '.join(path)} in tree skeleton for chunk assignment.")
            return False

    def _run_semantic_pass(self, chunks):
        """
        Performs a simple, high-confidence semantic mapping. This is used by
        other modules like the LLMHandler for quick, internal re-mapping tasks.
        Now computes and preserves top-k candidate sections with similarity distances.
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
        top_k = self.config.get('top_k_candidates', 3)
        soft_margin = self.config.get('soft_accept_margin', 0.05)
        gap_margin = self.config.get('gap_accept_margin', 0.1)
        
        for i, chunk in enumerate(chunks):
            score = best_match_scores[i]
            # Compute top-k candidate sections for this chunk
            chunk_similarities = similarity_matrix[i]
            sorted_indices = np.argsort(chunk_similarities)[::-1]
            
            candidate_sections = []
            for j in range(min(top_k, len(sorted_indices))):
                section_idx = sorted_indices[j]
                section_path = self.section_paths[section_idx]
                section_score = float(chunk_similarities[section_idx])
                section_distance = max(0.0, min(1.0, 1.0 - section_score))
                
                candidate_sections.append({
                     'path': section_path,
                     'path_str': ' > '.join(section_path),
                     'score': section_score,
                     'distance': section_distance
                 })
            
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata']['candidate_sections'] = candidate_sections
            chunk['metadata']['assignment_score'] = float(score)
            
            if candidate_sections:
                chunk['metadata']['nearest_section_suggestion'] = candidate_sections[0]['path_str']
                chunk['metadata']['distance_to_nearest'] = candidate_sections[0]['distance']
            
            # Gap heuristic
            if len(candidate_sections) >= 2:
                second_best = candidate_sections[1]['score']
            else:
                second_best = 0.0
            top_gap = float(score - second_best)
            chunk['metadata']['top2_gap'] = top_gap
            
            if score >= threshold:
                best_path = self.section_paths[best_match_indices[i]]
                if not self._assign_chunk_to_path(mapped_tree, chunk, best_path):
                    chunk.setdefault('metadata', {})
                    chunk['metadata']['orphan_reason'] = 'path_missing'
                    orphans.append(chunk)
            elif score >= (threshold - soft_margin) or top_gap >= gap_margin:
                best_path = self.section_paths[best_match_indices[i]]
                chunk['metadata']['soft_assigned'] = True
                chunk['metadata']['assignment_confidence'] = 'low'
                if not self._assign_chunk_to_path(mapped_tree, chunk, best_path):
                    chunk.setdefault('metadata', {})
                    chunk['metadata']['orphan_reason'] = 'path_missing_soft'
                    orphans.append(chunk)
            else:
                chunk.setdefault('metadata', {})
                chunk['metadata']['orphan_reason'] = 'low_similarity'
                chunk['metadata']['orphan_threshold'] = float(threshold)
                orphans.append(chunk)
        return mapped_tree, orphans

    def _run_llm_pass_langchain(self, mapped_tree, orphans):
        remaining_orphans = []
        section_details = "\n".join([f"- Path: {' -> '.join(s['path'])}\n  Description: {s['description']}" for s in self.flat_skeleton])
        prompt_template = PromptTemplate(input_variables=["section_details", "chunk_content"], template=PROMPTS['llm_map_chunk_to_section'])
        chain = LLMChain(llm=self.langchain_llm, prompt=prompt_template)
        # Build case-insensitive map for valid paths
        valid_paths_map = {" -> ".join(p).lower(): p for p in self.section_paths}
        soft_margin = self.config.get('soft_accept_margin', 0.05)
        similarity_threshold = self.config.get('similarity_threshold', 0.7)
        fallback_distance = 1.0 - max(similarity_threshold - soft_margin, 0.0)

        for orphan in orphans:
            try:
                response = chain.invoke({"section_details": section_details, "chunk_content": orphan['content']})['text'].strip()
                # Normalize LLM response
                first_line = response.splitlines()[0].strip().strip('"').strip("'")
                normalized = first_line.replace(' > ', ' -> ').replace('—', '->').replace(' - ', ' -> ').strip()
                target_path = valid_paths_map.get(normalized.lower())

                if target_path:
                    if not self._assign_chunk_to_path(mapped_tree, orphan, target_path):
                        remaining_orphans.append(orphan)
                else:
                    # Fallback: nearest candidate if sufficiently close
                    cand = orphan.get('metadata', {}).get('candidate_sections', [])
                    if cand:
                        nearest = cand[0]
                        if float(nearest.get('distance', 1.0)) <= fallback_distance:
                            orphan.setdefault('metadata', {})
                            orphan['metadata']['soft_assigned'] = True
                            orphan['metadata']['assignment_confidence'] = 'low'
                            orphan['metadata']['rescued_by'] = 'llm_fallback_nearest'
                            if not self._assign_chunk_to_path(mapped_tree, orphan, nearest['path']):
                                remaining_orphans.append(orphan)
                        else:
                            remaining_orphans.append(orphan)
                    else:
                        remaining_orphans.append(orphan)
            except Exception as e:
                log.error(f"LangChain LLM mapping failed for chunk {orphan.get('chunk_id', 'N/A')}: {e}")
                remaining_orphans.append(orphan)
        return mapped_tree, remaining_orphans

    def _run_cohesion_pass(self, mapped_tree, orphans, all_chunks_in_order):
        """
        A robust, deterministic pass to "rescue" orphans by checking their neighbors
        and their graph-based references. This version correctly inserts rescued
        chunks in their original order.
        """
        log.info(f"--- Running Orphan Cohesion Pass for {len(orphans)} orphans ---")
        if not orphans:
            return mapped_tree, []

        # --- STAGE 1: Build necessary data structures ---
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

        # --- STAGE 2: Mission Planning ---
        rescue_missions = []
        still_orphaned = []
        
        soft_margin = self.config.get('soft_accept_margin', 0.05)
        similarity_threshold = self.config.get('similarity_threshold', 0.7)
        fallback_distance = 1.0 - max(similarity_threshold - soft_margin, 0.0)
        
        for orphan in orphans:
            orphan_id = orphan.get('chunk_id')
            if orphan_id is None:
                still_orphaned.append(orphan)
                continue
            
            rescued = False
            
            # Method 1: Neighbor Cohesion (Extended)
            prev_chunk_path = chunk_id_to_path_map.get(orphan_id - 1)
            next_chunk_path = chunk_id_to_path_map.get(orphan_id + 1)
            if prev_chunk_path and next_chunk_path and prev_chunk_path == next_chunk_path:
                # Insert after its preceding neighbor when both neighbors are in the same section
                mission = {'orphan': orphan, 'target_path': prev_chunk_path, 'insert_after_id': orphan_id - 1, 'insert_before_id': None, 'method': 'neighbor_both'}
                rescue_missions.append(mission)
                log.info(f"  -> Planning rescue for orphan {orphan_id} via Neighbor Cohesion (both neighbors match).")
                rescued = True
            elif prev_chunk_path and not next_chunk_path:
                # Only previous neighbor exists: insert after previous
                mission = {'orphan': orphan, 'target_path': prev_chunk_path, 'insert_after_id': orphan_id - 1, 'insert_before_id': None, 'method': 'neighbor_prev_only'}
                rescue_missions.append(mission)
                log.info(f"  -> Planning rescue for orphan {orphan_id} via Neighbor Cohesion (prev only).")
                rescued = True
            elif next_chunk_path and not prev_chunk_path:
                # Only next neighbor exists: insert before next
                mission = {'orphan': orphan, 'target_path': next_chunk_path, 'insert_after_id': None, 'insert_before_id': orphan_id + 1, 'method': 'neighbor_next_only'}
                rescue_missions.append(mission)
                log.info(f"  -> Planning rescue for orphan {orphan_id} via Neighbor Cohesion (next only).")
                rescued = True
            elif prev_chunk_path and next_chunk_path and prev_chunk_path != next_chunk_path:
                # Neighbors disagree: bias to previous section to maintain narrative flow
                mission = {'orphan': orphan, 'target_path': prev_chunk_path, 'insert_after_id': orphan_id - 1, 'insert_before_id': None, 'method': 'neighbor_disagree_prev_bias'}
                rescue_missions.append(mission)
                log.info(f"  -> Planning rescue for orphan {orphan_id} via Neighbor Cohesion (neighbors disagree, prev-biased).")
                rescued = True
            
            # Method 2: Graph Cohesion using outgoing references (if not rescued by neighbor)
            if not rescued and reference_graph.has_node(orphan_id):
                for _, target_id in reference_graph.out_edges(orphan_id):
                    target_path = chunk_id_to_path_map.get(target_id)
                    if target_path:
                        mission = {'orphan': orphan, 'target_path': target_path, 'insert_after_id': None, 'insert_before_id': None, 'method': f'graph_out->{target_id}'}
                        rescue_missions.append(mission)
                        log.info(f"  -> Planning rescue for orphan {orphan_id} via Graph Cohesion (references chunk {target_id}).")
                        rescued = True
                        break
            
            # Method 3: Graph Cohesion using incoming references (commonly for figures/tables/equations)
            if not rescued and reference_graph.has_node(orphan_id):
                in_neighbors = []
                for source_id, _ in reference_graph.in_edges(orphan_id):
                    path = chunk_id_to_path_map.get(source_id)
                    if path:
                        in_neighbors.append((source_id, tuple(path)))
                if in_neighbors:
                    # Choose the most common target section among referrers
                    from collections import Counter
                    section_counts = Counter([p for _, p in in_neighbors])
                    chosen_section, _ = section_counts.most_common(1)[0]
                    # Anchor near the first referring chunk in that section if possible
                    anchor_id = None
                    for source_id, p in in_neighbors:
                        if p == chosen_section:
                            anchor_id = source_id
                            break
                    mission = {
                        'orphan': orphan,
                        'target_path': list(chosen_section),
                        'insert_after_id': anchor_id,
                        'insert_before_id': None,
                        'method': 'graph_in_majority'
                    }
                    rescue_missions.append(mission)
                    log.info(f"  -> Planning rescue for orphan {orphan_id} via Incoming Graph Cohesion (majority of referrers).")
                    rescued = True
            
            # Method 4: Nearest-candidate fallback if sufficiently close
            if not rescued:
                cand = orphan.get('metadata', {}).get('candidate_sections', [])
                if cand:
                    nearest = cand[0]
                    if float(nearest.get('distance', 1.0)) <= fallback_distance:
                        mission = {
                            'orphan': orphan,
                            'target_path': nearest['path'],
                            'insert_after_id': None,
                            'insert_before_id': None,
                            'method': 'candidate_nearest'
                        }
                        rescue_missions.append(mission)
                        log.info(f"  -> Planning rescue for orphan {orphan_id} via Nearest Candidate (distance {nearest.get('distance'):.3f}).")
                        rescued = True

            if not rescued:
                still_orphaned.append(orphan)

        log.info(f"  -> Cohesion Pass Analysis complete. Planned {len(rescue_missions)} rescue missions.")

        # Sort missions by target_path and chunk_id to preserve order when multiple rescues land in same section
        try:
            rescue_missions.sort(key=lambda m: (tuple(m['target_path']), m['orphan'].get('chunk_id', float('inf'))))
        except Exception:
            pass

        # --- STAGE 3: Mission Execution ---
        for mission in rescue_missions:
            orphan_to_rescue = mission['orphan']
            target_path = mission['target_path']
            insert_after_id = mission['insert_after_id']
            insert_before_id = mission['insert_before_id']
            
            try:
                parent_node = mapped_tree
                for part in target_path[:-1]:
                    parent_node = parent_node[part]['subsections']
                target_node = parent_node[target_path[-1]]
                
                # Update metadata BEFORE assignment
                orphan_to_rescue.setdefault('metadata', {})
                orphan_to_rescue['metadata']['hierarchy_path'] = target_path
                orphan_to_rescue['parent_section'] = ' -> '.join(target_path)
                orphan_to_rescue['metadata']['soft_assigned'] = True
                orphan_to_rescue['metadata']['assignment_confidence'] = 'low'
                orphan_to_rescue['metadata']['rescued_by'] = mission.get('method', 'cohesion')
                
                # Ensure target node has a chunks list to insert into
                if 'chunks' not in target_node or not isinstance(target_node.get('chunks'), list):
                    target_node['chunks'] = []

                if insert_before_id is not None:
                    # Find the index of the neighbor and insert before it
                    neighbor_index = -1
                    for i, chunk in enumerate(target_node['chunks']):
                        if chunk.get('chunk_id') == insert_before_id:
                            neighbor_index = i
                            break
                    if neighbor_index != -1:
                        target_node['chunks'].insert(neighbor_index, orphan_to_rescue)
                    else:
                        # If anchor not found, insert at beginning as a reasonable default
                        target_node['chunks'].insert(0, orphan_to_rescue)
                elif insert_after_id is not None:
                    # Find the index of the neighbor and insert after it
                    neighbor_index = -1
                    for i, chunk in enumerate(target_node['chunks']):
                        if chunk.get('chunk_id') == insert_after_id:
                            neighbor_index = i
                            break
                    if neighbor_index != -1:
                        target_node['chunks'].insert(neighbor_index + 1, orphan_to_rescue)
                    else:
                        target_node['chunks'].append(orphan_to_rescue)
                else:
                    # If rescued by graph without anchor, just append to the end of the section
                    target_node['chunks'].append(orphan_to_rescue)

            except Exception as e:
                log.error(f"  -> Mission failed for chunk {orphan_to_rescue.get('chunk_id')}: {e}")
                still_orphaned.append(orphan_to_rescue)

        return mapped_tree, still_orphaned

    def _run_graph_boost_pass(self, similarity_matrix, all_chunks_in_order):
        """
        Adjusts the semantic similarity matrix by boosting scores based on the
        pre-built structural graph from the KnowledgeGraphProcessor.
        """
        log.info("--- Mapper Pass: Graph-Based Confidence Boosting ---")
        
        # Get the structural graph directly from the processor instance
        reference_graph = self.kg_processor.structural_graph
        if not reference_graph:
            log.warning("Structural graph not available in KG Processor. Skipping graph boost pass.")
            return similarity_matrix

        CONFIDENCE_THRESHOLD = self.config.get('confidence_threshold', 0.6)
        BOOST_AMOUNT = self.config.get('boost_amount', 0.3)
        
        best_match_indices = np.argmax(similarity_matrix, axis=1)
        best_match_scores = np.max(similarity_matrix, axis=1)
        
        confident_placements = {
            chunk.get('chunk_id'): best_match_indices[i] 
            for i, chunk in enumerate(all_chunks_in_order)
            if chunk.get('chunk_id') is not None and best_match_scores[i] >= CONFIDENCE_THRESHOLD
        }
        
        # Use the 'type' attribute we added to the edges for more precise boosting
        for source_id, target_id, data in reference_graph.edges(data=True):
            if target_id in confident_placements and source_id < similarity_matrix.shape[0]:
                # We can even have different boost amounts for different link types
                edge_type = data.get('type')
                current_boost = BOOST_AMOUNT
                if edge_type == 'reference': # A \ref is a very strong link
                    current_boost = BOOST_AMOUNT * 1.5
                
                target_section_index = confident_placements[target_id]
                original_score = similarity_matrix[source_id, target_section_index]
                similarity_matrix[source_id, target_section_index] = min(1.0, original_score + current_boost)

        return similarity_matrix

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