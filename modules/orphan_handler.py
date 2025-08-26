
import logging
from .llm_handler import HierarchicalProcessingAgent
from .document_combiner import HierarchicalDocumentCombiner
from .llm_client import UnifiedLLMClient
from sentence_transformers.util import community_detection
from config import PROMPTS
import json

log = logging.getLogger(__name__)

class OrphanRemediator:
    """
    Analyzes, clusters, and intelligently reintegrates orphaned content.
    """
    def __init__(self, llm_agent: HierarchicalProcessingAgent):
        # We pass in the agent to get access to its LLM client and semantic validator
        self.agent = llm_agent
        self.llm_client = llm_agent.llm_client

    def remediate(self, tree, orphans):
        """
        The main entry point for the remediation process.
        
        Args:
            tree (dict): The main document tree to which new sections will be added.
            orphans (list): The list of orphaned chunk dictionaries.

        Returns:
            dict: The document tree, potentially with new sections grafted in.
        """
        log.info(f"-> Found {len(orphans)} orphaned chunks. Starting remediation...")
        if not orphans or len(orphans) < 2:
            log.info("-> Not enough orphaned chunks to form a new topic cluster. Appending as 'Uncategorized'.")
            if orphans:
                tree['Orphaned_Content'] = orphans
            return tree

        # --- Step 1: Cluster ---
        corpus_embeddings = self._get_orphan_embeddings(orphans)
        clusters = community_detection(corpus_embeddings, min_community_size=2, threshold=0.65)
        log.info(f"-> Clustered orphans into {len(clusters)} new topic(s).")
        
        # --- Step 2 & 3: Synthesize and Graft for each cluster ---
        grafting_combiner = HierarchicalDocumentCombiner()
        unclustered_orphans = orphans[:]

        for i, cluster_indices in enumerate(clusters):
            cluster_indices.sort(reverse=True)
            cluster_chunks = [unclustered_orphans.pop(j) for j in cluster_indices]
            
            try:
                new_title, new_node_data = self._create_new_node_from_cluster(cluster_chunks)
                log.info(f"-> New Topic Identified: '{new_title}'")
                
                # Use the combiner's grafting logic to find the best place for this new section
                tree = grafting_combiner._graft_new_structures(tree, {new_title: new_node_data})

            except Exception as e:
                log.error(f"-> Failed to remediate cluster {i+1}. Error: {e}")
        
        # Any remaining orphans that didn't form a cluster are added to the standard uncategorized section
        if unclustered_orphans:
            log.info(f"-> Appending {len(unclustered_orphans)} remaining unclustered orphans as 'Uncategorized'.")
            tree['Orphaned_Content'] = unclustered_orphans
            
        return tree

    def _get_orphan_embeddings(self, orphans):
        """Generates semantic vector embeddings for a list of orphan chunks."""
        orphan_contents = [c['content'] for c in orphans]
        return self.agent.semantic_validator.model.encode(orphan_contents, show_progress_bar=False)

    def _create_new_node_from_cluster(self, cluster_chunks):
        """Uses the LLM to synthesize a new title and node data from a cluster of chunks."""
        cluster_content = "\n\n---\n\n".join([c['content'] for c in cluster_chunks])
        
        # Synthesize a summary to understand the cluster's theme
        summary_prompt = PROMPTS['summarize_chunk_cluster'].format(chunk_cluster_content=cluster_content)
        summary = self.llm_client.call_llm([{"role": "user", "content": summary_prompt}])

        # Create a new title from that summary
        title_prompt = PROMPTS['create_title_from_summary'].format(content_summary=summary)
        new_title = self.llm_client.call_llm([{"role": "user", "content": title_prompt}])
        
        new_node_data = {
            "prompt": "You are a LaTeX expert. Refactor the following content, which has been grouped by topic, into a coherent and well-structured section. Preserve all technical details.",
            "description": summary, 
            "chunks": cluster_chunks, 
            "subsections": {}
        }
        
        return new_title, new_node_data
    
# --- ADD THIS ENTIRE NEW HELPER FUNCTION ---

def _run_orphan_cohesion_pass(mapped_tree, orphans, all_chunks_in_order):
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