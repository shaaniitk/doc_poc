
import logging
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

class KnowledgeGraphProcessor:
    """
    A dedicated module for creating and analyzing both semantic and structural
    knowledge graphs from a list of document chunks.
    """
    def __init__(self, all_chunks, embedding_model):
        self.all_chunks = all_chunks
        self.embedding_model = embedding_model
        
        self.chunk_map = {c['chunk_id']: c for c in all_chunks if 'chunk_id' in c}
        self.chunk_contents = [c.get('content', '') for c in all_chunks]
        
        # --- Lazy-loaded properties ---
        self._embeddings = None
        self.semantic_graph = None
        self.structural_graph = None

    @property
    def embeddings(self):
        """Lazy-loads and caches the chunk embeddings."""
        if self._embeddings is None:
            log.info("  -> KG Processor: Generating embeddings for all chunks...")
            self._embeddings = self.embedding_model.encode(self.chunk_contents, show_progress_bar=False)
        return self._embeddings

    def build_graphs(self, semantic_top_k=3, semantic_threshold=0.75):
        """Builds both the semantic and structural graphs."""
        log.info("--- KG Processor: Building Knowledge Graphs ---")
        self.build_semantic_graph(top_k=semantic_top_k, threshold=semantic_threshold)
        self.build_structural_graph()

    def build_semantic_graph(self, top_k=3, threshold=0.75):
        """
        Builds a graph where an edge represents high semantic similarity.
        """
        log.info("  -> Building semantic graph...")
        self.semantic_graph = nx.Graph() # Use an undirected graph for semantic similarity
        if len(self.all_chunks) < 2: return

        similarity_matrix = cosine_similarity(self.embeddings)
        
        for i in range(len(self.all_chunks)):
            source_id = self.all_chunks[i].get('chunk_id')
            if source_id is None: continue
            self.semantic_graph.add_node(source_id)

            # Find top_k most similar chunks (excluding itself)
            sim_scores = similarity_matrix[i]
            top_indices = np.argpartition(sim_scores, -top_k-1)[-top_k-1:]
            
            for j in top_indices:
                if i == j: continue
                score = sim_scores[j]
                if score >= threshold:
                    target_id = self.all_chunks[j].get('chunk_id')
                    if target_id is not None:
                        self.semantic_graph.add_edge(source_id, target_id, weight=score, type='semantic')

    def build_structural_graph(self):
        """
        Builds a graph where an edge represents a direct document link
        (e.g., preceding/succeeding chunk, or a \\ref->\\label link).
        """
        log.info("  -> Building structural graph...")
        self.structural_graph = nx.DiGraph() # Directed graph for structure
        label_to_chunk_id = {}

        # Pass 1: Add all nodes and find labels
        for chunk in self.all_chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id is None: continue
            self.structural_graph.add_node(chunk_id)
            for label in chunk.get('metadata', {}).get('labels', []):
                label_to_chunk_id[label] = chunk_id
        
        # Pass 2: Add edges
        for chunk in self.all_chunks:
            chunk_id = chunk.get('chunk_id')
            if chunk_id is None: continue

            # Add edges to next/prev chunks
            next_id = chunk.get('metadata', {}).get('next_chunk_id')
            if next_id is not None:
                self.structural_graph.add_edge(chunk_id, next_id, type='sequence')

            # Add edges for LaTeX references
            for ref in chunk.get('metadata', {}).get('refs', []):
                target_chunk_id = label_to_chunk_id.get(ref)
                if target_chunk_id:
                    self.structural_graph.add_edge(chunk_id, target_chunk_id, type='reference')

    def get_core_concept_chunks(self, top_n=10):
        """
        Identifies the most central/important chunks in the semantic graph.
        """
        if not self.semantic_graph or not self.semantic_graph.nodes:
            log.warning("  -> Semantic graph not built or is empty. Cannot find core concepts.")
            return []
        
        # Use degree centrality to find the most connected nodes
        centrality = nx.degree_centrality(self.semantic_graph)
        sorted_chunks = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        core_concept_ids = [chunk_id for chunk_id, score in sorted_chunks[:top_n]]
        
        log.info(f"  -> Identified {len(core_concept_ids)} core concept chunks.")
        return [self.chunk_map.get(cid) for cid in core_concept_ids if self.chunk_map.get(cid)]

    def find_semantic_neighbors(self, chunk_id, depth=1):
        """
        Finds all chunks semantically related to a given chunk up to a certain depth.
        """
        if not self.semantic_graph or not self.semantic_graph.has_node(chunk_id):
            return []
        
        # Use a breadth-first search to find neighbors
        neighbors = list(nx.bfs_tree(self.semantic_graph, source=chunk_id, depth_limit=depth).nodes())
        return [self.chunk_map.get(nid) for nid in neighbors if nid != chunk_id and self.chunk_map.get(nid)]