
import logging
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import networkx with explicit handling
try:
    import networkx as nx
    from networkx.readwrite import json_graph
    NETWORKX_AVAILABLE = True
except ImportError as e:
    print(f"NetworkX import error: {e}")
    # Create mock objects to prevent import errors
    nx = None
    json_graph = None
    NETWORKX_AVAILABLE = False

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
        self.unified_graph = None

    @property
    def embeddings(self):
        """Lazy-loads and caches the chunk embeddings."""
        if self._embeddings is None:
            log.info("  -> KG Processor: Generating embeddings for all chunks...")
            self._embeddings = self.embedding_model.encode(self.chunk_contents, show_progress_bar=False)
        return self._embeddings

    def build_graphs(self, semantic_top_k=3, semantic_threshold=0.75):
        """Builds both the semantic and structural graphs."""
        if not NETWORKX_AVAILABLE:
            log.warning("NetworkX not available. Skipping graph building.")
            return
        log.info("--- KG Processor: Building Knowledge Graphs ---")
        self.build_semantic_graph(top_k=semantic_top_k, threshold=semantic_threshold)
        self.build_structural_graph()

    def build_semantic_graph(self, top_k=3, threshold=0.75):
        """
        Builds a graph where an edge represents high semantic similarity.
        """
        if not NETWORKX_AVAILABLE:
            log.warning("NetworkX not available. Skipping semantic graph building.")
            return
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
        if not NETWORKX_AVAILABLE:
            log.warning("NetworkX not available. Skipping structural graph building.")
            return
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
        if not NETWORKX_AVAILABLE:
            log.warning("NetworkX not available. Cannot find core concepts.")
            return []
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
        if not NETWORKX_AVAILABLE:
            log.warning("NetworkX not available. Cannot find semantic neighbors.")
            return []
        if not self.semantic_graph or not self.semantic_graph.has_node(chunk_id):
            return []
        
        # Use a breadth-first search to find neighbors
        neighbors = list(nx.bfs_tree(self.semantic_graph, source=chunk_id, depth_limit=depth).nodes())
        return [self.chunk_map.get(nid) for nid in neighbors if nid != chunk_id and self.chunk_map.get(nid)]

    # -------------------- Phase 2: Unified Graph + Utilities --------------------
    def _section_node_id(self, path_list):
        if not path_list:
            return "section::ROOT"
        return "section::" + " -> ".join(path_list)

    def _hash_content(self, text):
        try:
            return hashlib.sha1(text.encode("utf-8")).hexdigest()
        except Exception:
            return None

    def _compute_cohesion_score(self, text):
        # Lightweight cohesion: average cosine similarity of adjacent sentence embeddings
        if not text:
            return None
        # naive sentence split
        sentences = [s.strip() for s in text.replace("\n", " ").split('.') if s.strip()]
        if len(sentences) < 2:
            return None
        sent_emb = self.embedding_model.encode(sentences, show_progress_bar=False)
        if len(sentences) == 2:
            sim = cosine_similarity([sent_emb[0]], [sent_emb[1]])[0][0]
            return float(sim)
        sims = []
        for i in range(len(sentences)-1):
            a, b = sent_emb[i], sent_emb[i+1]
            sims.append(cosine_similarity([a], [b])[0][0])
        return float(np.mean(sims)) if sims else None

    def build_unified_graph(self, semantic_top_k=5, semantic_threshold=0.8, duplicate_threshold=0.95, compute_cohesion=False):
        """Build a unified MultiDiGraph with chunk and section nodes and multiple edge layers."""
        if not NETWORKX_AVAILABLE:
            log.warning("NetworkX not available. Cannot build unified graph.")
            return None
        # Ensure base graphs
        if self.semantic_graph is None or self.structural_graph is None:
            self.build_graphs(semantic_top_k, semantic_threshold)

        G = nx.MultiDiGraph()

        # 1) Add chunk nodes with attributes
        for chunk in self.all_chunks:
            cid = chunk.get('chunk_id')
            if cid is None:
                continue
            meta = chunk.get('metadata', {}) or {}
            attrs = {
                'node_type': 'chunk',
                'token_count': meta.get('token_count') or len(chunk.get('content', "")),
                'block_type': meta.get('block_type'),
                'source_path': meta.get('source_path'),
                'hierarchy_path': meta.get('hierarchy_path'),
                'content_hash': self._hash_content(chunk.get('content', '')),
            }
            # Optional signals if present
            if 'assignment_score' in meta:
                attrs['mapping_confidence'] = meta.get('assignment_score')
            if 'candidate_sections' in meta:
                attrs['candidate_sections'] = meta.get('candidate_sections')
            # Optional cohesion computation
            if compute_cohesion:
                attrs['cohesion_score'] = self._compute_cohesion_score(chunk.get('content', ''))
            G.add_node(cid, **attrs)

        # 2) Add section nodes and mapping edges
        seen_sections = set()
        for chunk in self.all_chunks:
            cid = chunk.get('chunk_id')
            path = (chunk.get('metadata', {}) or {}).get('hierarchy_path') or []
            sid = self._section_node_id(path)
            if sid not in seen_sections:
                seen_sections.add(sid)
                G.add_node(sid, node_type='section', path=path, depth=len(path))
            # mapping edge chunk -> section
            score = (chunk.get('metadata', {}) or {}).get('assignment_score')
            G.add_edge(cid, sid, edge_type='mapping', score=score, rank=1)

        # 3) Import semantic edges; also mark duplicates when above duplicate_threshold
        if self.semantic_graph is not None:
            for u, v, data in self.semantic_graph.edges(data=True):
                w = data.get('weight')
                G.add_edge(u, v, edge_type='semantic', weight=w)
                if w is not None and w >= duplicate_threshold:
                    G.add_edge(u, v, edge_type='duplicate', weight=w)

        # 4) Import structural edges
        if self.structural_graph is not None:
            for u, v, data in self.structural_graph.edges(data=True):
                et = data.get('type') or 'structural'
                G.add_edge(u, v, edge_type=et)

        self.unified_graph = G
        log.info(f"  -> Unified KG built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def get_chunk_kg_scores(self, chunk_ids=None):
        """Extract comprehensive KG-based scores for chunks to enhance section mapping.
        
        Returns dict mapping chunk_id -> {
            'centrality_score': float,
            'cohesion_score': float, 
            'semantic_connectivity': float,
            'structural_importance': float,
            'composite_score': float
        }
        """
        if self.unified_graph is None:
            log.warning("Unified graph not built. Cannot compute KG scores.")
            return {}
            
        if chunk_ids is None:
            chunk_ids = [n for n in self.unified_graph.nodes() 
                        if self.unified_graph.nodes[n].get('node_type') == 'chunk']
        
        scores = {}
        
        # Compute centrality measures
        try:
            degree_centrality = nx.degree_centrality(self.unified_graph)
            betweenness_centrality = nx.betweenness_centrality(self.unified_graph)
            closeness_centrality = nx.closeness_centrality(self.unified_graph)
        except Exception as e:
            log.warning(f"Failed to compute centrality measures: {e}")
            degree_centrality = betweenness_centrality = closeness_centrality = {}
        
        for chunk_id in chunk_ids:
            if chunk_id not in self.unified_graph.nodes():
                continue
                
            node_attrs = self.unified_graph.nodes[chunk_id]
            
            # 1. Centrality score (combination of degree, betweenness, closeness)
            deg_cent = degree_centrality.get(chunk_id, 0.0)
            bet_cent = betweenness_centrality.get(chunk_id, 0.0) 
            clo_cent = closeness_centrality.get(chunk_id, 0.0)
            centrality_score = (deg_cent * 0.5 + bet_cent * 0.3 + clo_cent * 0.2)
            
            # 2. Cohesion score (from node attributes or compute)
            cohesion_score = node_attrs.get('cohesion_score', 0.0) or 0.0
            
            # 3. Semantic connectivity (count and strength of semantic edges)
            semantic_edges = [(u, v, d) for u, v, d in self.unified_graph.edges(chunk_id, data=True) 
                             if d.get('edge_type') == 'semantic']
            semantic_weights = [d.get('weight', 0.0) for _, _, d in semantic_edges]
            semantic_connectivity = np.mean(semantic_weights) if semantic_weights else 0.0
            
            # 4. Structural importance (mapping confidence + hierarchy depth)
            mapping_confidence = node_attrs.get('mapping_confidence', 0.0) or 0.0
            hierarchy_depth = len(node_attrs.get('hierarchy_path', [])) 
            # Normalize depth (deeper = more specific = higher importance)
            structural_importance = (mapping_confidence * 0.7 + 
                                   min(hierarchy_depth / 5.0, 1.0) * 0.3)
            
            # 5. Composite score (weighted combination)
            composite_score = (centrality_score * 0.3 + 
                             cohesion_score * 0.2 + 
                             semantic_connectivity * 0.3 + 
                             structural_importance * 0.2)
            
            scores[chunk_id] = {
                'centrality_score': float(centrality_score),
                'cohesion_score': float(cohesion_score),
                'semantic_connectivity': float(semantic_connectivity), 
                'structural_importance': float(structural_importance),
                'composite_score': float(composite_score)
            }
            
        log.info(f"  -> Computed KG scores for {len(scores)} chunks.")
        return scores
        
    def get_section_affinity_scores(self, chunk_id, candidate_sections):
        """Compute KG-based affinity scores between a chunk and candidate sections.
        
        Args:
            chunk_id: ID of the chunk
            candidate_sections: List of section names or paths
            
        Returns:
            dict mapping section -> affinity_score
        """
        if self.unified_graph is None or chunk_id not in self.unified_graph.nodes():
            return {}
            
        affinities = {}
        
        for section in candidate_sections:
            # Find section nodes that match this section name/path
            section_nodes = [n for n in self.unified_graph.nodes() 
                           if (self.unified_graph.nodes[n].get('node_type') == 'section' and
                               self._section_matches(n, section))]
            
            if not section_nodes:
                affinities[section] = 0.0
                continue
                
            # Compute affinity based on:
            # 1. Direct mapping edge strength
            # 2. Semantic similarity to chunks already in this section
            # 3. Structural proximity
            
            section_affinities = []
            for section_node in section_nodes:
                affinity = 0.0
                
                # Direct mapping edge
                if self.unified_graph.has_edge(chunk_id, section_node):
                    edge_data = self.unified_graph.get_edge_data(chunk_id, section_node)
                    for edge_key, edge_attrs in edge_data.items():
                        if edge_attrs.get('edge_type') == 'mapping':
                            affinity += edge_attrs.get('score', 0.0) or 0.0
                
                # Semantic similarity to section's existing chunks
                section_chunks = [n for n in self.unified_graph.predecessors(section_node)
                                if self.unified_graph.nodes[n].get('node_type') == 'chunk']
                
                if section_chunks:
                    semantic_scores = []
                    for other_chunk in section_chunks:
                        if self.unified_graph.has_edge(chunk_id, other_chunk):
                            edge_data = self.unified_graph.get_edge_data(chunk_id, other_chunk)
                            for edge_attrs in edge_data.values():
                                if edge_attrs.get('edge_type') == 'semantic':
                                    semantic_scores.append(edge_attrs.get('weight', 0.0))
                    
                    if semantic_scores:
                        affinity += np.mean(semantic_scores) * 0.5
                
                section_affinities.append(affinity)
            
            affinities[section] = max(section_affinities) if section_affinities else 0.0
            
        return affinities
        
    def _section_matches(self, section_node_id, section_name):
        """Check if a section node matches a given section name."""
        node_attrs = self.unified_graph.nodes[section_node_id]
        path = node_attrs.get('path', [])
        
        # Simple matching - could be enhanced
        if isinstance(section_name, str):
            return section_name in str(path) or section_name == section_node_id
        return False

    def dump_unified_graph(self, output_dir, fmt='json', filename='knowledge_graph_unified.json'):
        """Serialize the unified graph for offline inspection."""
        if not hasattr(self, 'unified_graph') or self.unified_graph is None:
            log.warning("Unified graph not built; nothing to dump.")
            return None
        os_path = None
        try:
            import os
            import json
            import numpy as np
            
            def convert_numpy_types(obj):
                """Convert numpy types to native Python types for JSON serialization."""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            if fmt.lower() == 'json':
                # Convert to JSON-serializable format
                data = nx.node_link_data(self.unified_graph, edges="links")
                # Convert numpy types to native Python types
                data = convert_numpy_types(data)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            elif fmt.lower() == 'graphml':
                nx.write_graphml(self.unified_graph, output_path)
            else:
                log.error(f"Unsupported format: {fmt}")
                return None
            log.info(f"Unified graph dumped to: {output_path}")
            return output_path
            
        except Exception as e:
            log.error(f"Failed to dump unified KG: {e}")
            return None