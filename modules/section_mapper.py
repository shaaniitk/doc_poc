"""
State-of-the-Art Section Mapping with Semantic Similarity.

This module assigns document chunks to a predefined document skeleton. Instead of
relying on fragile section title matching, it uses a sentence-transformer model
to perform a semantic comparison between the content of each chunk and the
description of each target section defined in the config.

Key Features:
- Semantic Assignment: Uses vector embeddings and cosine similarity to find the
  best thematic fit for each chunk, which is far more accurate than regex.
- Robustness: Can correctly assign content from a section named "Our Method" to a
  target section named "4. Proof-of-Work" if their meanings align.
- Orphan Handling: Chunks that don't semantically match any target section above
  a configured threshold are placed in a special "Orphaned_Content" section for
  review, ensuring no content is lost.
- Efficiency: Pre-calculates embeddings for the document skeleton and processes
  chunk embeddings in batches for high performance.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .error_handler import ProcessingError
from .embedding_client import UnifiedEmbeddingClient
from config import DOCUMENT_TEMPLATES, SEMANTIC_MAPPING_CONFIG, KG_CONFIG

# --- Main Semantic Mapping Logic ---

class SemanticMapper:
    """
    Assigns chunks to document sections using semantic similarity.
    """
    def __init__(self, template_name="bitcoin_paper", kg_processor=None):
        self.config = SEMANTIC_MAPPING_CONFIG
        self.kg_config = KG_CONFIG
        self.skeleton = DOCUMENT_TEMPLATES.get(template_name)
        if not self.skeleton:
            raise ProcessingError(f"Document template '{template_name}' not found in config.")

        self.model = self._load_model()
        self.kg_processor = kg_processor
        
        # Pre-calculate embeddings for the target sections for efficiency.
        # This is a critical optimization.
        self.section_names, self.section_embeddings = self._prepare_skeleton_embeddings()

    def _load_model(self):
        """Loads the embedding model specified in the config."""
        try:
            return UnifiedEmbeddingClient(self.config)
        except Exception as e:
            raise ProcessingError(f"Failed to load semantic mapping model: {e}")

    def _prepare_skeleton_embeddings(self):
    #Recursively flattens the hierarchical skeleton and generates embeddings for all nodes.
    
    # The flattened list will contain dicts like {'path': ['Section', 'Subsection'], 'description': '...'}
        self.flat_skeleton = self._flatten_skeleton_recursive(self.skeleton, [])
        
        if not self.flat_skeleton:
            raise ProcessingError("Document template is empty or could not be processed.")

        # self.section_paths will store the list of paths, e.g., [['Section'], ['Section', 'Subsection']]
        self.section_paths = [s['path'] for s in self.flat_skeleton]
        section_descriptions = [s['description'] for s in self.flat_skeleton]
        
        embeddings = self.model.encode(section_descriptions, show_progress_bar=False)
        return self.section_paths, embeddings                       

    def _flatten_skeleton_recursive(self, node_level, current_path):
        """
        A helper function to traverse the nested skeleton dictionary or list.
        """
        flat_list = []
        
        # Handle list structure (like bitcoin_paper template)
        if isinstance(node_level, list):
            for item in node_level:
                if isinstance(item, dict) and 'section' in item:
                    new_path = current_path + [item['section']]
                    flat_list.append({
                        'path': new_path,
                        'description': item.get('description', '')
                    })
                    # Handle subsections in list structure
                    if item.get('subsections'):
                        flat_list.extend(
                            self._flatten_skeleton_recursive(item['subsections'], new_path)
                        )
        # Handle dictionary structure (like bitcoin_paper_hierarchical template)
        elif isinstance(node_level, dict):
            for title, data in node_level.items():
                new_path = current_path + [title]
                flat_list.append({
                    'path': new_path,
                    'description': data.get('description', '') if isinstance(data, dict) else str(data)
                })
                # Handle subsections in dictionary structure
                if isinstance(data, dict) and data.get('subsections'):
                    flat_list.extend(
                        self._flatten_skeleton_recursive(data['subsections'], new_path)
                    )
        return flat_list

    def assign_chunks(self, chunks, sections, threshold: float = 0.5, top_k: int = 3):
        """Assign chunks to sections with confidence scores and candidate list.
        
        Enhanced with KG scores when available for improved assignment accuracy.

        Returns list of dicts with keys: 'chunk_id', 'section', 'assignment_score', 'candidate_sections'
        and preserves original fields in metadata for backward compatibility.
        """
        chunk_texts = [chunk['content'] for chunk in chunks]
        chunk_embeddings = self.model.encode(chunk_texts)

        # Use pre-calculated section embeddings based on descriptions instead of names
        # This provides much better semantic matching
        section_embeddings = self.section_embeddings
        section_texts = [self._flatten_section_name(s) for s in sections]

        sim_matrix = cosine_similarity(chunk_embeddings, section_embeddings)
        
        # Get KG scores if available and enabled
        kg_scores = None
        section_affinities = None
        if (self.kg_processor and 
            self.kg_config.get('enhance_section_mapping', False) and 
            hasattr(self.kg_processor, 'unified_graph') and 
            self.kg_processor.unified_graph is not None):
            try:
                kg_scores = self.kg_processor.get_chunk_kg_scores(chunks)
                section_affinities = self.kg_processor.get_section_affinity_scores(chunks, section_texts)
            except Exception as e:
                print(f"Warning: Failed to get KG scores: {e}")
                kg_scores = None
                section_affinities = None

        assignments = []
        for i, chunk in enumerate(chunks):
            sims = sim_matrix[i]
            
            # Apply KG enhancement if available
            if kg_scores and section_affinities:
                enhanced_sims = self._enhance_scores_with_kg(
                    sims, kg_scores[i], section_affinities[i], section_texts
                )
            else:
                enhanced_sims = sims
            
            best_idx = int(np.argmax(enhanced_sims))
            best_score = float(enhanced_sims[best_idx])

            # build top-k candidates (idx, score) using enhanced scores
            top_indices = np.argpartition(enhanced_sims, -top_k)[-top_k:]
            top_sorted = sorted([(int(idx), float(enhanced_sims[idx])) for idx in top_indices], key=lambda x: x[1], reverse=True)
            candidates = [
                {
                    'section': section_texts[idx],
                    'index': int(idx),
                    'score': float(score),
                    'embedding_score': float(sims[idx]),  # Original embedding score
                    'kg_enhanced': kg_scores is not None
                }
                for idx, score in top_sorted
            ]

            assigned_section = section_texts[best_idx] if best_score >= threshold else 'Orphaned_Content'
            result = {
                'chunk_id': chunk.get('chunk_id'),
                'section': assigned_section,
                'assignment_score': best_score,
                'candidate_sections': candidates,
            }

            # also enrich chunk metadata in-place if present
            meta = chunk.get('metadata') or {}
            meta['assignment_score'] = best_score
            meta['candidate_sections'] = candidates
            chunk['metadata'] = meta

            assignments.append(result)

        return assignments

    def _flatten_section_name(self, section):
        """Convert section dict to a flattened string representation."""
        if isinstance(section, str):
            return section
        elif isinstance(section, list):
            # Handle list paths by joining them into a readable section name
            return ' > '.join(str(item) for item in section)
        elif isinstance(section, dict):
            # Extract meaningful text from section structure
            title = section.get('title', '')
            description = section.get('description', '')
            return f"{title} {description}".strip()
        else:
            return str(section)
    
    def _enhance_scores_with_kg(self, embedding_sims, chunk_kg_score, chunk_section_affinities, section_texts):
        """Enhance embedding similarity scores with KG-based scores.
        
        Args:
            embedding_sims: Array of embedding similarity scores for each section
            chunk_kg_score: Dict with KG metrics for this chunk
            chunk_section_affinities: Dict mapping section names to affinity scores
            section_texts: List of section names
            
        Returns:
            Enhanced similarity scores combining embedding and KG information
        """
        enhanced_sims = embedding_sims.copy()
        
        # Get weights from config
        embedding_weight = self.kg_config.get('embedding_weight', 0.6)
        kg_weight = self.kg_config.get('kg_weight', 0.4)
        affinity_boost = self.kg_config.get('affinity_boost_factor', 1.2)
        use_section_affinity = self.kg_config.get('use_section_affinity', True)
        
        # Get chunk's composite KG score (normalized 0-1)
        chunk_kg_composite = chunk_kg_score.get('composite_score', 0.0)
        
        for i, section_name in enumerate(section_texts):
            # Base enhanced score: weighted combination of embedding and KG scores
            base_enhanced = (embedding_weight * embedding_sims[i] + 
                           kg_weight * chunk_kg_composite)
            
            # Apply section affinity boost if available
            if use_section_affinity and section_name in chunk_section_affinities:
                section_affinity = chunk_section_affinities[section_name]
                # Boost score based on KG-derived section affinity
                enhanced_sims[i] = base_enhanced * (1.0 + section_affinity * (affinity_boost - 1.0))
            else:
                enhanced_sims[i] = base_enhanced
                
        return enhanced_sims

    def _create_empty_skeleton(self, node_level, current_path=None):
        """
        Recursively creates a deep copy of the skeleton, preserving all metadata
        (like 'generative' flags) and setting up empty 'chunks' lists.
        Additionally, annotates each node with its hierarchy_path for downstream analysis.
        """
        if current_path is None:
            current_path = []
        new_level = {}
        for title, data in node_level.items():
            path_here = current_path + [title]
            # Start by copying ALL keys from the template (prompt, description, generative, etc.)
            new_node = data.copy()
            # Ensure/merge metadata and set hierarchy_path
            existing_meta = new_node.get('metadata', {}) if isinstance(new_node.get('metadata', {}), dict) else {}
            existing_meta['hierarchy_path'] = path_here
            new_node['metadata'] = existing_meta
            # Then, specifically set the 'chunks' list to be empty.
            new_node['chunks'] = []
            # Finally, recurse to build the subsections.
            new_node['subsections'] = self._create_empty_skeleton(data.get('subsections', {}), path_here)
            new_level[title] = new_node
        return new_level

# --- Top-Level Functions ---

def get_document_skeleton(template_name="bitcoin_paper"):
    """
    Retrieves the document skeleton from the configuration.
    """
    return DOCUMENT_TEMPLATES.get(template_name, DOCUMENT_TEMPLATES["bitcoin_paper"])

def assign_chunks_to_skeleton(grouped_chunks, template_name="bitcoin_paper", kg_processor=None):
    """
    Main entry point for assigning chunks to the document skeleton.
    This function flattens the grouped chunks and uses the SemanticMapper.

    Args:
        grouped_chunks (dict): A dictionary of chunks grouped by their original section.
        template_name (str): The name of the document template to use.
        kg_processor (KnowledgeGraphProcessor, optional): KG processor for enhanced scoring.

    Returns:
        dict: A dictionary mapping target skeleton sections to lists of assigned chunks.
    """
    try:
        mapper = SemanticMapper(template_name, kg_processor=kg_processor)
        
        # Flatten the dictionary of chunks into a single list for processing.
        all_chunks = [chunk for section_chunks in grouped_chunks.values() for chunk in section_chunks]
        
        # Add chunk_id to chunks that don't have one (for matching purposes)
        for i, chunk in enumerate(all_chunks):
            if 'chunk_id' not in chunk:
                chunk['chunk_id'] = f"chunk_{i}"
        
        assignments = mapper.assign_chunks(all_chunks, mapper.section_names)
        
        # Transform assignments list into dictionary format expected by tests
        result = {}
        for assignment in assignments:
            section = assignment['section']
            chunk_id = assignment['chunk_id']
            
            # Find the original chunk to include in the result
            chunk = next((c for c in all_chunks if c.get('chunk_id') == chunk_id), None)
            if chunk:
                if section not in result:
                    result[section] = []
                result[section].append(chunk)
        
        # Combine chunks assigned to the same section into a single chunk with combined content
        for section in result:
            if len(result[section]) > 1:
                combined_content = " ".join([chunk['content'] for chunk in result[section]])
                # Keep the first chunk but update its content to include all chunks
                result[section] = [{
                    'content': combined_content,
                    'metadata': result[section][0].get('metadata', {})
                }]
        
        return result
        
    except Exception as e:
        # Catch and re-raise as a more specific error.
        raise ProcessingError(f"Failed to assign chunks to skeleton: {e}")

