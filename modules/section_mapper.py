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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .error_handler import ProcessingError
from config import DOCUMENT_TEMPLATES, SEMANTIC_MAPPING_CONFIG

# --- Main Semantic Mapping Logic ---

class SemanticMapper:
    """
    Assigns chunks to document sections using semantic similarity.
    """
    def __init__(self, template_name="bitcoin_paper"):
        self.config = SEMANTIC_MAPPING_CONFIG
        self.skeleton = DOCUMENT_TEMPLATES.get(template_name)
        if not self.skeleton:
            raise ProcessingError(f"Document template '{template_name}' not found in config.")

        self.model = self._load_model()
        
        # Pre-calculate embeddings for the target sections for efficiency.
        # This is a critical optimization.
        self.section_names, self.section_embeddings = self._prepare_skeleton_embeddings()

    def _load_model(self):
        """Loads the sentence-transformer model specified in the config."""
        try:
            model_name = self.config['model']
            return SentenceTransformer(model_name)
        except Exception as e:
            raise ProcessingError(f"Failed to load semantic mapping model '{model_name}': {e}")

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

    def assign_chunks(self, chunks):
        """
        Assigns a list of chunks to the skeleton sections based on semantic similarity.
        
        Args:
            chunks (list): A list of chunk dictionaries from the ASTChunker.

        Returns:
            dict: A dictionary mapping section names to lists of assigned chunks.
        """
        if not chunks:
            return {name: [] for name in self.section_names}

        # 1. Prepare chunk embeddings in a single batch for performance.
        chunk_contents = [c['content'] for c in chunks]
        chunk_embeddings = self.model.encode(chunk_contents, show_progress_bar=False)

        # 2. Calculate the similarity matrix between all chunks and all sections.
        # The result is a matrix where similarity_matrix[i, j] is the similarity
        # between chunk i and section j.
        similarity_matrix = cosine_similarity(chunk_embeddings, self.section_embeddings)

       # 3. Create a deep, empty copy of the hierarchical skeleton to populate.
        assignments = self._create_empty_skeleton(self.skeleton)
        assignments['Orphaned_Content'] = [] # For chunks that don't fit well anywhere.

        # 4. Find the best hierarchical path for each chunk.
        best_match_indices = np.argmax(similarity_matrix, axis=1)
        best_match_scores = np.max(similarity_matrix, axis=1)

        # 5. Assign chunks based on the best score and the similarity threshold.
        threshold = self.config['similarity_threshold']

        for i, chunk in enumerate(chunks):
            best_path_index = best_match_indices[i]
            score = best_match_scores[i]
            
            chunk['metadata']['assignment_score'] = float(score)

            if score >= threshold:
                # The best match is now a full hierarchical path
                best_path = self.section_paths[best_path_index]
                
               # Navigate to the parent of the target node
                target_parent = assignments
                # The path to the parent is all but the last element of the best_path
                for part in best_path[:-1]:
                    target_parent = target_parent[part]['subsections']
                
                # Access the correct node from the parent and append the chunk
                target_node = target_parent[best_path[-1]]
                target_node['chunks'].append(chunk)
                
            else:
                assignments['Orphaned_Content'].append(chunk)

        return assignments

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

# --- Top-Level Functions ---

def get_document_skeleton(template_name="bitcoin_paper"):
    """
    Retrieves the document skeleton from the configuration.
    """
    return DOCUMENT_TEMPLATES.get(template_name, DOCUMENT_TEMPLATES["bitcoin_paper"])

def assign_chunks_to_skeleton(grouped_chunks, template_name="bitcoin_paper"):
    """
    Main entry point for assigning chunks to the document skeleton.
    This function flattens the grouped chunks and uses the SemanticMapper.

    Args:
        grouped_chunks (dict): A dictionary of chunks grouped by their original section.
        template_name (str): The name of the document template to use.

    Returns:
        dict: A dictionary mapping target skeleton sections to lists of assigned chunks.
    """
    try:
        mapper = SemanticMapper(template_name)
        
        # Flatten the dictionary of chunks into a single list for processing.
        all_chunks = [chunk for section_chunks in grouped_chunks.values() for chunk in section_chunks]
        
        assignments = mapper.assign_chunks(all_chunks)
        return assignments
        
    except Exception as e:
        # Catch and re-raise as a more specific error.
        raise ProcessingError(f"Failed to assign chunks to skeleton: {e}")

