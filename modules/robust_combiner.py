"""
Robust Hierarchical Document Combiner with Fallback Strategy.

This module provides a simple, deterministic, and non-LLM-dependent method for
merging two hierarchical document trees. Its primary purpose is to serve as a
fail-safe if the more advanced, LLM-driven HierarchicalDocumentCombiner fails.

Key Features:
- Operates on Hierarchical Trees: Directly manipulates the nested dictionary structure.
- Deterministic Logic: Uses simple rules to append new sections and subsections.
- No LLM Dependency: Guarantees execution without risk of API failures.
"""
import copy
import logging

# Configure logging
log = logging.getLogger(__name__)

class RobustHierarchicalCombiner:
    """
    Combines two document trees using a simple, robust append strategy.
    """
    def combine_documents(self, base_tree, aug_tree):
        """
        Merges the augmentation tree into the base tree.

        If a section from the augmentation tree does not exist in the base tree,
        it is appended. If it does exist, its subsections are recursively merged.
        Content (`chunks`) of existing sections is not modified.

        Args:
            base_tree (dict): The primary document tree.
            aug_tree (dict): The augmentation document tree to merge from.

        Returns:
            dict: The combined document tree.
        """
        # Create a deep copy to ensure the original base_tree is not mutated.
        combined_tree = copy.deepcopy(base_tree)

        # Start the recursive append process at the top level of the trees.
        self._recursive_append(combined_tree, aug_tree)

        return combined_tree

    def _recursive_append(self, base_node, aug_node):
        """
        The core recursive logic for appending new structures.

        Args:
            base_node (dict): The current node (or level) in the base tree.
            aug_node (dict): The corresponding node in the augmentation tree.
        """
        for key, aug_data in aug_node.items():
            if key not in base_node:
                # If the section/subsection is new, simply add the entire node.
                log.info(f"Robustly appending new section: '{key}'")
                base_node[key] = aug_data
            else:
                # If the section already exists, do not overwrite its content.
                # Instead, recursively check its subsections for new content to append.
                if 'subsections' in aug_data and aug_data['subsections']:
                    # Ensure the base node has a subsections dict to append to.
                    if 'subsections' not in base_node[key]:
                        base_node[key]['subsections'] = {}
                    
                    self._recursive_append(base_node[key]['subsections'], aug_data['subsections'])