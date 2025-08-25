"""
State-of-the-Art Hierarchical Document Combination & Augmentation Engine.

This module operates on hierarchical document trees (ASTs), enabling sophisticated
structural merging and content integration. It moves beyond simple chunk pasting
to perform intelligent "structure grafting" and "content weaving."

Prompts are imported from the central config file for maintainability.
"""
import copy
from .llm_client import UnifiedLLMClient
from .error_handler import ProcessingError, robust_llm_call
from config import PROMPTS  # <-- IMPORT PROMPTS
import logging

# Configure logging
log = logging.getLogger(__name__)

class HierarchicalDocumentCombiner:
    """
    Performs state-of-the-art combination of two hierarchically structured documents.
    """
    def __init__(self):
        self.llm_client = UnifiedLLMClient()

    def combine_documents(self, base_doc_tree, aug_doc_tree):
        """
        Main entry point for combining two document trees.
        """
        if not isinstance(base_doc_tree, dict) or not isinstance(aug_doc_tree, dict):
            raise ProcessingError("Document inputs for combiner must be hierarchical trees (dictionaries).")

        combined_tree = copy.deepcopy(base_doc_tree)
        self._weave_overlapping_content(combined_tree, aug_doc_tree)
        self._graft_new_structures(combined_tree, aug_doc_tree)
        return combined_tree

    def _weave_overlapping_content(self, base_node, aug_node):
        """
        Recursively traverses both trees and weaves content for sections that exist in both.
        """
        for section_name, aug_section_data in aug_node.items():
            if section_name in base_node:
                base_chunks = base_node[section_name].get('chunks', [])
                aug_chunks = aug_section_data.get('chunks', [])

                if base_chunks and aug_chunks:
                    log.info(f"Weaving content for overlapping section: '{section_name}'")
                    weaved_content = self._llm_weave_chunks(base_chunks, aug_chunks, section_name)
                    
                    base_node[section_name]['chunks'] = [{
                        'type': 'weaved_paragraph',
                        'content': weaved_content,
                        'metadata': {'source': 'weaved'}
                    }]

                if 'subsections' in base_node[section_name] and 'subsections' in aug_section_data:
                    self._weave_overlapping_content(
                        base_node[section_name]['subsections'],
                        aug_section_data['subsections']
                    )

    @robust_llm_call(max_retries=2)
    def _llm_weave_chunks(self, base_chunks, aug_chunks, section_name):
        """
        Uses an LLM to intelligently weave content from two sources, using a prompt from config.
        """
        original_content = "\n\n".join([c['content'] for c in base_chunks])
        augmentation_content = "\n\n".join([c['content'] for c in aug_chunks])

        # Format the prompt from the config file with dynamic content
        prompt = PROMPTS['content_weaving'].format(
            section_name=section_name,
            original_content=original_content,
            augmentation_content=augmentation_content
        )
        return self.llm_client.call_llm(prompt, max_tokens=3000)

    def _graft_new_structures(self, base_node, aug_node):
        """

        Finds structures in the augmentation tree that are missing from the base tree
        and grafts them into the most logical position.
        """
        for section_name, aug_section_data in aug_node.items():
            if section_name not in base_node:
                log.info(f"Found new structure to graft: '{section_name}'")
                target_parent_path = self._llm_find_grafting_location(base_node, section_name, aug_section_data)
                
                target_node = base_node
                try:
                    if target_parent_path:
                        for part in target_parent_path:
                            target_node = target_node[part]['subsections']
                    
                    target_node[section_name] = aug_section_data
                    log.info(f"Successfully grafted '{section_name}' into '{' -> '.join(target_parent_path) or 'root'}'")
                except KeyError:
                    base_node[section_name] = aug_section_data
                    log.info(f"Grafting fallback: placed '{section_name}' at current level.")

            elif 'subsections' in base_node[section_name] and 'subsections' in aug_section_data:
                self._graft_new_structures(
                    base_node[section_name]['subsections'],
                    aug_section_data['subsections']
                )

    @robust_llm_call(max_retries=2)
    def _llm_find_grafting_location(self, base_tree, new_section_title, new_section_data):
        """
        Uses an LLM to determine the best location to insert a new section/subsection.
        """
        def get_tree_summary(node, indent=0):
            summary = ""
            for name in node.keys():
                summary += "  " * indent + f"- {name}\n"
                if 'subsections' in node[name] and node[name]['subsections']:
                    summary += get_tree_summary(node[name]['subsections'], indent + 1)
            return summary

        tree_summary = get_tree_summary(base_tree)
        new_section_description = new_section_data.get('description', 'No description available.')

        # Format the prompt from the config file
        prompt = PROMPTS['structure_grafting_location'].format(
            tree_summary=tree_summary,
            new_section_title=new_section_title,
            new_section_description=new_section_description
        )
        response = self.llm_client.call_llm(prompt, max_tokens=100).strip()
        
        if response == "ROOT":
            return []
        
        response = response.replace('"', '')
        return [part.strip() for part in response.split('->')]