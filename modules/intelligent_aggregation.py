"""
State-of-the-Art Document Polishing Engine.

This module acts as a final, optional post-processing step to enhance the global
coherence and consistency of the entire document. It operates on the fully
processed hierarchical document tree.

Key Features:
- Operates on the Document Tree: All functions traverse the hierarchical structure.
- Global Terminology Consistency: Extracts key terms from the whole document and
  ensures they are used consistently in every node.
- Coherence Optimization: Inserts smooth transition sentences between adjacent
  sections and subsections to improve narrative flow.
"""

from .llm_client import UnifiedLLMClient
from .error_handler import robust_llm_call
from config import PROMPTS
import copy
import logging

# Configure logging
log = logging.getLogger(__name__)

class DocumentPolisher:
    """
    Applies global, document-wide enhancements to a processed tree.
    """
    def __init__(self, llm_client: UnifiedLLMClient):
        self.llm_client = llm_client

    def polish_tree(self, processed_tree):
        """

        The main entry point to apply all polishing passes to the tree.
        """
        polished_tree = copy.deepcopy(processed_tree)
        
        log.info("Applying Polishing Pass 1: Terminology Consistency...")
        polished_tree = self._terminology_consistency_pass(polished_tree)
        
        log.info("Applying Polishing Pass 2: Coherence Optimization (Transitions)...")
        polished_tree = self._coherence_optimization_pass(polished_tree)
        
        return polished_tree

    # --- Pass 1: Terminology Consistency ---

    def _terminology_consistency_pass(self, tree):
        # 1. Extract all text content from the tree.
        full_text = self._extract_all_text_from_tree(tree)
         # 2. Add a guard clause for robustness. If no processable text was found, skip this pass.
        if not full_text.strip():
            log.warning("No processable content found in the tree. Skipping terminology consistency pass.")
            return tree
        # 3. Use an LLM to identify the key technical terms.
        key_terms_str = self._llm_extract_key_terms(full_text)
        key_terms_list = [term.strip() for term in key_terms_str.split(',')]
        log.info(f"Identified Key Terms: {key_terms_list}")
        
        # 4. Recursively traverse the tree and standardize the content in each node.
        return self._standardize_terms_recursive(tree, key_terms_list)

    def _extract_all_text_from_tree(self, node_level):
        """Recursively concatenates all 'processed_content' from the tree."""
        text = ""
        for node_data in node_level.values():
            # This check gracefully handles special, non-dictionary items like the 'Orphaned_Content' list.
            if not isinstance(node_data, dict):
                continue

            if node_data.get('processed_content'):
                text += node_data['processed_content'] + "\n\n"
            if node_data.get('subsections'):
                text += self._extract_all_text_from_tree(node_data['subsections'])
        return text

    @robust_llm_call(max_retries=1)
    def _llm_extract_key_terms(self, full_text):
        """LLM call to identify key terms from the entire document."""
        prompt = PROMPTS['term_extraction'].format(
            full_text_excerpt=full_text[:4000] # Use a large excerpt
        )
        return self.llm_client.call_llm([{"role": "user", "content": prompt}])

    # --- AFTER ---
    def _standardize_terms_recursive(self, node_level, key_terms_list):
        """Recursively traverses the tree, standardizing text in each node."""
        for node_data in node_level.values():
            # This check gracefully handles special, non-dictionary items like the 'Orphaned_Content' list.
            if not isinstance(node_data, dict):
                continue

            content = node_data.get('processed_content')
            if content:
                node_data['processed_content'] = self._llm_standardize_text(content, key_terms_list)
            
            if node_data.get('subsections'):
                self._standardize_terms_recursive(node_data['subsections'], key_terms_list)
            return node_level

    @robust_llm_call(max_retries=1)
    def _llm_standardize_text(self, text_content, key_terms_list):
        """LLM call to standardize a single piece of text."""
        prompt = PROMPTS['term_standardization'].format(
            key_terms_list=key_terms_list,
            text_content=text_content
        )
        return self.llm_client.call_llm([{"role": "user", "content": prompt}])

    # --- Pass 2: Coherence Optimization (Transitions) ---

    def _coherence_optimization_pass(self, tree):
        """Initiates the recursive process to add transition sentences."""
        # The recursive function modifies the tree in place.
        self._add_transitions_recursive(tree)
        # We must return the modified tree itself.
        return tree

    def _add_transitions_recursive(self, node_level):
        """
        Recursively traverses the tree, adding transitions between sibling nodes.
        """
        # Convert node dictionary to a list of (title, data) tuples to work with indices
        nodes_as_list = [(title, data) for title, data in node_level.items() if isinstance(data, dict)]
        
        # Iterate through adjacent pairs of nodes at the current level
        for i in range(len(nodes_as_list) - 1):
            # Get the current node and the next node
            prev_title, prev_data = nodes_as_list[i]
            current_title, current_data = nodes_as_list[i+1]
            
            prev_content = prev_data.get('processed_content', '')
            current_content = current_data.get('processed_content', '')

            if prev_content and current_content:
                transition = self._llm_generate_transition(
                    prev_title, prev_content, current_title, current_content
                )
                # Prepend the transition to the content of the second node
                current_data['processed_content'] = transition + "\n\n" + current_content
                log.info(f"Added transition between '{prev_title}' and '{current_title}'")
        
        # After processing siblings, recurse into the children of each node
        for _, node_data in nodes_as_list:
        # This check gracefully handles special, non-dictionary items.
            if not isinstance(node_data, dict):
                continue

            if node_data.get('subsections'):
                self._add_transitions_recursive(node_data['subsections'])
    
    @robust_llm_call(max_retries=1)
    def _llm_generate_transition(self, prev_title, prev_content, current_title, current_content):
        """LLM call to generate a single transition sentence."""
        prompt = PROMPTS['section_transition'].format(
            prev_section_title=prev_title,
            prev_section_ending=prev_content[-300:], # Last 300 chars
            current_section_title=current_title,
            current_section_beginning=current_content[:300] # First 300 chars
        )
        return self.llm_client.call_llm([{"role": "user", "content": prompt}])