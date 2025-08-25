"""
State-of-the-Art Hierarchical LLM Processing Agent.

This module contains the core engine of the document refactoring system.
The HierarchicalProcessingAgent traverses a document's tree structure, building
deeply contextual prompts at each node and applying advanced, multi-pass
processing strategies to ensure the highest quality output.

It leverages the FormatEnforcer utility to sanitize all LLM outputs, ensuring
syntactic correctness before they are stored in the processed tree.
"""

from .llm_client import UnifiedLLMClient
from .error_handler import robust_llm_call
from .format_enforcer import FormatEnforcer  # <-- IMPORT THE ENFORCER
from .semantic_validator import SemanticValidator  # <-- IMPORT SEMANTIC VALIDATOR
from config import PROMPTS
import re
import logging

# Configure logging
log = logging.getLogger(__name__)

class HierarchicalProcessingAgent:
    """
    Traverses a document tree to refactor content node by node with hierarchical context.
    """
    def __init__(self, llm_client: UnifiedLLMClient, output_format="latex"):
        self.llm_client = llm_client
        self.full_tree = None
        self.global_context = ""
        # 1. INSTANTIATE THE FORMAT ENFORCER
        self.format_enforcer = FormatEnforcer(output_format)
        self.semantic_validator = SemanticValidator()

    def process_tree(self, document_tree):
        """
        Main entry point to start the processing of the entire document tree.
        """
        self.full_tree = document_tree
        self.global_context = document_tree.get("Abstract", {}).get('description', 
                                "A peer-to-peer electronic cash system.")
        
        return self._recursive_process_node(document_tree, parent_context="", path=[])

    def _recursive_process_node(self, current_level_nodes, parent_context, path):
        """
        The core recursive method that traverses the tree and processes each node.
        """
        processed_level = {}

        for title, node_data in current_level_nodes.items():
            # This check gracefully handles special, non-dictionary items
            # like the 'Orphaned_Content' list.
            if not isinstance(node_data, dict):
                continue # Skip to the next item in the loop
            
            current_path = path + [title]
            log.info(f"Processing Node: {' -> '.join(current_path)}")

            node_content = "\n\n".join([chunk['content'] for chunk in node_data.get('chunks', [])])

            if node_content:
                context = {
                    "node_path": " -> ".join(current_path),
                    "global_context": self.global_context,
                    "parent_context": parent_context,
                    "node_content": node_content
                }
                
                refactored_content = self._strategy_refactor_content(context, node_data.get('prompt', ''))
                is_valid, score = self.semantic_validator.validate_content_preservation(
                    original=node_content,
                    processed=refactored_content,
                    threshold=0.6 # Use a reasonable threshold
                )
                log.info(f"Semantic Similarity Score: {score:.2f}")
                if not is_valid:
                    log.warning(f"Low semantic similarity for '{' -> '.join(current_path)}'. May have deviated from original meaning.")
                
                if any(keyword in title for keyword in ["Introduction", "Conclusion", "Abstract"]):
                    log.info(f"Applying advanced self-critique pass for '{title}'...")
                    refactored_content = self._llm_self_critique_pass(context, refactored_content)

                node_data['processed_content'] = refactored_content
            else:
                node_data['processed_content'] = ""

            if node_data.get('subsections'):
                processed_subsections = self._recursive_process_node(
                    node_data['subsections'],
                    parent_context=node_data.get('processed_content', ''), # Pass clean content down
                    path=current_path
                )
                node_data['subsections'] = processed_subsections
            
            processed_level[title] = node_data
        
        return processed_level

    @robust_llm_call(max_retries=2)
    def _strategy_refactor_content(self, context, node_prompt):
        """
        Performs the primary content refactoring and immediately enforces format compliance.
        """
        system_prompt = node_prompt or "You are a professional technical editor."
        user_prompt = PROMPTS['hierarchical_refactor'].format(**context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 2. INTEGRATION POINT
        # Get the raw output from the LLM
        raw_output = self.llm_client.call_llm(messages)
        
        # Immediately clean and enforce the format
        clean_output, issues = self.format_enforcer.enforce_format(raw_output)
        
        if issues:
            log.warning(f"FormatEnforcer found issues (pass 1): {issues}")
        
        return clean_output

    @robust_llm_call(max_retries=2)
    def _llm_self_critique_pass(self, original_context, refactored_text):
        """
        Performs a self-critique pass and immediately enforces format compliance on the result.
        """
        prompt = PROMPTS['self_critique_and_refine'].format(
            node_path=original_context['node_path'],
            refactored_text=refactored_text
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        raw_response = self.llm_client.call_llm(messages)
        
        match = re.search(r"Final Polished Version:\s*(.*)", raw_response, re.DOTALL | re.IGNORECASE)
        
        if match:
            # 3. INTEGRATION POINT
            raw_final_version = match.group(1).strip()
            
            # Immediately clean and enforce the format on the final version
            clean_final_version, issues = self.format_enforcer.enforce_format(raw_final_version)
            
            if issues:
                log.warning(f"FormatEnforcer found issues (pass 2): {issues}")
            
            return clean_final_version
        else:
            # If parsing fails, fall back to the already cleaned first version.
            return refactored_text