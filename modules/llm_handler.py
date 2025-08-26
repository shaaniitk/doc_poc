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
import json
from .semantic_validator import SemanticValidator
from .section_mapper import SemanticMapper # <-- ADD THIS IMPORT

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

    def process_tree(self, document_tree, generative_context=None):
        """
        Main entry point to start the processing of a document tree.
        Can operate in 'refactoring' mode or 'generative' mode.
        """
        self.full_tree = document_tree
        if not self.global_context:
            self.global_context = document_tree.get("Abstract", {}).get('description', 
                                    "A peer-to-peer electronic cash system.")
        
        return self._recursive_process_node(document_tree, parent_context="", path=[], generative_context=generative_context)
 

    def _recursive_process_node(self, current_level_nodes, parent_context, path, generative_context=None):
        """
        The core recursive method. If `generative_context` is provided, it uses
        that as the input; otherwise, it uses the node's own chunks.
        """
        processed_level = {}
        for title, node_data in current_level_nodes.items():
            if not isinstance(node_data, dict): continue

            current_path = path + [title]
            
            # --- CORE LOGIC CHANGE ---
            # Determine the source content for this node.
            if generative_context:
                # In generative mode, the source is the full document text.
                node_content = generative_context
                log.info(f"  Generatively processing node: {' -> '.join(current_path)}")
            else:
                # In normal mode, the source is the node's own chunks.
                log.info(f"  Refactoring node: {' -> '.join(current_path)}")
                node_content = "\n\n".join([chunk['content'] for chunk in node_data.get('chunks', [])])

            if node_content:
                context = {
                    "node_path": " -> ".join(current_path), "global_context": self.global_context,
                    "parent_context": parent_context, "node_content": node_content
                }
                refactored_content = self._strategy_refactor_content(context, node_data.get('prompt', ''))
                
                # ... (Self-critique pass remains the same) ...

                node_data['processed_content'] = refactored_content
            else:
                # If there's no content to process, ensure the key exists but is empty.
                node_data['processed_content'] = ""

            if node_data.get('subsections'):
                processed_subsections = self._recursive_process_node(
                    node_data['subsections'], parent_context=node_data.get('processed_content', ''),
                    path=current_path, generative_context=generative_context
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
    def _dynamically_generate_subsections(self, parent_node_data, parent_title):
        """
        A "first pass" that uses an LLM to discover and create a subsection
        structure for a node marked as dynamic.
        """
        print(f"    -> Running dynamic subsection discovery for '{parent_title}'...")
        # 1. Combine all chunks for the parent section into one text block.
        all_content = "\n\n".join([chunk['content'] for chunk in parent_node_data.get('chunks', [])])
        if not all_content:
            return # Cannot generate subsections from no content

        # 2. Call the LLM with the identifier prompt to get a list of titles.
        try:
            prompt = PROMPTS['dynamic_subsection_identifier'].format(
                parent_section_title=parent_title,
                text_content=all_content
            )
            response = self.llm_client.call_llm([{"role": "user", "content": prompt}])
            subsection_titles = json.loads(response)
            if not isinstance(subsection_titles, list):
                raise ValueError("LLM did not return a valid JSON list.")
        except Exception as e:
            print(f"    -> WARNING: Failed to dynamically generate subsections for '{parent_title}'. Error: {e}")
            return # Abort the dynamic process on failure

        # 3. Dynamically create the new subsection nodes in the tree.
        print(f"    -> Discovered {len(subsection_titles)} subsections to create.")
        parent_node_data['subsections'] = {} # Clear any predefined subsections
        for title in subsection_titles:
            parent_node_data['subsections'][title] = {
                'prompt': parent_node_data['prompt'], # Inherit prompt from parent
                'description': f"Content related to {title}", # Create a dynamic description
                'chunks': [],
                'subsections': {}
            }

        # 4. Use the SemanticMapper to re-distribute the original chunks into the new subsections.
        if parent_node_data.get('chunks'):
            # This is a clever re-use of our existing powerful module!
            mapper = SemanticMapper(template_name=None) # We don't need a full template
            # Manually set the skeleton to our newly created dynamic one
            mapper.skeleton = list(parent_node_data['subsections'].values())
            mapper.section_paths, mapper.section_embeddings = mapper._prepare_skeleton_embeddings()
            
            # Re-assign the parent's chunks into the new child subsections
            assignments = mapper.assign_chunks(parent_node_data['chunks'])
            for section_title, assigned_chunks in assignments.items():
                if section_title in parent_node_data['subsections']:
                    parent_node_data['subsections'][section_title]['chunks'].extend(assigned_chunks)
        
        # 5. Clear the parent's chunks, as they have all been moved down.
        parent_node_data['chunks'] = []

    def process_single_node(self, node_title, node_data, full_processed_content):
        """
        Processes a single node, typically a generative one, using the full
        processed content of the document as context.
        """
        print(f"  -> Running on-demand processing for generative node: '{node_title}'")
        
        # The "node_content" for a generative prompt is the full document text.
        context = {
            "node_path": node_title,
            "global_context": self.global_context,
            "parent_context": "N/A", # No parent in this context
            "node_content": full_processed_content 
        }

        # Use the node's specific prompt for generation
        generated_content = self._strategy_refactor_content(context, node_data.get('prompt', ''))
        node_data['processed_content'] = generated_content
        
        return node_data