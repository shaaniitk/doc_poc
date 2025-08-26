"""
Main Orchestrator for State-of-the-Art Document Refactoring & Augmentation.

This script serves as the central controller for the entire document processing
pipeline. It leverages a suite of advanced, hierarchy-aware modules to perform
either a deep refactoring of a single document or a structural augmentation
by combining two documents.

The workflow is as follows:
1.  Parse document(s) into a hierarchical Abstract Syntax Tree (AST).
2.  Map the document structure to a standardized, hierarchical template.
3.  IF two documents are provided, combine them using an advanced hierarchical
    merging strategy (LLM-driven or robust fallback).
4.  Process the resulting document tree node-by-node using the
    HierarchicalProcessingAgent, which applies contextual, multi-pass LLM calls.
5.  (Optional) Perform a global polishing pass on the entire tree to enhance
    coherence and consistency.
6.  Format the final, processed tree into the desired output format (e.g., LaTeX)
    and save the results.
"""
import sys
import os
import logging


# --- Configure Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__) # <-- GET THE LOGGER INSTANCE

# --- Import all state-of-the-art modules ---
from modules.file_loader import load_latex_file
from modules.chunker import extract_latex_sections, group_chunks_by_section
from modules.section_mapper import assign_chunks_to_skeleton
from modules.llm_client import UnifiedLLMClient
from modules.llm_handler import HierarchicalProcessingAgent
from modules.document_combiner import HierarchicalDocumentCombiner
from modules.robust_combiner import RobustHierarchicalCombiner
from modules.intelligent_aggregation import DocumentPolisher
from modules.output_formatter import HierarchicalOutputFormatter
from modules.output_manager import OutputManager
from config import DOCUMENT_TEMPLATES # Import for checking template existence

def main(source, source2=None, combine_strategy="smart", output_format="latex", 
         template="bitcoin_paper_hierarchical", polishing=True, run_analysis=False, stage="all"):
    
    log.info("--- Starting Document Processing Engine ---")
    log.info("Initializing core modules for session...")
    output_manager = OutputManager()
    log_entries = [f"Session started: {output_manager.session_id}"]
    
    try:
        # --- Stage 1 & 2: Parsing and Mapping ---
        # This part is common to both workflows.
        def parse_and_map_document(file_path, template_name):
            log.info(f"Parsing and mapping source file: {file_path}")
            content = load_latex_file(file_path)
            chunks = extract_latex_sections(content, source_path=file_path)
            grouped_chunks = group_chunks_by_section(chunks)
            mapped_tree = assign_chunks_to_skeleton(grouped_chunks, template_name)
            return mapped_tree

        if template not in DOCUMENT_TEMPLATES or not isinstance(DOCUMENT_TEMPLATES[template], dict):
            # ... (template validation logic remains the same) ...
            return None

        base_tree = parse_and_map_document(source, template)
        
        # This variable will hold the unified tree, regardless of the workflow.
        tree_to_process = None

        # --- Stage 3: (Conditional) Document Combination ---
        if source2:
            # --- Two-Document Augmentation Workflow ---
            log.info(f"--- Running in Two-Document Augmentation Mode (Strategy: {combine_strategy}) ---")
            log_entries.append("Entering two-document augmentation workflow.")
            
            aug_tree = parse_and_map_document(source2, template)

            if combine_strategy == "smart":
                combiner = HierarchicalDocumentCombiner()
            else: # 'robust'
                combiner = RobustHierarchicalCombiner()
            
            # The result of the combination is our tree to process.
            tree_to_process = combiner.combine_documents(base_tree, aug_tree)
            log_entries.append(f"Successfully combined two document trees using '{combine_strategy}' strategy.")
        else:
            # --- Single-Document Refactoring Workflow ---
            log.info("--- Running in Single-Document Refactoring Mode ---")
            log_entries.append("Entering single-document refactoring workflow.")
            # In this case, the mapped base tree is our tree to process.
            tree_to_process = base_tree
            
        # --- Stage 4: Pre-Processing for Generative Content ---
        # This logic is now IDENTICAL for both workflows.
        # --- AFTER ---
        # --- STAGE 4: Pre-Processing for Generative Content ---
        log.info("Separating main content from generative content...")
        
        # First, safely pop the 'Orphaned_Content' list to handle it separately.
        # This ensures the following loops only deal with structured dictionary nodes.
        orphaned_chunks = tree_to_process.pop('Orphaned_Content', [])
        
        generative_nodes = {}
        main_content_tree = {}
        # This loop is now safe, as it will only iterate over actual document nodes.
        for title, node in tree_to_process.items():
            if node.get('generative'):
                generative_nodes[title] = node
            else:
                main_content_tree[title] = node
        log_entries.append(f"Identified {len(generative_nodes)} generative node(s).")
        
        # --- Stage 5, Phase 1: Process Main Content ---
        log.info("--- STAGE 5.1: Processing Main Document Content ---")
        llm_client = UnifiedLLMClient()
        agent = HierarchicalProcessingAgent(llm_client, output_format)
        processed_main_tree = agent.process_tree(main_content_tree)
        log_entries.append("Main content processing completed.")

        # --- Stage 5, Phase 2: Create Generative Content ---
        log.info("--- STAGE 5.2: Creating Generative Content (e.g., Summary) ---")
        temp_formatter = HierarchicalOutputFormatter(output_format)
        full_processed_content = temp_formatter.format_document(processed_main_tree)

        processed_generative_nodes = {}
        for title, node in generative_nodes.items():
            processed_node = agent.process_single_node(title, node, full_processed_content)
            processed_generative_nodes[title] = processed_node
        log_entries.append("Generative content creation completed.")
        
          # --- Stage 5, Phase 3: Recombine Tree ---
        final_tree = {**processed_generative_nodes, **processed_main_tree}
        # Add the orphaned chunks back into the final tree for the formatter.
        if orphaned_chunks:
            final_tree['Orphaned_Content'] = orphaned_chunks
        log.info("Recombined all content into final tree.")

        # --- Stage 6: (Optional) Polishing ---
        if polishing:
            log.info("--- Applying Final Polishing Pass ---")
            polisher = DocumentPolisher(llm_client)
            final_tree = polisher.polish_tree(final_tree)
            log_entries.append("Document-wide polishing pass completed.")
        
        # --- Stage 7: Saving and Analysis ---
        log.info("--- Saving All Processed Nodes Individually ---")
        output_manager.save_processed_tree_nodes(final_tree)
        log_entries.append(f"All processed nodes saved to: {output_manager.nodes_path}")

        log.info("--- Formatting and Saving Final Document ---")
        formatter = HierarchicalOutputFormatter(output_format)
        final_document_string = formatter.format_document(final_tree)
        final_path = output_manager.aggregate_document(final_document_string, output_format)
        log_entries.append(f"Final document saved to: {final_path}")

        log.info("--- Document Processing Successful! ---")
        log.info(f"  Final Document: {final_path}")
        
        return output_manager.session_path

    except Exception as e:
        log.error(f"FATAL ERROR: An unexpected error occurred: {str(e)}", exc_info=True)
        log_entries.append(f"FATAL ERROR: {str(e)}")
        raise

    finally:
        log_path = output_manager.save_processing_log(log_entries)
        log.info(f"  Processing Log: {log_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="State-of-the-Art Document Refactoring and Augmentation Engine.")
    parser.add_argument("--source", required=True, help="Path to the primary source .tex file.")
    parser.add_argument("--source2", help="Path to the second .tex file for augmentation.")
    parser.add_argument("--combine", dest="combine_strategy", default="smart", choices=["smart", "robust"], help="Strategy for combining two documents.")
    parser.add_argument("--format", dest="output_format", default="latex", choices=["latex", "markdown"], help="The output format for the final document.")
    parser.add_argument("--template", default="bitcoin_paper_hierarchical", help="The hierarchical document template to use from config.py.")
    parser.add_argument("--no-polishing", dest="polishing", action="store_false", help="Disable the final, global polishing pass.")
    parser.add_argument("--analysis", action="store_true", help="Run a full post-processing analysis after the main run completes.") # <-- ADD THIS LINE
    parser.add_argument("--stage", default="all", choices=["chunk", "map", "all"],
                        help="Run only a specific stage of the pipeline for debugging (chunk, map).") # <-- ADD THIS
   
    args = parser.parse_args()
    
    session_path = main(source=args.source, source2=args.source2, combine_strategy=args.combine_strategy,
                    output_format=args.output_format, template=args.template, 
                    polishing=args.polishing, run_analysis=args.analysis) #<-- Pass args.analysis here

# Conditionally run the analysis script
    if args.analysis and session_path:
        log.info("--- Handing off to Post-Processing Analysis Suite ---")
        try:
            # Ensure your run_analysis.py is in the modules folder
            from modules.run_analysis import main as run_analysis_main
            run_analysis_main(
                session_path=session_path,
                original_source=args.source,
                aug_source=args.source2,
                template=args.template
            )
        except Exception as e:
            log.error(f"Post-processing analysis failed: {e}", exc_info=True)