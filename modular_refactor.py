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

# Configure logging
log = logging.getLogger(__name__)

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
         template="bitcoin_paper_hierarchical", polishing=True, run_analysis=False):
    """
    Main function to orchestrate the entire document processing pipeline.

    Args:
        source (str): Path to the primary source LaTeX file.
        source2 (str, optional): Path to the secondary (augmentation) file.
        combine_strategy (str): 'smart' (LLM-based) or 'robust' (append-based).
        output_format (str): The desired output format (e.g., 'latex', 'markdown').
        template (str): The name of the hierarchical template from config to use.
        polishing (bool): Whether to run the final document-wide polishing pass.
    """
    log.info("--- Starting Document Processing Engine ---")

    # --- 1. Initialization ---
    log.info("Initializing core modules...")
    output_manager = OutputManager()
    log_entries = [f"Session started: {output_manager.session_id}"]
    
    try:
        # Initialize shared components
        llm_client = UnifiedLLMClient()
        agent = HierarchicalProcessingAgent(llm_client, output_format)
        
        # --- 2. Document Parsing and Mapping ---
        # This step is common to both workflows: parse source file(s) into a
        # standardized hierarchical tree structure.
        
        def parse_and_map_document(file_path, template_name):
            log.info(f"Parsing and mapping source file: {file_path}")
            content = load_latex_file(file_path)
            # The new AST chunker understands the document's true structure
            chunks = extract_latex_sections(content, source_path=file_path)
            grouped_chunks = group_chunks_by_section(chunks)
            # The new semantic mapper assigns chunks to the hierarchical skeleton
            mapped_tree = assign_chunks_to_skeleton(grouped_chunks, template_name)
            return mapped_tree

        # Check if the chosen template is hierarchical
        if not isinstance(list(DOCUMENT_TEMPLATES[template].values())[0], dict):
             raise ValueError(f"Template '{template}' is not hierarchical. Please use a hierarchical template.")

        base_tree = parse_and_map_document(source, template)
        tree_to_process = base_tree

        # --- 3. Workflow Branching: Augmentation or Single Refactoring ---
        if source2:
            # --- Two-Document Augmentation Workflow ---
            log.info(f"\n--- Running in Two-Document Augmentation Mode (Strategy: {combine_strategy}) ---")
            log_entries.append("Entering two-document augmentation workflow.")
            
            aug_tree = parse_and_map_document(source2, template)

            if combine_strategy == "smart":
                combiner = HierarchicalDocumentCombiner()
            else: # 'robust'
                combiner = RobustHierarchicalCombiner()
            
            tree_to_process = combiner.combine_documents(base_tree, aug_tree)
            log_entries.append(f"Successfully combined two document trees using '{combine_strategy}' strategy.")

        else:
            # --- Single-Document Refactoring Workflow ---
            log.info("\n--- Running in Single-Document Refactoring Mode ---")
            log_entries.append("Entering single-document refactoring workflow.")
            # The 'tree_to_process' is already the mapped 'base_tree'
        
        # --- 4. Core LLM Processing ---
        log.info("\n--- Starting Core Hierarchical Processing ---")
        processed_tree = agent.process_tree(tree_to_process)
        log_entries.append("Core hierarchical processing completed.")

        # --- 5. (Optional) Global Polishing Pass ---
        final_tree = processed_tree
        if polishing:
            log.info("\n--- Applying Final Polishing Pass ---")
            polisher = DocumentPolisher(llm_client)
            final_tree = polisher.polish_tree(processed_tree)
            log_entries.append("Document-wide polishing pass completed.")
        else:
            log.info("\nSkipping final polishing pass.")
            log_entries.append("Skipped optional polishing pass.")
        
        
        # --- 6. Saving All Processed Nodes Individually (RESTORED) ---
        log.info("\n--- Saving All Processed Nodes Individually ---")
        output_manager.save_processed_tree_nodes(final_tree)
        log_entries.append(f"All processed nodes saved to individual files in: {output_manager.nodes_path}")

        # --- 7. Formatting and Saving Output ---
        log.info("\n--- Formatting and Saving Final Document ---")
        formatter = HierarchicalOutputFormatter(output_format)
        final_document_string = formatter.format_document(final_tree)
        
        # The OutputManager now just saves the final string
        final_path = output_manager.save_section_output("final_document", final_document_string)
        log_entries.append(f"Final document formatted and saved to: {final_path}")

        log.info("\n--- Document Processing Successful! ---")
        log.info(f"  Final Document: {final_path}")
        log.info(f"  Session Folder: {output_manager.session_path}")

    except Exception as e:
        error_msg = f"FATAL ERROR: An unexpected error occurred: {str(e)}"
        log.error(f"\n{error_msg}")
        log_entries.append(error_msg)
        raise # Re-raise the exception after logging for better debugging

    finally:
        # Always save the log file, even if an error occurred
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
    
    args = parser.parse_args()
    
    session_path = main(source=args.source, source2=args.source2, combine_strategy=args.combine_strategy,
                    output_format=args.output_format, template=args.template, 
                    polishing=args.polishing, run_analysis=args.analysis) #<-- Pass args.analysis here

# Conditionally run the analysis script
    if args.analysis and session_path:
        print("\n--- Handing off to Post-Processing Analysis Suite ---")
        # Ensure your run_analysis.py is in the modules folder
        from modules.run_analysis import main as run_analysis_main
        run_analysis_main(
            session_path=session_path,
            original_source=args.source,
            aug_source=args.source2,
            template=args.template
        )