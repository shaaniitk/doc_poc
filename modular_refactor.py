"""


The workflow is as follows:
1.  Parse document(s) into a hierarchical list of chunks with metadata.
2.  Run the multi-pass IntelligentMapper to assign all chunks to a standardized,
    hierarchical document tree, optionally using an LLM to remediate orphans.
3.  IF two documents are provided, combine their trees using a hierarchical merging strategy.
4.  Separate the unified tree into "main content" and "generative content" (e.g., Summary).
5.  Process the main content tree using the HierarchicalProcessingAgent.
6.  Process the generative content tree, using the processed main content as its context.
7.  Recombine the trees and run an optional global polishing pass.
8.  Format the final, processed tree into the desired output format and save all results.
"""
import sys
import os
import argparse
import logging
import re

# --- Import all state-of-the-art modules ---
from modules.file_loader import load_file_content
from modules.chunker import extract_document_sections
from modules.intelligent_mapper import IntelligentMapper
from modules.llm_client import UnifiedLLMClient
from modules.llm_handler import HierarchicalProcessingAgent
from modules.document_combiner import HierarchicalDocumentCombiner
from modules.robust_combiner import RobustHierarchicalCombiner
from modules.intelligent_aggregation import DocumentPolisher
from modules.output_formatter import HierarchicalOutputFormatter
from modules.output_manager import OutputManager
from config import DOCUMENT_TEMPLATES
from modules.output_manager import final_latex_sanitization
from modules.template_enhancer import TemplateEnhancer

# --- Configure Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def main(source, source2=None, combine_strategy="smart", output_format="latex", 
         template="bitcoin_paper_hierarchical", polishing=True, run_analysis=False, 
         stage="all", remediate_orphans=False):
    """
    Main function to orchestrate the entire document processing pipeline.
    """
    log.info("--- Starting Document Processing Engine ---")
    log.info("Initializing core modules for session...")
    output_manager = OutputManager()
    log_entries = [f"Session started: {output_manager.session_id}"]
    
    try:
        # --- Template Validation ---
        if template not in DOCUMENT_TEMPLATES or not isinstance(DOCUMENT_TEMPLATES[template], dict):
            log.error(f"FATAL: The template '{template}' is not a valid hierarchical template (must be a dictionary).")
            log.info("Please use a valid hierarchical template, for example: --template bitcoin_paper_hierarchical")
            output_manager.save_processing_log(log_entries)
            return None

        # --- Stage 1: Parsing & Chunking ---
        log.info("--- STAGE 1: PARSING & CHUNKING ---")
        log_entries.append("Entering Stage 1: Parsing & Chunking.")
        content = load_file_content(source)
        all_chunks_from_parser, preserved_data = extract_document_sections(content, source_path=source)
        chunk_output_path = output_manager.save_json_output("1_chunk_output.json", all_chunks_from_parser)
        log.info(f"-> Found {len(all_chunks_from_parser)} initial chunks.")
        log.info(f"-> Chunking results saved to: {chunk_output_path}")
        log_entries.append(f"Stage 1 completed. Found {len(all_chunks_from_parser)} chunks.")

            # ---  DYNAMIC TEMPLATE ENHANCEMENT ---
        log.info("-- DYNAMIC TEMPLATE ENHANCEMENT ---")
        llm_client = UnifiedLLMClient() # We'll need the client early
        original_template = DOCUMENT_TEMPLATES[template]
        enhancer = TemplateEnhancer(llm_client)
        # Create the full text from chunks for the enhancer
        full_text_content = "\n\n".join([chunk['content'] for chunk in all_chunks_from_parser])
        enhanced_template = enhancer.enhance_template(original_template, full_text_content)

        if stage == "chunk":
            log.info("--- Pipeline halted after 'chunk' stage as requested. ---")
            return output_manager.session_path

        # --- Stage 2: Intelligent Mapping ---
        log.info("--- STAGE 2: INTELLIGENT MAPPING ---")
        log_entries.append("Entering Stage 2: Intelligent Mapping.")
        mapper = IntelligentMapper(template_name=template, template_object=enhanced_template) # No flag in the constructor
        mapped_tree = mapper.map_chunks(all_chunks_from_parser, use_llm_pass=remediate_orphans) # Flag goes here
        map_output_path = output_manager.save_json_output("2_mapped_tree.json", mapped_tree)
        log.info(f"-> Mapped tree structure saved to: {map_output_path}")
        log_entries.append("Stage 2 completed.")

        if stage == "map":
            log.info("--- Pipeline halted after 'map' stage as requested. ---")
            return output_manager.session_path
            
        tree_to_process = mapped_tree

        # --- Stage 3: (Conditional) Document Combination ---
        if source2:
            log.info(f"--- STAGE 3: Combining Documents (Strategy: {combine_strategy}) ---")
            aug_content = load_file_content(source2)
            aug_chunks = extract_document_sections(aug_content, source_path=source2)
            aug_mapped_tree = mapper.map_chunks(aug_chunks)
            
            if combine_strategy == "smart":
                combiner = HierarchicalDocumentCombiner()
            else:
                combiner = RobustHierarchicalCombiner()
            tree_to_process = combiner.combine_documents(tree_to_process, aug_mapped_tree)
            log_entries.append("Document combination completed.")

        # --- Stage 4: Generative Content Separation ---
        log.info("--- STAGE 4: Separating Main and Generative Content ---")
        orphaned_chunks = tree_to_process.pop('Orphaned_Content', [])
        generative_nodes, main_content_tree = {}, {}
        for title, node in tree_to_process.items():
            if node.get('generative'):
                generative_nodes[title] = node
            else:
                main_content_tree[title] = node
        log_entries.append(f"Identified {len(generative_nodes)} generative node(s).")
        
        # --- Stage 5: Core LLM Processing ---
        log.info("--- STAGE 5: Core Document Processing ---")
        llm_client = UnifiedLLMClient()
        agent = HierarchicalProcessingAgent(llm_client, output_format)
        
        # Phase 1: Process main content
        processed_main_tree = agent.process_tree(main_content_tree)
        
        # Phase 2: Process generative content
        processed_generative_nodes = {}
        if generative_nodes:
            log.info("-> Creating generative content (Summary, etc.)...")
            temp_formatter = HierarchicalOutputFormatter("latex")
            full_processed_content = temp_formatter.format_document(processed_main_tree)
            processed_generative_nodes = agent.process_tree(generative_nodes, generative_context=full_processed_content)
        
        # Recombine all parts
        final_tree = {**processed_generative_nodes, **processed_main_tree}
        if orphaned_chunks:
            final_tree['Orphaned_Content'] = orphaned_chunks

        # --- Stage 6: (Optional) Polishing ---
        if polishing:
            log.info("--- STAGE 6: Applying Final Polishing Pass ---")
            polisher = DocumentPolisher(llm_client)
            final_tree = polisher.polish_tree(final_tree)
            log_entries.append("Polishing pass completed.")

        # --- Stage 7: Saving Outputs ---
        log.info("--- STAGE 7: Saving All Outputs ---")
        output_manager.save_processed_tree_nodes(final_tree)
        formatter = HierarchicalOutputFormatter(output_format)
        final_document_string = formatter.format_document(final_tree, preserved_data=preserved_data)

        if output_format == "latex":
            log.info("-> Running final sanitization pass on LaTeX output...")
            final_document_string = final_latex_sanitization(final_document_string)

        final_path = output_manager.aggregate_document(final_document_string, output_format)
        log.info("--- Document Processing Successful! ---")
        log.info(f"-> Final Document: {final_path}")
        
        return output_manager.session_path

    except Exception as e:
        log.error(f"FATAL ERROR: An unexpected error occurred: {str(e)}", exc_info=True)
        log_entries.append(f"FATAL ERROR: {str(e)}")
        return None

    finally:
        log_path = output_manager.save_processing_log(log_entries)
        log.info(f"-> Processing Log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State-of-the-Art Document Refactoring and Augmentation Engine.")
    parser.add_argument("--source", required=True, help="Path to the primary source .tex file.")
    parser.add_argument("--source2", help="Path to the second .tex file for augmentation.")
    parser.add_argument("--combine", dest="combine_strategy", default="smart", choices=["smart", "robust"], help="Strategy for combining two documents.")
    parser.add_argument("--format", dest="output_format", default="latex", choices=["latex", "markdown"], help="The output format for the final document.")
    parser.add_argument("--template", default="bitcoin_paper_hierarchical", help="The hierarchical document template to use from config.py.")
    parser.add_argument("--no-polishing", dest="polishing", action="store_false", help="Disable the final, global polishing pass.")
    parser.add_argument("--stage", default="all", choices=["chunk", "map", "all"], help="Run only a specific stage of the pipeline for debugging.")
    parser.add_argument("--remediate-orphans", action="store_true", help="Enable the LLM to dynamically create new sections for orphaned content.")
    parser.add_argument("--analysis", action="store_true", help="Run a full post-processing analysis after the main run completes.")
    
    args = parser.parse_args()
    
    session_path = main(source=args.source, source2=args.source2, combine_strategy=args.combine_strategy,
                        output_format=args.output_format, template=args.template, 
                        polishing=args.polishing, run_analysis=args.analysis, 
                        stage=args.stage, remediate_orphans=args.remediate_orphans)
    
    if args.analysis and session_path:
        log.info("--- Handing off to Post-Processing Analysis Suite ---")
        try:
            from modules.run_analysis import main as run_analysis_main
            run_analysis_main(
                session_path=session_path,
                original_source=args.source,
                aug_source=args.source2,
                template=args.template
            )
        except Exception as e:
            log.error(f"Post-processing analysis failed: {e}", exc_info=True)