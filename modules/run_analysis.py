"""
Main Entry Point for Post-Processing Analysis.

This script loads the results of a processing session and uses the unified
analysis engine to generate comprehensive reports on quality and contribution.
"""
import os
import json
import logging
from .analysis_engine import DocumentAnalyzer

# Configure logging
log = logging.getLogger(__name__)
# The following imports are for re-parsing the documents
from .file_loader import load_file_content
from .chunker import extract_document_sections, group_chunks_by_section
from .section_mapper import assign_chunks_to_skeleton

def main(session_path, original_source, aug_source=None, template="bitcoin_paper_hierarchical"):
    """
    Runs the full analysis suite on a completed session.
    """
    log.info("--- Running Post-Processing Analysis Suite ---")

    try:
        # --- Load Processed Data ---
        final_doc_path = os.path.join(session_path, "final_document.tex")
        if not os.path.exists(final_doc_path):
            log.error(f"ERROR: Final document not found at {final_doc_path}")
            return
            
        # To analyze, we need to parse the original and final docs into trees
        def parse_for_analysis(file_path):
            content = load_file_content(file_path)
            chunks = extract_document_sections(content, source_path=file_path)
            grouped = group_chunks_by_section(chunks)
            # For analysis, we use the grouped structure directly as a simple "tree"
            return grouped

        log.info("Parsing original document for analysis...")
        original_tree = parse_for_analysis(original_source)
        
        log.info("Parsing final document for analysis...")
        processed_tree = parse_for_analysis(final_doc_path)
        
        aug_tree = None
        if aug_source:
            log.info("Parsing augmentation document for analysis...")
            aug_tree = parse_for_analysis(aug_source)

        # --- Run Document Analyzer ---
        log.info("Initializing Document Analyzer...")
        analyzer = DocumentAnalyzer(original_tree, processed_tree, aug_tree)
        report = analyzer.generate_report()
        
        report_path = os.path.join(session_path, "quality_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        log.info(f"  -> Quality report saved to: {report_path}")

        # --- Contribution Tracker Report ---
        # Note: A full contribution trace requires integrating the tracker
        # into the main `modular_refactor.py` loop. This script can only
        # show the final structural analysis.
        log.info("Analysis complete. For a detailed chunk contribution report,")
        log.info("ensure the ContributionTracker is active during the main processing run.")

    except Exception as e:
        log.error(f"An error occurred during analysis: {e}")