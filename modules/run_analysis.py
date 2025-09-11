"""
Main Entry Point for Post-Processing Analysis.

This script loads the results of a processing session and uses the unified
analysis engine to generate comprehensive reports on quality and contribution.
"""
import os
import json
import logging
from .analysis_engine import DocumentAnalyzer
from .knowledge_graph_processor import KnowledgeGraphProcessor
from .embedding_client import UnifiedEmbeddingClient
from config import SEMANTIC_MAPPING_CONFIG, KG_CONFIG, EMBEDDING_COHESION_CONFIG

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
            # extract_document_sections returns (chunks, preserved_data)
            chunks, _preserved = extract_document_sections(content, source_path=file_path)
            # Build a nested tree using each chunk's hierarchy_path metadata
            tree = {}
            for chunk in chunks:
                path = chunk.get('metadata', {}).get('hierarchy_path', []) or ['Preamble']
                current = tree
                for i, title in enumerate(path):
                    if title not in current:
                        current[title] = {
                            'metadata': {'hierarchy_path': path[:i+1]},
                            'chunks': [],
                            'subsections': {}
                        }
                    if i == len(path) - 1:
                        current[title]['chunks'].append(chunk)
                    current = current[title]['subsections']
            return tree

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

        # --- Unified Knowledge Graph Build + Dump (optional, via config) ---
        try:
            if KG_CONFIG.get('ENABLE_KG_UNIFIED') and KG_CONFIG.get('DUMP_ON_ANALYSIS'):
                log.info("Initializing Knowledge Graph build for analysis dump...")
                
                # Helper: flatten chunks from a mapped/processed tree
                def _flatten_chunks(tree_dict):
                    mapped, orphans = [], []
                    if isinstance(tree_dict, dict):
                        orphan_list = tree_dict.get('Orphaned_Content')
                        if isinstance(orphan_list, list):
                            orphans.extend([c for c in orphan_list if isinstance(c, dict)])
                        def walk(level):
                            for _, node in level.items():
                                if not isinstance(node, dict):
                                    continue
                                chunks = node.get('chunks', [])
                                if isinstance(chunks, list):
                                    mapped.extend([c for c in chunks if isinstance(c, dict)])
                                subs = node.get('subsections')
                                if isinstance(subs, dict):
                                    walk(subs)
                        walk({k: v for k, v in tree_dict.items() if k != 'Orphaned_Content'})
                    return mapped + orphans
                
                all_chunks = []
                mapped_tree_path = os.path.join(session_path, "2_mapped_tree.json")
                if os.path.exists(mapped_tree_path):
                    try:
                        with open(mapped_tree_path, 'r', encoding='utf-8') as f:
                            mapped_tree = json.load(f)
                        all_chunks = _flatten_chunks(mapped_tree)
                        log.info(f"  -> Loaded {len(all_chunks)} chunks from mapped tree for KG.")
                    except Exception as e:
                        log.warning(f"  -> Failed to load mapped tree, falling back to processed tree: {e}")
                        all_chunks = _flatten_chunks(processed_tree)
                else:
                    all_chunks = _flatten_chunks(processed_tree)
                    log.info(f"  -> Using processed tree with {len(all_chunks)} chunks for KG.")

                # Initialize embedding model consistent with pipeline
                embedding_model = UnifiedEmbeddingClient(SEMANTIC_MAPPING_CONFIG)
                kgp = KnowledgeGraphProcessor(all_chunks, embedding_model)
                compute_cohesion = bool(EMBEDDING_COHESION_CONFIG.get('ENABLE', False))
                kgp.build_unified_graph(
                    semantic_top_k=KG_CONFIG.get('SEMANTIC_TOP_K', 5),
                    semantic_threshold=KG_CONFIG.get('SEMANTIC_THRESHOLD', 0.8),
                    duplicate_threshold=KG_CONFIG.get('DUPLICATE_THRESHOLD', 0.95),
                    compute_cohesion=compute_cohesion
                )
                dump_path = kgp.dump_unified_graph(
                    output_dir=session_path,
                    fmt=KG_CONFIG.get('DUMP_FORMAT', 'json'),
                    filename=KG_CONFIG.get('DUMP_FILENAME', 'knowledge_graph_unified.json')
                )
                if dump_path:
                    log.info(f"  -> Unified Knowledge Graph dumped: {dump_path}")
                else:
                    log.warning("  -> Unified Knowledge Graph dump failed or was skipped.")
        except Exception as kg_err:
            log.warning(f"KG build/dump during analysis encountered a non-fatal error: {kg_err}")

        # --- Contribution Tracker Report ---
        # Note: A full contribution trace requires integrating the tracker
        # into the main `modular_refactor.py` loop. This script can only
        # show the final structural analysis.
        log.info("Analysis complete. For a detailed chunk contribution report,")
        log.info("ensure the ContributionTracker is active during the main processing run.")

    except Exception as e:
        log.error(f"An error occurred during analysis: {e}")