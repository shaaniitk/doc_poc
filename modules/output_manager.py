"""
Hierarchy-Aware Output Management and Document Aggregation.

This module is responsible for saving all processing outputs. It can save the
content of each individual node in a processed tree and can aggregate the
final, processed document tree into a valid, well-formatted output file.
"""
import os
from datetime import datetime
from config import OUTPUT_FORMATS, SEMANTIC_MAPPING_CONFIG
import json 
import re
from typing import Dict, Any, List, Tuple

class OutputManager:
    def __init__(self, base_path="outputs"):
        self.base_path = base_path
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = os.path.join(base_path, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        # Create a dedicated subfolder for the individual node outputs
        self.nodes_path = os.path.join(self.session_path, "processed_nodes")
        os.makedirs(self.nodes_path, exist_ok=True)

    # --- NEW: HIERARCHICAL NODE SAVING ---
    def save_processed_tree_nodes(self, processed_tree):
        """
        Saves the processed content of every node in the tree as a separate file.

        This is a valuable feature for debugging and reviewing the output of
        the LLM at each step of the document's hierarchy.

        Args:
            processed_tree (dict): The final, processed hierarchical document tree.
        """
        # Start the recursive saving process at the top level of the tree.
        self._save_node_recursively(processed_tree, path_parts=[])

    def _save_node_recursively(self, node_level, path_parts):
        """
        The core recursive engine for saving each node's content.
        """
        for title, node_data in node_level.items():
            # This check gracefully handles special, non-dictionary items
            # like the 'Orphaned_Content' list, preventing the crash.
            if not isinstance(node_data, dict):
                continue
            current_path = path_parts + [title]
            
            # If the node has processed content, save it.
            content_to_save = node_data.get('processed_content')
            if content_to_save:
                # Create a descriptive filename from the hierarchical path.
                # e.g., ['Section 1', 'Subsection 1.1'] -> "Section_1_Subsection_1.1.tex"
                safe_title = title.replace(" ", "_").replace(".", "")
                filename = "_".join([p.replace(" ", "_").replace(".", "") for p in path_parts] + [safe_title]) + ".tex"
                file_path = os.path.join(self.nodes_path, filename)

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content_to_save)

            # Recurse into the subsections, passing down the updated path.
            if node_data.get('subsections'):
                self._save_node_recursively(node_data['subsections'], path_parts=current_path)

    # --- Analytics (Phase 5) ---
    def _iter_chunks(self, tree: Dict[str, Any]) -> Tuple[List[dict], List[dict]]:
        """Return (mapped_chunks, orphan_chunks) flattened from a mapped tree."""
        mapped: List[dict] = []
        orphans: List[dict] = []
        # Orphans are kept at top-level key by convention
        if isinstance(tree, dict):
            orphan_list = tree.get('Orphaned_Content')
            if isinstance(orphan_list, list):
                orphans.extend([c for c in orphan_list if isinstance(c, dict)])

            def walk(node_level: Dict[str, Any]):
                for _, node in node_level.items():
                    if not isinstance(node, dict):
                        continue
                    chunks = node.get('chunks', [])
                    if isinstance(chunks, list):
                        mapped.extend([c for c in chunks if isinstance(c, dict)])
                    subs = node.get('subsections')
                    if isinstance(subs, dict):
                        walk(subs)
            walk({k: v for k, v in tree.items() if k != 'Orphaned_Content'})
        return mapped, orphans

    def compute_basic_analytics(self, mapped_tree: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute orphan stats, low-confidence metrics, and candidate distributions
        from a mapped tree prior to LLM processing. Safe against missing fields.
        """
        threshold = SEMANTIC_MAPPING_CONFIG.get('similarity_threshold', 0.6)
        margin = SEMANTIC_MAPPING_CONFIG.get('low_confidence_margin', SEMANTIC_MAPPING_CONFIG.get('soft_accept_margin', 0.05))
        gap_margin = SEMANTIC_MAPPING_CONFIG.get('gap_accept_margin', 0.1)

        mapped_chunks, orphan_chunks = self._iter_chunks(mapped_tree)
        all_chunks = mapped_chunks + orphan_chunks

        def get_md(c: dict) -> dict:
            return c.get('metadata', {}) if isinstance(c, dict) else {}

        # Counts
        total_chunks = len(all_chunks)
        mapped_count = len(mapped_chunks)
        orphan_count = len(orphan_chunks)
        rescued_count = sum(1 for c in mapped_chunks if get_md(c).get('soft_assigned', False))

        # NEW: rescued_by breakdown (among mapped chunks)
        rescued_by_breakdown: Dict[str, int] = {}
        for c in mapped_chunks:
            rb = get_md(c).get('rescued_by')
            if isinstance(rb, str) and rb:
                rescued_by_breakdown[rb] = rescued_by_breakdown.get(rb, 0) + 1

        # Scores & distances
        scores: List[float] = []
        distances: List[float] = []
        low_conf_mapped = 0
        ambiguous = 0
        candidate_presence = 0
        top2_gap_list: List[float] = []
        for c in all_chunks:
            md = get_md(c)
            if 'assignment_score' in md:
                try:
                    s = float(md['assignment_score'])
                    scores.append(s)
                except Exception:
                    pass
            if 'distance_to_nearest' in md:
                try:
                    d = float(md['distance_to_nearest'])
                    distances.append(d)
                except Exception:
                    pass
            # low-confidence among mapped only
            if c in mapped_chunks:
                s = float(md.get('assignment_score', 0.0))
                soft = bool(md.get('soft_assigned', False))
                if soft or (threshold - margin) <= s < threshold:
                    low_conf_mapped += 1
            # ambiguity via top-2 candidate gap
            cand = md.get('candidate_sections')
            if isinstance(cand, list) and len(cand) >= 1:
                candidate_presence += 1
                if len(cand) >= 2:
                    try:
                        top_gap = float(cand[0].get('score', 0.0)) - float(cand[1].get('score', 0.0))
                        if top_gap < gap_margin:
                            ambiguous += 1
                        top2_gap_list.append(top_gap)
                    except Exception:
                        pass

        def safe_avg(vals: List[float]) -> float:
            return float(sum(vals) / len(vals)) if vals else 0.0

        # Orphan suggestion signals
        orphans_with_suggestions = 0
        orphan_top_suggestions: Dict[str, int] = {}
        orphan_avg_dist: List[float] = []
        for c in orphan_chunks:
            md = get_md(c)
            cand = md.get('candidate_sections')
            if isinstance(cand, list) and cand:
                orphans_with_suggestions += 1
                top = cand[0]
                path_str = top.get('path_str', 'UNKNOWN')
                orphan_top_suggestions[path_str] = orphan_top_suggestions.get(path_str, 0) + 1
                try:
                    orphan_avg_dist.append(float(top.get('distance', 0.0)))
                except Exception:
                    pass

        # NEW: Orphan reasons breakdown
        orphan_reasons: Dict[str, int] = {}
        for c in orphan_chunks:
            reason = get_md(c).get('orphan_reason', 'unspecified')
            if not isinstance(reason, str) or not reason:
                reason = 'unspecified'
            orphan_reasons[reason] = orphan_reasons.get(reason, 0) + 1

        analytics = {
            'totals': {
                'total_chunks': total_chunks,
                'mapped_chunks': mapped_count,
                'orphan_chunks': orphan_count,
                'rescued_chunks': rescued_count
            },
            'confidence': {
                'low_confidence_mapped': low_conf_mapped,
                'avg_assignment_score': round(safe_avg(scores), 4),
                'avg_distance_to_nearest': round(safe_avg(distances), 4)
            },
            'candidates': {
                'has_candidate_fraction': round(candidate_presence / total_chunks, 4) if total_chunks else 0.0,
                'ambiguous_count': ambiguous,
                'gap_margin': gap_margin,
                'avg_top2_gap': round(safe_avg(top2_gap_list), 4)
            },
            'orphans': {
                'count': orphan_count,
                'with_suggestions': orphans_with_suggestions,
                'top_suggested_sections': sorted([
                    {'section': k, 'count': v} for k, v in orphan_top_suggestions.items()
                ], key=lambda x: x['count'], reverse=True)[:5],
                'avg_top_suggestion_distance': round(safe_avg(orphan_avg_dist), 4),
                'reasons_breakdown': sorted([
                    {'reason': k, 'count': v} for k, v in orphan_reasons.items()
                ], key=lambda x: x['count'], reverse=True)
            },
            'rescues': {
                'soft_assigned_count': rescued_count,
                'rescued_by_breakdown': sorted([
                    {'rescued_by': k, 'count': v} for k, v in rescued_by_breakdown.items()
                ], key=lambda x: x['count'], reverse=True)
            },
            'config': {
                'similarity_threshold': threshold,
                'low_confidence_margin': margin,
                'gap_accept_margin': gap_margin
            }
        }
        return analytics

    def save_analytics_json(self, filename: str, analytics: Dict[str, Any]) -> str:
        """Persist analytics dict to JSON within the session path."""
        return self.save_json_output(filename, analytics)

    # --- AGGREGATION & LOGGING (No changes to the methods below) ---
    def aggregate_document(self, final_document_string, output_format="latex"):
        """
        Saves the final, fully formatted document string to a file.
        This method is now simpler: it just performs the file write operation.
        """
        format_config = OUTPUT_FORMATS.get(output_format, OUTPUT_FORMATS['latex'])
        final_path = os.path.join(self.session_path, f"final_document{format_config['extension']}")
        
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(final_document_string)
        
        return final_path

    def save_processing_log(self, log_entries):
        """Save a log of all processing steps for traceability."""
        log_path = os.path.join(self.session_path, "processing_log.txt")
        
        with open(log_path, 'w', encoding='utf-8') as f:
            for entry in log_entries:
                f.write(f"{entry}\n")
        
        return log_path
    
    def save_json_output(self, filename, data):
        """Saves a Python dictionary or list to a formatted JSON file."""
        file_path = os.path.join(self.session_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            # Use indent=2 for nice, human-readable formatting
            json.dump(data, f, indent=2)
        return file_path

    def save_generic_output(self, filename, content):
        """Saves a generic string content to a file in the session path."""
        file_path = os.path.join(self.session_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path


def final_latex_sanitization(document_string: str) -> str:
    """
    A final, aggressive sanitization pass to remove nested LaTeX document structures
    that may have been generated by the LLM.
    """
    # Preserve the top-level LaTeX document, sanitize only nested documents inside the body
    start_marker = "\\begin{document}"
    end_marker = "\\end{document}"

    start_idx = document_string.find(start_marker)
    end_idx = document_string.rfind(end_marker)

    # If not a full LaTeX document or malformed, do nothing
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return document_string

    preamble_and_begin = document_string[:start_idx]
    body_content = document_string[start_idx + len(start_marker): end_idx]
    tail = document_string[end_idx:]  # includes \end{document}

    # Remove any fully nested LaTeX documents within the body only
    nested_doc_pattern = re.compile(r"\\documentclass.*?\\begin{document}(.*?)\\end{document}", re.DOTALL)

    def extract_content(match):
        return match.group(1).strip()

    body_content = nested_doc_pattern.sub(extract_content, body_content)

    # Remove any stray \\documentclass lines that accidentally appeared inside the body
    body_content = re.sub(r"\\documentclass\{.*?\}", "", body_content)

    # Reassemble preserving the original preamble and outer document structure
    return preamble_and_begin + start_marker + body_content + tail