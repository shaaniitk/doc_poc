"""
State-of-the-Art Hierarchical Output Formatter using a Templating Engine.
This module is the central rendering engine. It takes a final, processed
hierarchical document tree and converts it into a valid, formatted string
using the Jinja2 templating engine.
Key Features:
- Decoupled Logic: Separates data (the document tree) from presentation (the templates).
- Easy to Extend: Add new output formats by simply creating a new template file.
- Powerful Templating: Leverages Jinja2 for loops, conditionals, and macros.
- Maintainable: Format changes are made in template files, not Python code.
"""

from jinja2 import Environment, FileSystemLoader # <-- JINJA2 IMPORT
import os
from config import SEMANTIC_MAPPING_CONFIG

class HierarchicalOutputFormatter:
    """
    Formats a hierarchical document tree into a string using Jinja2 templates.
    """
    def __init__(self, format_type="latex"):
        self.format_type = format_type
        
        # --- JINJA2 ENVIRONMENT SETUP ---
        # Look for templates in a directory named 'templates'
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(template_dir))

        # IMPORTANT: For LaTeX, we change Jinja's delimiters to avoid conflicts
        # with LaTeX's curly braces {} and comment % characters.
        # Now, in .tex.j2 files, use:
        # ((* for item in items *)) ... ((* endfor *)) for logic blocks
        # ((( variable ))) for printing variables
        # ((# a comment #)) for comments
        if self.format_type == 'latex':
            self.env.block_start_string = '((*'
            self.env.block_end_string = '*))'
            self.env.variable_start_string = '((('
            self.env.variable_end_string = ')))'
            self.env.comment_start_string = '((#'
            self.env.comment_end_string = '#))'
        # --- END JINJA2 SETUP ---

    def format_document(self, processed_tree, preserved_data=None):
        """
        The main entry point for formatting the entire document tree.
        It loads the appropriate template and renders it with the tree data.
        """
        template_map = {
            "latex": "latex_template.tex.j2",
            "markdown": "markdown_template.md.j2"
        }
        template_name = template_map.get(self.format_type)

        if not template_name:
            raise NotImplementedError(f"No template found for format: '{self.format_type}'")

        try:
            template = self.env.get_template(template_name)
        except Exception as e:
            raise FileNotFoundError(f"Could not load template '{template_name}'. Ensure it exists in the '{self.template_dir}' directory. Error: {e}")

        # Separate orphaned content so it can be handled specifically in the template
        # .pop() conveniently removes it so the recursive renderer doesn't see it.
        orphaned_chunks = processed_tree.pop('Orphaned_Content', [])
        
        # --- Phase 2 UX: Annotate nodes with low-confidence flags for rendering ---
        def annotate_low_confidence(node):
            cfg = SEMANTIC_MAPPING_CONFIG
            threshold = cfg.get('similarity_threshold', 0.6)
            margin = cfg.get('low_confidence_margin', cfg.get('soft_accept_margin', 0.05))
        
            def walk(n):
                if isinstance(n, dict):
                    # Count low-confidence/soft-assigned chunks on this node
                    if 'chunks' in n:
                        low_count = 0
                        for ch in n['chunks']:
                            md = ch.get('metadata', {})
                            score = md.get('assignment_score', 0.0)
                            soft = md.get('soft_assigned', False)
                            # Mark as low-confidence if soft-assigned or within margin below threshold
                            if soft or (score < threshold and score >= threshold - margin):
                                low_count += 1
                        if low_count:
                            n.setdefault('data', {})['low_confidence_count'] = low_count
                    # Recurse into subsections if present
                    subs = n.get('subsections', {})
                    if isinstance(subs, dict):
                        for k in list(subs.keys()):
                            walk(subs[k])
                elif isinstance(n, list):
                    for item in n:
                        walk(item)
            walk(node)
            return node

        # Apply low-confidence annotations before rendering
        processed_tree = annotate_low_confidence(processed_tree)

        # The 'render' call passes your Python data to the template and returns the final string
        return template.render(
                    processed_tree=processed_tree,
                    orphaned_content=orphaned_chunks,
                    preserved=preserved_data or {} # Ensure it's never None
                    )