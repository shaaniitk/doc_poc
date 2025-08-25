"""
State-of-the-Art Hierarchical Output Formatter.

This module is the central rendering engine for the application. It takes a final,
processed hierarchical document tree and converts it into a valid, formatted
string in a specified output format (e.g., LaTeX, Markdown).

Key Features:
- Centralized Logic: All document-to-string conversion logic lives here.
- Recursive Rendering: Elegantly handles any depth of document nesting.
- Multi-Format Support: Easily extensible to support various output formats
  by adding new recursive rendering methods.
- Separation of Concerns: Focuses solely on creating the content string, leaving
  file I/O operations to the OutputManager.
"""

from config import OUTPUT_FORMATS

class HierarchicalOutputFormatter:
    """
    Formats a hierarchical document tree into a string of a specified format.
    """
    def __init__(self, format_type="latex"):
        self.format_type = format_type
        if format_type not in OUTPUT_FORMATS:
            raise ValueError(f"Unsupported format type: '{format_type}'. Supported formats are {list(OUTPUT_FORMATS.keys())}")
        self.config = OUTPUT_FORMATS[format_type]

    def format_document(self, processed_tree):
        """
        The main entry point for formatting the entire document tree.

        Args:
            processed_tree (dict): The final, processed hierarchical document tree.

        Returns:
            str: The fully formatted document as a single string.
        """
        # Select the appropriate recursive renderer based on the format type.
        render_method_map = {
            "latex": self._to_latex_recursive,
            "markdown": self._to_markdown_recursive
        }
        render_method = render_method_map.get(self.format_type)

        if not render_method:
            raise NotImplementedError(f"No rendering method implemented for format: '{self.format_type}'")

        header = self.config.get('header', '')
        footer = self.config.get('footer', '')
        
        # Separate the orphaned content from the main hierarchical tree.
        # .pop() conveniently removes it so the recursive renderer doesn't see it.
        orphaned_chunks = processed_tree.pop('Orphaned_Content', [])
        
        # Generate the body content for the main tree by starting the recursive process.
        body_parts = render_method(processed_tree, level=1)
        
        # If there were any orphaned chunks, format them and append them at the end.
        if orphaned_chunks:
            # For LaTeX, use an un-numbered section
            if self.format_type == 'latex':
                body_parts.append("\n\\section*{Uncategorized Content}")
            elif self.format_type == 'markdown':
                body_parts.append("\n# Uncategorized Content")
            
            orphaned_content = "\n\n".join([chunk['content'] for chunk in orphaned_chunks])
            body_parts.append(orphaned_content)
        
        # Combine all parts into the final document string.
        full_document_parts = filter(None, [header, *body_parts, footer])
        return "\n".join(full_document_parts)

    def _to_latex_recursive(self, node_level, level):
        """
        The recursive engine for rendering the document tree into LaTeX source.
        """
        rendered_parts = []
        level_to_command = {1: "section", 2: "subsection", 3: "subsubsection"}
        command = level_to_command.get(level, "paragraph")

        for title, node_data in node_level.items():
            rendered_parts.append(f"\\{command}{{{title}}}")
            
            if node_data.get('processed_content'):
                rendered_parts.append(node_data['processed_content'])
            
            rendered_parts.append("") # For readability

            if node_data.get('subsections'):
                child_parts = self._to_latex_recursive(node_data['subsections'], level + 1)
                rendered_parts.extend(child_parts)
        
        return rendered_parts

    def _to_markdown_recursive(self, node_level, level):
        """
        The recursive engine for rendering the document tree into Markdown source.
        """
        rendered_parts = []
        
        for title, node_data in node_level.items():
            # Markdown heading level is determined by the number of '#'
            heading_prefix = '#' * level
            rendered_parts.append(f"{heading_prefix} {title}")
            
            if node_data.get('processed_content'):
                rendered_parts.append(node_data['processed_content'])
            
            rendered_parts.append("") # For readability

            if node_data.get('subsections'):
                child_parts = self._to_markdown_recursive(node_data['subsections'], level + 1)
                rendered_parts.extend(child_parts)

        return rendered_parts