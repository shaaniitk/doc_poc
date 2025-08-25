"""
Track how chunks contribute to different sections in the hierarchical tree.
"""

class ContributionTracker:
    def __init__(self):
        self.chunk_contributions = {}
        self.section_sources = {}
        self.chunk_id_counter = 0

    def track_chunk_assignment(self, chunk, target_section_path):
        """
        Track which chunk goes to which target section in the tree.

        Args:
            chunk (dict): The chunk dictionary from the AST chunker.
            target_section_path (list): The hierarchical path of the target section.
        """
        chunk_id = self.chunk_id_counter
        self.chunk_id_counter += 1

        # Convert paths to readable strings
        original_path_str = " -> ".join(chunk['metadata'].get('hierarchy_path', ['Unknown']))
        target_path_str = " -> ".join(target_section_path)
        
        content_preview = chunk['content'][:100].replace('\n', ' ') + "..."

        self.chunk_contributions[chunk_id] = {
            'original_path': original_path_str,
            'target_path': target_path_str,
            'content_preview': content_preview
        }

        if target_path_str not in self.section_sources:
            self.section_sources[target_path_str] = []
        
        self.section_sources[target_path_str].append({
            'chunk_id': chunk_id,
            'original_path': original_path_str
        })

    def save_report(self, output_path):
        """Saves a comprehensive contribution report in Markdown format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Chunk Contribution Analysis\n\n")
            f.write("This report shows how chunks from the original document(s) were mapped to the final document structure.\n\n")
            
            f.write("## Section Sources\n")
            f.write("Shows which original chunks were used to build each final section.\n\n")
            for section, sources in sorted(self.section_sources.items()):
                f.write(f"### Final Section: `{section}`\n")
                f.write(f"- **Source Chunks:** {len(sources)}\n")
                for source in sources:
                    f.write(f"  - Chunk `{source['chunk_id']}` (from *{source['original_path']}*)\n")
            
            f.write("\n## Chunk Distribution\n")
            f.write("Shows where each original chunk ended up in the final structure.\n\n")
            for chunk_id, info in sorted(self.chunk_contributions.items()):
                f.write(f"### Chunk `{chunk_id}` (from *{info['original_path']}*)\n")
                f.write(f"- **Content Preview:** `{info['content_preview']}`\n")
                f.write(f"- **Mapped To:** `{info['target_path']}`\n")
        
        return output_path