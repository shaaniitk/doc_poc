"""Track how chunks contribute to different sections"""
import json

class ContributionTracker:
    def __init__(self):
        self.chunk_contributions = {}
        self.section_sources = {}
        
    def track_chunk_assignment(self, chunk_id, original_section, target_section, chunk_content_preview):
        """Track which chunk goes to which section"""
        if chunk_id not in self.chunk_contributions:
            self.chunk_contributions[chunk_id] = {
                'original_section': original_section,
                'target_sections': [],
                'content_preview': chunk_content_preview[:100] + "..." if len(chunk_content_preview) > 100 else chunk_content_preview
            }
        
        if target_section not in self.chunk_contributions[chunk_id]['target_sections']:
            self.chunk_contributions[chunk_id]['target_sections'].append(target_section)
        
        # Track reverse mapping
        if target_section not in self.section_sources:
            self.section_sources[target_section] = []
        
        self.section_sources[target_section].append({
            'chunk_id': chunk_id,
            'original_section': original_section,
            'content_preview': chunk_content_preview[:100] + "..."
        })
    
    def generate_contribution_report(self):
        """Generate detailed contribution report"""
        report = {
            'chunk_to_sections': self.chunk_contributions,
            'section_to_chunks': self.section_sources,
            'summary': {
                'total_chunks': len(self.chunk_contributions),
                'total_sections': len(self.section_sources),
                'multi_section_chunks': len([c for c in self.chunk_contributions.values() if len(c['target_sections']) > 1])
            }
        }
        return report
    
    def save_contribution_report(self, output_path):
        """Save contribution report to file"""
        report = self.generate_contribution_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Chunk Contribution Analysis\n\n")
            
            f.write(f"## Summary\n")
            f.write(f"- Total chunks processed: {report['summary']['total_chunks']}\n")
            f.write(f"- Total sections created: {report['summary']['total_sections']}\n")
            f.write(f"- Chunks used in multiple sections: {report['summary']['multi_section_chunks']}\n\n")
            
            f.write("## Section Sources\n")
            for section, sources in report['section_to_chunks'].items():
                f.write(f"\n### {section}\n")
                f.write(f"Sources: {len(sources)} chunks\n")
                for source in sources:
                    f.write(f"- Chunk {source['chunk_id']} (from {source['original_section']}): {source['content_preview']}\n")
            
            f.write("\n## Chunk Distribution\n")
            for chunk_id, info in report['chunk_to_sections'].items():
                f.write(f"\n**Chunk {chunk_id}** (from {info['original_section']}):\n")
                f.write(f"- Content: {info['content_preview']}\n")
                f.write(f"- Used in sections: {', '.join(info['target_sections'])}\n")
        
        return output_path