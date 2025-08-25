"""
State-of-the-Art Unified Analysis Engine.

This module provides a single, powerful class for analyzing the results of
document processing. It operates directly on the hierarchical document trees,
offering a deep and accurate measure of quality, preservation, and structure.

This engine replaces the functionality of several previous, redundant analysis scripts.
"""

class DocumentAnalyzer:
    """
    Analyzes and compares document trees to generate a comprehensive quality report.
    """
    def __init__(self, original_tree, processed_tree, aug_tree=None):
        self.original_stats = self._count_elements_recursive(original_tree)
        self.processed_stats = self._count_elements_recursive(processed_tree)
        self.aug_stats = self._count_elements_recursive(aug_tree) if aug_tree else {}

    def _count_elements_recursive(self, node_level):
        """
        Traverses a document tree to count all structural elements.
        This is far more accurate than regex-based counting.
        """
        stats = {'sections': 0, 'subsections': 0, 'subsubsections': 0, 
                 'figure': 0, 'table': 0, 'equation': 0, 'chunks': 0, 'chars': 0}
        
        # Mapping from node type (from chunker) to stats key
        type_map = {'figure': 'figure', 'table': 'table', 'equation': 'equation'}

        for title, node_data in node_level.items():
            # Count sections based on level (a simple heuristic)
            level = len(node_data.get('metadata', {}).get('hierarchy_path', []))
            if level == 1: stats['sections'] += 1
            elif level == 2: stats['subsections'] += 1
            elif level == 3: stats['subsubsections'] += 1

            # Count chunks and characters
            num_chunks = len(node_data.get('chunks', []))
            stats['chunks'] += num_chunks
            for chunk in node_data.get('chunks', []):
                stats['chars'] += len(chunk['content'])
                chunk_type = chunk.get('type')
                if chunk_type in type_map:
                    stats[type_map[chunk_type]] += 1

            if node_data.get('subsections'):
                child_stats = self._count_elements_recursive(node_data['subsections'])
                for key, value in child_stats.items():
                    stats[key] += value
        return stats

    def analyze_preservation(self):
        """Calculates the preservation rate of key structural elements."""
        preservation = {}
        # Use a combined original+aug count for augmentation analysis
        total_original = self.original_stats.copy()
        if self.aug_stats:
            for key in total_original:
                total_original[key] += self.aug_stats.get(key, 0)

        for key in ['sections', 'figure', 'table', 'equation']:
            original_count = total_original.get(key, 0)
            processed_count = self.processed_stats.get(key, 0)
            if original_count > 0:
                rate = (processed_count / original_count) * 100
            else:
                rate = 100.0 if processed_count == 0 else 0.0
            preservation[key] = {'original': original_count, 'processed': processed_count, 'rate': rate}
        return preservation

    def analyze_structure_change(self):
        """Analyzes how the document's structure and size have changed."""
        total_original_chars = self.original_stats['chars']
        if self.aug_stats:
            total_original_chars += self.aug_stats['chars']

        char_change_percent = 0
        if total_original_chars > 0:
            char_change_percent = ((self.processed_stats['chars'] - total_original_chars) / total_original_chars) * 100

        return {
            'char_change_percent': char_change_percent,
            'original_sections': self.original_stats['sections'],
            'processed_sections': self.processed_stats['sections'],
            'original_depth': self.original_stats['subsections'] + self.original_stats['subsubsections'],
            'processed_depth': self.processed_stats['subsections'] + self.processed_stats['subsubsections']
        }

    def calculate_quality_score(self):
        """Calculates a weighted quality score based on preservation and structure."""
        score = 100.0
        preservation = self.analyze_preservation()
        structure = self.analyze_structure_change()

        # Deduct for loss of important elements (30 points total)
        for key in ['equation', 'figure', 'table']:
            rate = preservation.get(key, {}).get('rate', 100.0)
            if rate < 95.0:
                score -= 10 * (1 - rate / 100) # Proportional deduction

        # Deduct for major structural changes in a refactoring context (20 points)
        if not self.aug_stats:
             if abs(structure['char_change_percent']) > 25.0:
                 score -= 20

        # Reward for increased structure
        if structure['processed_depth'] > structure['original_depth']:
            score += 5 # Bonus points for adding subsections

        return max(0, min(100, score))

    def generate_report(self):
        """Generates a full, human-readable analysis report."""
        preservation = self.analyze_preservation()
        structure = self.analyze_structure_change()
        score = self.calculate_quality_score()
        
        report_lines = [
            "# Document Processing Analysis Report",
            "---",
            f"**Overall Quality Score: {score:.1f}/100**",
            "\n## 1. Content Preservation Analysis",
            "| Element   | Original | Processed | Preservation Rate |",
            "|-----------|----------|-----------|-------------------|",
        ]
        for key, data in preservation.items():
            report_lines.append(f"| {key.title():<9} | {data['original']:<8} | {data['processed']:<9} | {data['rate']:.1f}%              |")
        
        report_lines.extend([
            "\n## 2. Structural & Size Analysis",
            f"- **Character Count Change:** {structure['char_change_percent']:+.1f}%",
            f"- **Section Count:** {structure['original_sections']} -> {structure['processed_sections']}",
            f"- **Subsection/Depth Count:** {structure['original_depth']} -> {structure['processed_depth']}",
        ])
        return "\n".join(report_lines)