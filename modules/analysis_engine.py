"""Unified Analysis Engine.

This module provides a single, powerful class for analyzing the results of
document processing. It operates directly on the hierarchical document trees,
offering a deep and accurate measure of quality, preservation, and structure.

This engine replaces the functionality of several previous, redundant analysis scripts.
Enhanced with semantic preservation metrics and topic drift detection.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import config
from .embedding_client import UnifiedEmbeddingClient

log = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """
    Analyzes and compares document trees to generate a comprehensive quality report.
    Enhanced with semantic preservation metrics and topic drift detection.
    """
    def __init__(self, original_tree, processed_tree, aug_tree=None):
        self.original_stats = self._count_elements_recursive(original_tree)
        self.processed_stats = self._count_elements_recursive(processed_tree)
        self.aug_stats = self._count_elements_recursive(aug_tree) if aug_tree else {}
        
        # Store trees for semantic analysis
        self.original_tree = original_tree
        self.processed_tree = processed_tree
        self.aug_tree = aug_tree
        
        # Initialize semantic analysis components
        self.llm_config = getattr(config, 'LOCAL_LLM_REFINEMENT', {})
        self.semantic_cache = {}
        self.embedding_client = None
        if self.llm_config.get('enable_semantic_analysis', False):
            try:
                self.embedding_client = UnifiedEmbeddingClient()
            except Exception as e:
                log.warning(f"Failed to initialize embedding client for semantic analysis: {e}")
        
        # Cache for semantic analysis results
        self._semantic_cache = {}
        self.semantic_cache = self._semantic_cache
        
        # Initialize semantic config
        self.semantic_config = self.llm_config
        
        # Add mistral_config alias for test compatibility
        self.mistral_config = getattr(config, 'MISTRAL_REFINEMENT', {})

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
    
    def _create_cache_key(self, original_sections, processed_sections) -> str:
        """
        Create a cache key from section content for semantic analysis caching.
        
        Args:
            original_sections: List of original document sections
            processed_sections: List of processed document sections
            
        Returns:
            String cache key
        """
        import hashlib
        
        # Create a hash from the content of both section lists
        content_str = ""
        
        # Add original sections content
        if original_sections:
            for section in original_sections:
                content_str += str(section)
        
        # Add processed sections content
        if processed_sections:
            for section in processed_sections:
                content_str += str(section)
        
        # Create hash
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _extract_text_content(self, section_or_tree) -> str:
        """
        Extract text content from a section or document tree.
        
        Args:
            section_or_tree: Section dict or document tree structure
            
        Returns:
            String of extracted text content
        """
        if not section_or_tree:
            return ""
        
        # Handle simple section with 'content' key
        if isinstance(section_or_tree, dict) and 'content' in section_or_tree:
            return section_or_tree['content']
        
        # Handle document tree format
        tree = section_or_tree.get('tree', section_or_tree) if isinstance(section_or_tree, dict) else section_or_tree
        if not tree:
            return ""
            
        text_chunks = []
        
        def extract_recursive(node_level):
            if not isinstance(node_level, dict):
                return
                
            for title, node_data in node_level.items():
                # Add section title
                if title and isinstance(title, str):
                    text_chunks.append(title)
                
                # Handle node_data safely
                if not isinstance(node_data, dict):
                    continue
                    
                # Add chunk content
                for chunk in node_data.get('chunks', []):
                    if isinstance(chunk, dict):
                        content = chunk.get('content', '')
                        if content and isinstance(content, str):
                            text_chunks.append(content)
                
                # Recurse into subsections
                if node_data.get('subsections'):
                    extract_recursive(node_data['subsections'])
        
        extract_recursive(tree)
        return ' '.join(chunk.strip() for chunk in text_chunks if chunk.strip())
    
    def _extract_text_content_for_section(self, section):
        """Extract text content from a section."""
        # This method is now redundant since _extract_text_content handles both cases
        return self._extract_text_content(section)
    
    def analyze_semantic_preservation(self, original_sections, processed_sections) -> Dict[str, Any]:
        """
        Analyze semantic preservation between original and processed sections.
        
        Args:
            original_sections: List of original document sections
            processed_sections: List of processed document sections
            
        Returns:
            Dictionary containing semantic preservation metrics
        """
        # Create cache key from section content
        cache_key = self._create_cache_key(original_sections, processed_sections)
        
        # Check cache first
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
        
        if not self.embedding_client:
            result = {
                'enabled': False,
                'reason': 'Semantic analysis disabled or embedding client unavailable'
            }
            self.semantic_cache[cache_key] = result
            return result
        
        try:
            # Extract text content from both section lists
            original_texts = []
            for section in original_sections:
                text_content = self._extract_text_content_for_section(section)
                if text_content:
                    original_texts.append(text_content)
            
            processed_texts = []
            for section in processed_sections:
                text_content = self._extract_text_content_for_section(section)
                if text_content:
                    processed_texts.append(text_content)
            
            if not original_texts or not processed_texts:
                result = {
                     'enabled': True,
                     'semantic_similarity': 0.0,
                     'content_coverage': 0.0,
                     'topic_drift_score': 1.0,
                     'overall_similarity': 1.0 if not original_texts and not processed_texts else 0.0,
                     'preservation_score': 1.0 if not original_texts and not processed_texts else 0.0,
                     'structural_change_penalty': 1.0,
                     'section_similarities': [],
                     'error': 'Insufficient text content for analysis'
                 }
                self.semantic_cache[cache_key] = result
                return result
            
            # Generate embeddings for original and processed content
            original_embeddings = self.embedding_client.encode(original_texts)
            processed_embeddings = self.embedding_client.encode(processed_texts)
            
            # Calculate overall semantic similarity
            semantic_similarity = self._calculate_document_similarity(
                original_embeddings, processed_embeddings
            )
            
            # Calculate content coverage (how much original content is preserved)
            content_coverage = self._calculate_content_coverage(
                original_embeddings, processed_embeddings
            )
            
            # Calculate topic drift score
            topic_drift_score = self._calculate_topic_drift(
                original_embeddings, processed_embeddings
            )
            
            # Calculate section-level similarities
            section_similarities = []
            for i, (orig_text, proc_text) in enumerate(zip(original_texts, processed_texts)):
                if orig_text and proc_text:
                    orig_emb = self.embedding_client.encode([orig_text])
                    proc_emb = self.embedding_client.encode([proc_text])
                    sim_matrix = self.embedding_client.similarity(orig_emb, proc_emb)
                    section_similarities.append(float(sim_matrix[0][0]))
                else:
                    section_similarities.append(0.0)
            
            # Calculate preservation score (average of similarity and coverage)
            preservation_score = (semantic_similarity + content_coverage) / 2
            
            # Generate quality assessment
            if preservation_score >= 0.8:
                quality_assessment = "Excellent preservation"
            elif preservation_score >= 0.6:
                quality_assessment = "Good preservation"
            elif preservation_score >= 0.4:
                quality_assessment = "Fair preservation"
            else:
                quality_assessment = "Poor preservation"
            
            result = {
                'enabled': True,
                'semantic_similarity': float(semantic_similarity),
                'content_coverage': float(content_coverage),
                'topic_drift_score': float(topic_drift_score),
                'overall_similarity': float(semantic_similarity),
                'section_similarities': section_similarities,
                'preservation_score': float(preservation_score),
                'structural_change_penalty': 0.0,
                'quality_assessment': quality_assessment,
                'original_chunks': len(original_texts),
                'processed_chunks': len(processed_texts)
            }
            
            # Store in cache
            self.semantic_cache[cache_key] = result
            return result
            
        except Exception as e:
            log.error(f"Semantic preservation analysis failed: {e}")
            result = {
                'enabled': True,
                'error': str(e),
                'semantic_similarity': 0.0,
                'content_coverage': 0.0,
                'topic_drift_score': 1.0,
                'overall_similarity': 0.0,
                'preservation_score': 0.0,
                'structural_change_penalty': 1.0
            }
            self.semantic_cache[cache_key] = result
            return result
    
    def _create_cache_key(self, original_sections, processed_sections) -> str:
        """Create a cache key from section content for semantic analysis caching."""
        import hashlib
        
        # Create a hash from the content of both section lists
        content_str = ""
        
        # Add original sections content
        if original_sections:
            for section in original_sections:
                content_str += str(section)
        
        # Add processed sections content
        if processed_sections:
            for section in processed_sections:
                content_str += str(section)
        
        # Create hash
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _calculate_section_similarity(self, section1, section2):
        """Calculate similarity between two sections using embeddings."""
        try:
            # Extract text content from both sections
            text1 = self._extract_text_content_for_section(section1)
            text2 = self._extract_text_content_for_section(section2)
            
            if not text1 or not text2:
                return 0.0
            
            # Generate embeddings
            emb1 = self.embedding_client.encode([text1])
            emb2 = self.embedding_client.encode([text2])
            
            # Calculate similarity
            similarity_matrix = self.embedding_client.similarity(emb1, emb2)
            return float(similarity_matrix[0][0])
            
        except Exception as e:
            logger.error(f"Error calculating section similarity: {e}")
            return 0.0
    
    def calculate_section_similarity(self, section1, section2):
        """Public method to calculate similarity between two sections."""
        return self._calculate_section_similarity(section1, section2)
    
    def _calculate_document_similarity(self, original_embeddings: np.ndarray, 
                                     processed_embeddings: np.ndarray) -> float:
        """
        Calculate overall semantic similarity between document embeddings.
        
        Args:
            original_embeddings: Embeddings from original document
            processed_embeddings: Embeddings from processed document
            
        Returns:
            Similarity score between 0 and 1
        """
        # Calculate centroid embeddings for each document
        original_centroid = np.mean(original_embeddings, axis=0)
        processed_centroid = np.mean(processed_embeddings, axis=0)
        
        # Calculate cosine similarity between centroids
        similarity_matrix = self.embedding_client.similarity(
            original_centroid.reshape(1, -1),
            processed_centroid.reshape(1, -1)
        )
        
        return similarity_matrix[0][0]
    
    def _calculate_content_coverage(self, original_embeddings: np.ndarray,
                                  processed_embeddings: np.ndarray) -> float:
        """
        Calculate how much of the original content is covered in the processed document.
        
        Args:
            original_embeddings: Embeddings from original document
            processed_embeddings: Embeddings from processed document
            
        Returns:
            Coverage score between 0 and 1
        """
        if len(original_embeddings) == 0:
            return 1.0
        
        coverage_threshold = self.mistral_config.get('coverage_threshold', 0.7)
        covered_count = 0
        
        # For each original chunk, find the best match in processed chunks
        for orig_embedding in original_embeddings:
            similarities = self.embedding_client.similarity(
                orig_embedding.reshape(1, -1),
                processed_embeddings
            )[0]
            
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0
            if max_similarity >= coverage_threshold:
                covered_count += 1
        
        return covered_count / len(original_embeddings)
    
    def _calculate_topic_drift(self, original_embeddings: np.ndarray,
                             processed_embeddings: np.ndarray) -> float:
        """
        Calculate topic drift between original and processed documents.
        
        Args:
            original_embeddings: Embeddings from original document
            processed_embeddings: Embeddings from processed document
            
        Returns:
            Topic drift score (0 = no drift, 1 = complete drift)
        """
        if len(original_embeddings) == 0 or len(processed_embeddings) == 0:
            return 1.0
        
        # Calculate topic coherence within each document
        original_coherence = self._calculate_internal_coherence(original_embeddings)
        processed_coherence = self._calculate_internal_coherence(processed_embeddings)
        
        # Calculate cross-document topic alignment
        cross_alignment = self._calculate_cross_alignment(
            original_embeddings, processed_embeddings
        )
        
        # Topic drift is inversely related to coherence preservation and alignment
        coherence_preservation = min(processed_coherence / max(original_coherence, 0.1), 1.0)
        
        # Combine metrics (lower values indicate more drift)
        topic_stability = (coherence_preservation + cross_alignment) / 2
        
        return 1.0 - topic_stability
    
    def _calculate_internal_coherence(self, embeddings: np.ndarray) -> float:
        """
        Calculate internal coherence of a document based on embedding similarities.
        
        Args:
            embeddings: Document embeddings
            
        Returns:
            Coherence score between 0 and 1
        """
        if len(embeddings) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarity_matrix = self.embedding_client.similarity(embeddings, embeddings)
        
        # Remove diagonal (self-similarities)
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]
        
        # Handle NaN values
        similarities = similarities[~np.isnan(similarities)]
        
        if len(similarities) == 0:
            return 0.0
        
        mean_sim = np.mean(similarities)
        return float(mean_sim) if not np.isnan(mean_sim) else 0.0
    
    def _calculate_cross_alignment(self, original_embeddings: np.ndarray,
                                 processed_embeddings: np.ndarray) -> float:
        """
        Calculate alignment between topics in original and processed documents.
        
        Args:
            original_embeddings: Embeddings from original document
            processed_embeddings: Embeddings from processed document
            
        Returns:
            Alignment score between 0 and 1
        """
        # Calculate cross-document similarity matrix
        cross_similarities = self.embedding_client.similarity(
            original_embeddings, processed_embeddings
        )
        
        # Calculate average maximum similarity for each original chunk
        max_similarities = np.max(cross_similarities, axis=1)
        
        return float(np.mean(max_similarities))
    
    def detect_topic_drift_patterns(self) -> Dict[str, Any]:
        """
        Detect specific patterns of topic drift in the document processing.
        
        Returns:
            Dictionary containing drift pattern analysis
        """
        if not self.embedding_client:
            return {'enabled': False, 'reason': 'Semantic analysis disabled'}
        
        try:
            original_text = self._extract_text_content(self.original_tree)
            processed_text = self._extract_text_content(self.processed_tree)
            
            if not original_text or not processed_text:
                return {'enabled': True, 'error': 'Insufficient content'}
            
            # Split into chunks for analysis
            original_texts = [chunk.strip() for chunk in original_text.split('.') if chunk.strip()]
            processed_texts = [chunk.strip() for chunk in processed_text.split('.') if chunk.strip()]
            
            # Analyze sequential topic flow
            original_flow = self._analyze_topic_flow(original_texts)
            processed_flow = self._analyze_topic_flow(processed_texts)
            
            # Detect drift patterns
            patterns = {
                'topic_introduction': self._detect_topic_introduction(original_flow, processed_flow),
                'topic_loss': self._detect_topic_loss(original_flow, processed_flow),
                'topic_reordering': self._detect_topic_reordering(original_flow, processed_flow),
                'coherence_breaks': self._detect_coherence_breaks(processed_flow)
            }
            
            return {
                'enabled': True,
                'patterns': patterns,
                'original_topic_count': len(original_flow),
                'processed_topic_count': len(processed_flow)
            }
            
        except Exception as e:
            log.error(f"Topic drift pattern detection failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def _analyze_topic_flow(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze the flow of topics through a document.
        
        Args:
            texts: List of text chunks or single text string
            
        Returns:
            List of topic flow segments
        """
        if not texts:
            return []
        
        # Handle case where texts is a single string
        if isinstance(texts, str):
            texts = [chunk.strip() for chunk in texts.split('.') if chunk.strip()]
        
        if not texts:
            return []
        
        embeddings = self.embedding_client.encode(texts)
        flow_segments = []
        
        # Group consecutive similar chunks into topic segments
        current_segment = {'start': 0, 'texts': [texts[0]], 'embeddings': [embeddings[0]]}
        similarity_threshold = self.semantic_config.get('topic_flow_threshold', 0.6)
        
        for i in range(1, len(texts)):
            # Calculate similarity with current segment centroid
            segment_centroid = np.mean(current_segment['embeddings'], axis=0)
            similarity = self.embedding_client.similarity(
                segment_centroid.reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            if similarity >= similarity_threshold:
                # Continue current segment
                current_segment['texts'].append(texts[i])
                current_segment['embeddings'].append(embeddings[i])
            else:
                # Start new segment
                current_segment['end'] = i - 1
                current_segment['centroid'] = segment_centroid
                flow_segments.append(current_segment)
                
                current_segment = {
                    'start': i,
                    'texts': [texts[i]],
                    'embeddings': [embeddings[i]]
                }
        
        # Add final segment
        current_segment['end'] = len(texts) - 1
        current_segment['centroid'] = np.mean(current_segment['embeddings'], axis=0)
        flow_segments.append(current_segment)
        
        return flow_segments
    
    def _detect_topic_introduction(self, original_flow: List[Dict], 
                                 processed_flow: List[Dict]) -> Dict[str, Any]:
        """Detect introduction of new topics in processed document."""
        if not original_flow or not processed_flow:
            return {'detected': False, 'new_topics': 0}
        
        new_topic_count = 0
        introduction_threshold = self.semantic_config.get('topic_introduction_threshold', 0.5)
        
        for proc_segment in processed_flow:
            # Check if this processed segment matches any original segment
            max_similarity = 0.0
            for orig_segment in original_flow:
                similarity = self.embedding_client.similarity(
                    proc_segment['centroid'].reshape(1, -1),
                    orig_segment['centroid'].reshape(1, -1)
                )[0][0]
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity < introduction_threshold:
                new_topic_count += 1
        
        return {
            'detected': new_topic_count > 0,
            'new_topics': new_topic_count,
            'introduction_rate': new_topic_count / len(processed_flow) if processed_flow else 0
        }
    
    def _detect_topic_loss(self, original_flow: List[Dict], 
                         processed_flow: List[Dict]) -> Dict[str, Any]:
        """Detect loss of topics from original document."""
        if not original_flow or not processed_flow:
            return {'detected': False, 'lost_topics': 0}
        
        lost_topic_count = 0
        loss_threshold = self.semantic_config.get('topic_loss_threshold', 0.5)
        
        for orig_segment in original_flow:
            # Check if this original segment is preserved in processed document
            max_similarity = 0.0
            for proc_segment in processed_flow:
                similarity = self.embedding_client.similarity(
                    orig_segment['centroid'].reshape(1, -1),
                    proc_segment['centroid'].reshape(1, -1)
                )[0][0]
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity < loss_threshold:
                lost_topic_count += 1
        
        return {
            'detected': lost_topic_count > 0,
            'lost_topics': lost_topic_count,
            'loss_rate': lost_topic_count / len(original_flow) if original_flow else 0
        }
    
    def _detect_topic_reordering(self, original_flow: List[Dict], 
                               processed_flow: List[Dict]) -> Dict[str, Any]:
        """Detect reordering of topics between documents."""
        if len(original_flow) < 2 or len(processed_flow) < 2:
            return {'detected': False, 'reordering_score': 0.0}
        
        # Create topic alignment mapping
        alignment_map = []
        for i, orig_segment in enumerate(original_flow):
            best_match_idx = -1
            best_similarity = 0.0
            
            for j, proc_segment in enumerate(processed_flow):
                similarity = self.embedding_client.similarity(
                    orig_segment['centroid'].reshape(1, -1),
                    proc_segment['centroid'].reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = j
            
            if best_similarity > 0.5:  # Only consider strong matches
                alignment_map.append((i, best_match_idx))
        
        # Calculate reordering score based on sequence disruption
        reordering_score = 0.0
        if len(alignment_map) > 1:
            disruptions = 0
            for i in range(1, len(alignment_map)):
                if alignment_map[i][1] < alignment_map[i-1][1]:  # Out of order
                    disruptions += 1
            reordering_score = disruptions / (len(alignment_map) - 1)
        
        return {
            'detected': reordering_score > 0.2,
            'reordering_score': reordering_score,
            'aligned_topics': len(alignment_map)
        }
    
    def _detect_coherence_breaks(self, flow: List[Dict]) -> Dict[str, Any]:
        """Detect breaks in topic coherence within the document."""
        if len(flow) < 2:
            return {'detected': False, 'coherence_breaks': 0}
        
        coherence_threshold = self.semantic_config.get('coherence_threshold', 0.6)
        breaks = 0
        
        for i in range(1, len(flow)):
            similarity = self.embedding_client.similarity(
                flow[i-1]['centroid'].reshape(1, -1),
                flow[i]['centroid'].reshape(1, -1)
            )[0][0]
            
            if similarity < coherence_threshold:
                breaks += 1
        
        return {
            'detected': breaks > len(flow) * 0.3,  # More than 30% transitions are breaks
            'coherence_breaks': breaks,
            'break_rate': breaks / (len(flow) - 1) if len(flow) > 1 else 0
        }

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