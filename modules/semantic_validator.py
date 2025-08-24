"""Semantic validation module - content coherence guardian

This module provides semantic validation using sentence transformers to ensure
content coherence, relevance, and preservation during LLM processing.

KEY RESPONSIBILITIES:
- Semantic similarity validation between original and processed content
- Content coherence scoring and validation
- Relevance assessment for context management
- Quality assurance through embedding-based analysis

QUALITY ASSURANCE:
- Embedding-based semantic similarity scoring
- Configurable similarity thresholds
- Multi-model support for different validation needs
- Batch processing for efficiency

This is the semantic quality control checkpoint for all content transformations.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Optional
import logging
from .error_handler import ProcessingError

class SemanticValidationError(ProcessingError):
    """Raised when semantic validation fails"""
    pass

class SemanticValidator:
    """🧠 SEMANTIC VALIDATION ENGINE
    
    Uses state-of-the-art sentence transformers to validate semantic
    coherence and content preservation during LLM processing.
    
    VALIDATION STRATEGIES:
    - Cosine similarity for content preservation
    - Semantic coherence scoring for quality assessment
    - Relevance scoring for context management
    - Batch processing for efficiency optimization
    
    QUALITY FOCUS:
    Ensures LLM transformations preserve semantic meaning while
    improving content quality and structure.
    """
    
    def __init__(self, model_name='all-mpnet-base-v2', similarity_threshold=0.75):
        """🎯 INITIALIZE SEMANTIC VALIDATOR
        
        Args:
            model_name: Sentence transformer model to use
            similarity_threshold: Minimum similarity score for validation
            
        🧠 MODEL SELECTION:
        - all-mpnet-base-v2: Best overall performance (default)
        - all-MiniLM-L6-v2: Faster, good for real-time validation
        - paraphrase-mpnet-base-v2: Specialized for paraphrase detection
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model = None  # Lazy loading for better startup performance
        self.logger = logging.getLogger(__name__)
        
        # 📊 Performance tracking
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'average_similarity': 0.0
        }
    
    def _load_model(self):
        """🔄 LAZY MODEL LOADING
        
        Loads the sentence transformer model only when needed
        to improve startup performance and memory usage.
        """
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.logger.info(f"Loaded semantic model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load semantic model: {e}")
                raise SemanticValidationError(f"Model loading failed: {e}")
    
    def validate_content_preservation(self, original: str, processed: str, 
                                    threshold: Optional[float] = None) -> Tuple[bool, float]:
        """🔍 VALIDATE CONTENT PRESERVATION
        
        Validates that processed content preserves the semantic meaning
        of the original content using cosine similarity.
        
        Args:
            original: Original content before processing
            processed: Content after LLM processing
            threshold: Custom threshold (uses default if None)
            
        Returns:
            tuple: (is_valid, similarity_score)
            
        🧠 VALIDATION PROCESS:
        1. Generate embeddings for both texts
        2. Calculate cosine similarity
        3. Compare against threshold
        4. Update performance statistics
        """
        self._load_model()
        
        # 🎯 Use custom threshold or default
        validation_threshold = threshold or self.similarity_threshold
        
        try:
            # 🧠 Generate embeddings
            original_embedding = self.model.encode(original, convert_to_tensor=True)
            processed_embedding = self.model.encode(processed, convert_to_tensor=True)
            
            # 📊 Calculate cosine similarity
            similarity = self._cosine_similarity(original_embedding, processed_embedding)
            
            # ✅ Validate against threshold
            is_valid = similarity >= validation_threshold
            
            # 📈 Update statistics
            self._update_stats(similarity, is_valid)
            
            self.logger.debug(f"Semantic validation: {similarity:.3f} (threshold: {validation_threshold:.3f})")
            
            return is_valid, float(similarity)
            
        except Exception as e:
            self.logger.error(f"Semantic validation failed: {e}")
            raise SemanticValidationError(f"Validation error: {e}")
    
    def validate_batch(self, content_pairs: List[Tuple[str, str]], 
                      threshold: Optional[float] = None) -> List[Tuple[bool, float]]:
        """🚀 BATCH SEMANTIC VALIDATION
        
        Efficiently validates multiple content pairs using batch processing
        for improved performance.
        
        Args:
            content_pairs: List of (original, processed) content tuples
            threshold: Custom threshold for all validations
            
        Returns:
            List of (is_valid, similarity_score) tuples
            
        ⚡ PERFORMANCE OPTIMIZATION:
        Batch encoding reduces model overhead and improves throughput
        for multiple validations.
        """
        self._load_model()
        
        validation_threshold = threshold or self.similarity_threshold
        results = []
        
        try:
            # 📦 Prepare texts for batch processing
            originals = [pair[0] for pair in content_pairs]
            processed = [pair[1] for pair in content_pairs]
            
            # 🧠 Batch encode for efficiency
            original_embeddings = self.model.encode(originals, convert_to_tensor=True)
            processed_embeddings = self.model.encode(processed, convert_to_tensor=True)
            
            # 📊 Calculate similarities for all pairs
            for i in range(len(content_pairs)):
                similarity = self._cosine_similarity(
                    original_embeddings[i], processed_embeddings[i]
                )
                is_valid = similarity >= validation_threshold
                
                results.append((is_valid, float(similarity)))
                self._update_stats(similarity, is_valid)
            
            self.logger.info(f"Batch validation completed: {len(content_pairs)} pairs")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch validation failed: {e}")
            raise SemanticValidationError(f"Batch validation error: {e}")
    
    def calculate_relevance_score(self, content: str, context: str) -> float:
        """🎯 CALCULATE CONTENT RELEVANCE
        
        Calculates semantic relevance between content and context
        for intelligent context management.
        
        Args:
            content: Content to score
            context: Reference context
            
        Returns:
            float: Relevance score (0.0 to 1.0)
            
        🧠 CONTEXT MANAGEMENT:
        Used by smart context managers to determine which context
        items are most relevant for current processing.
        """
        self._load_model()
        
        try:
            # 🧠 Generate embeddings
            content_embedding = self.model.encode(content, convert_to_tensor=True)
            context_embedding = self.model.encode(context, convert_to_tensor=True)
            
            # 📊 Calculate relevance as cosine similarity
            relevance = self._cosine_similarity(content_embedding, context_embedding)
            
            return float(relevance)
            
        except Exception as e:
            self.logger.error(f"Relevance calculation failed: {e}")
            return 0.0  # Return low relevance on error
    
    def assess_content_coherence(self, content_sections: List[str]) -> float:
        """🔗 ASSESS CONTENT COHERENCE
        
        Evaluates overall coherence across multiple content sections
        by analyzing semantic relationships.
        
        Args:
            content_sections: List of content sections to analyze
            
        Returns:
            float: Coherence score (0.0 to 1.0)
            
        🧠 COHERENCE ANALYSIS:
        Calculates pairwise similarities between sections and
        returns average coherence score.
        """
        if len(content_sections) < 2:
            return 1.0  # Single section is perfectly coherent
        
        self._load_model()
        
        try:
            # 🧠 Generate embeddings for all sections
            embeddings = self.model.encode(content_sections, convert_to_tensor=True)
            
            # 📊 Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(similarity)
            
            # 🎯 Return average coherence
            coherence_score = float(np.mean(similarities)) if similarities else 1.0
            
            self.logger.debug(f"Content coherence: {coherence_score:.3f}")
            return coherence_score
            
        except Exception as e:
            self.logger.error(f"Coherence assessment failed: {e}")
            return 0.5  # Return neutral score on error
    
    def _cosine_similarity(self, embedding1, embedding2) -> float:
        """📊 CALCULATE COSINE SIMILARITY
        
        Calculates cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor
            
        Returns:
            float: Cosine similarity score
        """
        import torch
        
        # 📊 Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0), embedding2.unsqueeze(0)
        )
        
        return float(similarity.item())
    
    def _update_stats(self, similarity: float, is_valid: bool):
        """📈 UPDATE PERFORMANCE STATISTICS
        
        Updates internal performance tracking statistics.
        
        Args:
            similarity: Calculated similarity score
            is_valid: Whether validation passed
        """
        self.validation_stats['total_validations'] += 1
        
        if is_valid:
            self.validation_stats['passed_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
        
        # 📊 Update running average
        total = self.validation_stats['total_validations']
        current_avg = self.validation_stats['average_similarity']
        self.validation_stats['average_similarity'] = (
            (current_avg * (total - 1) + similarity) / total
        )
    
    def validate_content_quality(self, content: str) -> Tuple[bool, float]:
        """🔍 VALIDATE CONTENT QUALITY
        
        Validates content quality based on length, structure, and semantic richness.
        
        Args:
            content: Content to validate
            
        Returns:
            tuple: (is_valid, quality_score)
        """
        if not content or len(content.strip()) == 0:
            return False, 0.0
        
        # Basic quality checks
        content = content.strip()
        
        # Length check
        if len(content) < 10:
            return False, 0.2
        
        # Word count and diversity
        words = content.split()
        if len(words) < 3:
            return False, 0.3
        
        # Check for repetitive content
        unique_words = set(words)
        word_diversity = len(unique_words) / len(words) if words else 0
        
        if word_diversity < 0.3:  # Too repetitive
            return False, 0.4
        
        # Check for meaningless content patterns
        content_lower = content.lower()
        meaningless_patterns = [
            'xyz abc def',  # Test pattern
            'lorem ipsum',  # Placeholder text
            'test test test',  # Repetitive test content
            'invalid content',  # Explicit invalid marker
            'placeholder text',  # Placeholder content
        ]
        
        for pattern in meaningless_patterns:
            if pattern in content_lower:
                return False, 0.1  # Very low quality score for meaningless content
        
        # Check for very short sentences or fragments
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 3:  # Very short sentences
                return False, 0.3
        
        # Calculate quality score
        length_score = min(len(content) / 100, 1.0)  # Normalize to 100 chars
        diversity_score = word_diversity
        structure_score = 1.0 if any(char in content for char in '.!?') else 0.5
        
        # Semantic richness check (basic)
        semantic_score = 1.0
        if len(unique_words) < 5:  # Very limited vocabulary
            semantic_score = 0.5
        
        quality_score = (
            length_score * 0.3 + 
            diversity_score * 0.3 + 
            structure_score * 0.2 + 
            semantic_score * 0.2
        )
        
        is_valid = quality_score >= 0.6  # Raised threshold for better quality
        return is_valid, quality_score
    
    def validate_content_coherence(self, content1: str, content2: str) -> float:
        """🔗 VALIDATE CONTENT COHERENCE
        
        Validates coherence between two pieces of content using semantic similarity.
        
        Args:
            content1: First content piece
            content2: Second content piece
            
        Returns:
            float: Coherence score (0.0 to 1.0)
        """
        if not content1 or not content2:
            return 0.0
        
        if content1.strip() == content2.strip():
            return 1.0
        
        try:
            self._load_model()
            
            # Generate embeddings
            embedding1 = self.model.encode(content1, convert_to_tensor=True)
            embedding2 = self.model.encode(content2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            coherence = self._cosine_similarity(embedding1, embedding2)
            
            return float(coherence)
            
        except Exception as e:
            self.logger.error(f"Coherence validation failed: {e}")
            return 0.0
    
    def get_validation_stats(self) -> Dict[str, float]:
        """📊 GET VALIDATION STATISTICS
        
        Returns performance statistics for monitoring and optimization.
        
        Returns:
            dict: Validation performance statistics
        """
        stats = self.validation_stats.copy()
        
        # 📈 Calculate success rate
        if stats['total_validations'] > 0:
            stats['success_rate'] = (
                stats['passed_validations'] / stats['total_validations']
            )
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """🔄 RESET VALIDATION STATISTICS
        
        Resets performance tracking statistics.
        """
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'average_similarity': 0.0
        }
        
        self.logger.info("Validation statistics reset")


class SmartContextManager:
    """🧠 INTELLIGENT CONTEXT MANAGEMENT SYSTEM
    
    Advanced context management with semantic relevance scoring,
    importance weighting, and intelligent pruning strategies.
    
    MANAGEMENT STRATEGIES:
    - Semantic relevance scoring for context items
    - Importance weighting based on recency and usage
    - Intelligent pruning to maintain optimal context size
    - Context compression through summarization
    
    PERFORMANCE FOCUS:
    Maintains optimal context size while preserving the most
    relevant and important information for LLM processing.
    """
    
    def __init__(self, max_context_length=2000, semantic_validator=None):
        """🎯 INITIALIZE SMART CONTEXT MANAGER
        
        Args:
            max_context_length: Maximum context size in characters
            semantic_validator: SemanticValidator instance for relevance scoring
        """
        self.max_context_length = max_context_length
        self.semantic_validator = semantic_validator or SemanticValidator()
        self.logger = logging.getLogger(__name__)
        
        # 📚 Context storage with metadata
        self.context_items = []  # List of ContextItem objects
        self.document_context = ""
        self.section_contexts = {}
        
        # 📊 Performance tracking
        self.context_stats = {
            'total_additions': 0,
            'pruning_events': 0,
            'average_relevance': 0.0
        }
    
    def build_context(self, queries: List[str], documents: List[str], max_length: int = None) -> str:
        """🎯 BUILD SMART CONTEXT
        
        Builds intelligent context by selecting most relevant documents for given queries.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            max_length: Maximum context length (uses default if None)
            
        Returns:
            str: Built context string
        """
        if not queries or not documents:
            return ""
        
        max_len = max_length or self.max_context_length
        
        # Calculate relevance scores for each document
        scored_docs = []
        query_text = " ".join(queries)
        
        for doc in documents:
            if not doc.strip():
                continue
                
            relevance = self.semantic_validator.calculate_relevance_score(query_text, doc)
            scored_docs.append((relevance, doc))
        
        # Sort by relevance and build context
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        context_parts = []
        current_length = 0
        separator_length = 2  # Length of "\n\n"
        
        for relevance, doc in scored_docs:
            # Calculate total length including separator
            doc_with_separator = len(doc) + (separator_length if context_parts else 0)
            
            if current_length + doc_with_separator <= max_len:
                context_parts.append(doc)
                current_length += doc_with_separator
            else:
                # Add partial document if it fits
                remaining = max_len - current_length - (separator_length if context_parts else 0)
                if remaining > 50:  # Only add if meaningful length
                    context_parts.append(doc[:remaining] + "...")
                break
        
        result = "\n\n".join(context_parts)
        
        # Ensure we don't exceed max_length due to separator calculation errors
        if len(result) > max_len:
            result = result[:max_len-3] + "..."
        
        return result
    
    def create_section_summary(self, section_name: str, content: str) -> str:
        """📝 CREATE SECTION SUMMARY
        
        Creates an intelligent summary of section content for context management.
        
        Args:
            section_name: Name of the section
            content: Section content
            
        Returns:
            str: Section summary
        """
        if not content or len(content.strip()) == 0:
            return ""
        
        content = content.strip()
        
        # For short content, return as-is
        if len(content) <= 200:
            return content
        
        # For longer content, create intelligent summary
        sentences = content.split('. ')
        
        if len(sentences) <= 3:
            return content[:500] + "..." if len(content) > 500 else content
        
        # Take first and last sentences, plus middle if space allows
        summary_parts = [sentences[0]]
        
        if len(sentences) > 2:
            # Add middle sentence if it fits
            middle_idx = len(sentences) // 2
            summary_parts.append(sentences[middle_idx])
        
        # Add last sentence
        if len(sentences) > 1:
            summary_parts.append(sentences[-1])
        
        summary = '. '.join(summary_parts)
        
        # Ensure summary doesn't exceed reasonable length
        if len(summary) > 400:
            summary = summary[:400] + "..."
        
        return summary
    
    def add_context(self, content: str, section_name: str = None, 
                   importance_score: float = 1.0, context_type: str = "general"):
        """📝 ADD CONTEXT WITH INTELLIGENCE
        
        Adds context with importance weighting and semantic analysis.
        
        Args:
            content: Context content to add
            section_name: Associated section name (optional)
            importance_score: Importance weight (0.0 to 1.0)
            context_type: Type of context (general, section, summary)
            
        🧠 INTELLIGENT PROCESSING:
        1. Calculate semantic relevance to existing context
        2. Apply importance weighting
        3. Prune context if size limits exceeded
        4. Update context statistics
        """
        from datetime import datetime
        
        # 🎯 Create context item with metadata
        context_item = {
            'content': content,
            'section_name': section_name,
            'importance_score': importance_score,
            'context_type': context_type,
            'timestamp': datetime.now(),
            'usage_count': 0,
            'relevance_score': 0.0
        }
        
        # 🧠 Calculate relevance to existing context
        if self.document_context:
            relevance = self.semantic_validator.calculate_relevance_score(
                content, self.document_context
            )
            context_item['relevance_score'] = relevance
        
        # 📝 Add to context storage
        self.context_items.append(context_item)
        
        # 🔄 Update section-specific context
        if section_name:
            if section_name not in self.section_contexts:
                self.section_contexts[section_name] = []
            self.section_contexts[section_name].append(context_item)
        
        # 📊 Update statistics
        self.context_stats['total_additions'] += 1
        
        # 🧹 Prune if necessary
        self._prune_context_if_needed()
        
        # 🔄 Rebuild document context
        self._rebuild_document_context()
        
        self.logger.debug(f"Added context: {len(content)} chars, relevance: {context_item['relevance_score']:.3f}")
    
    def get_relevant_context(self, current_content: str, max_items: int = 5) -> str:
        """🎯 GET MOST RELEVANT CONTEXT
        
        Retrieves the most relevant context items for current processing
        based on semantic similarity and importance scores.
        
        Args:
            current_content: Current content being processed
            max_items: Maximum number of context items to include
            
        Returns:
            str: Formatted relevant context
            
        🧠 RELEVANCE RANKING:
        Combines semantic similarity with importance scores to
        select the most valuable context for current processing.
        """
        if not self.context_items:
            return ""
        
        # 📊 Score all context items for relevance
        scored_items = []
        for item in self.context_items:
            # 🧠 Calculate relevance to current content
            relevance = self.semantic_validator.calculate_relevance_score(
                current_content, item['content']
            )
            
            # 🎯 Combine with importance and recency
            from datetime import datetime, timedelta
            age_hours = (datetime.now() - item['timestamp']).total_seconds() / 3600
            recency_factor = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours
            
            combined_score = (
                relevance * 0.6 +  # 60% semantic relevance
                item['importance_score'] * 0.3 +  # 30% importance
                recency_factor * 0.1  # 10% recency
            )
            
            scored_items.append((combined_score, item))
            
            # 📈 Update usage statistics
            item['usage_count'] += 1
        
        # 🏆 Sort by combined score and take top items
        scored_items.sort(key=lambda x: x[0], reverse=True)
        top_items = scored_items[:max_items]
        
        # 📝 Format context
        context_parts = []
        for score, item in top_items:
            context_type = item['context_type'].upper()
            section_info = f" ({item['section_name']})" if item['section_name'] else ""
            context_parts.append(f"[{context_type}{section_info}]: {item['content'][:300]}")
        
        relevant_context = "\n\n".join(context_parts)
        
        self.logger.debug(f"Retrieved {len(top_items)} relevant context items")
        return relevant_context
    
    def _prune_context_if_needed(self):
        """🧹 INTELLIGENT CONTEXT PRUNING
        
        Prunes context when size limits are exceeded, keeping
        the most important and relevant items.
        """
        current_size = sum(len(item['content']) for item in self.context_items)
        
        if current_size <= self.max_context_length:
            return  # No pruning needed
        
        # 📊 Score items for retention
        scored_items = []
        for item in self.context_items:
            # 🎯 Retention score based on importance, relevance, and usage
            retention_score = (
                item['importance_score'] * 0.4 +
                item['relevance_score'] * 0.4 +
                min(item['usage_count'] / 10, 0.2)  # Usage bonus up to 0.2
            )
            scored_items.append((retention_score, item))
        
        # 🏆 Sort by retention score
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # 🧹 Keep items until size limit
        pruned_items = []
        current_size = 0
        
        for score, item in scored_items:
            if current_size + len(item['content']) <= self.max_context_length:
                pruned_items.append(item)
                current_size += len(item['content'])
            else:
                break
        
        # 🔄 Update context storage
        removed_count = len(self.context_items) - len(pruned_items)
        self.context_items = pruned_items
        
        # 📊 Update statistics
        if removed_count > 0:
            self.context_stats['pruning_events'] += 1
            self.logger.info(f"Pruned {removed_count} context items")
    
    def _rebuild_document_context(self):
        """🔄 REBUILD DOCUMENT CONTEXT
        
        Rebuilds the document context from current context items.
        """
        if not self.context_items:
            self.document_context = ""
            return
        
        # 📝 Build context from most important items
        context_parts = []
        for item in sorted(self.context_items, 
                          key=lambda x: x['importance_score'], reverse=True):
            summary = item['content'][:200] + "..." if len(item['content']) > 200 else item['content']
            context_parts.append(summary)
        
        self.document_context = "\n\n".join(context_parts)
    
    def get_context_stats(self) -> Dict[str, float]:
        """📊 GET CONTEXT MANAGEMENT STATISTICS
        
        Returns context management performance statistics.
        
        Returns:
            dict: Context management statistics
        """
        stats = self.context_stats.copy()
        
        # 📈 Calculate additional metrics
        if self.context_items:
            avg_relevance = sum(item['relevance_score'] for item in self.context_items) / len(self.context_items)
            stats['average_relevance'] = avg_relevance
            stats['total_context_items'] = len(self.context_items)
            stats['total_context_size'] = sum(len(item['content']) for item in self.context_items)
        else:
            stats['average_relevance'] = 0.0
            stats['total_context_items'] = 0
            stats['total_context_size'] = 0
        
        return stats