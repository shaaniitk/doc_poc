from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class SimilarityResult:
    """Result of similarity calculation."""
    similarity_score: float
    confidence: float
    method_used: str
    metadata: Dict[str, Any] = None

class SemanticMapper(ABC):
    """Abstract base class for semantic mapping operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        pass
    
    @abstractmethod
    async def find_similar_chunks(self, query: str, chunks: List[Any], 
                                 threshold: float = 0.5) -> List[Tuple[Any, float]]:
        """Find chunks similar to the query."""
        pass

class BasicSemanticMapper(SemanticMapper):
    """Basic implementation of semantic mapper using simple text similarity."""
    
    def __init__(self):
        super().__init__()
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using TF-IDF and cosine similarity."""
        try:
            # Preprocess texts
            tokens1 = self._preprocess_text(text1)
            tokens2 = self._preprocess_text(text2)
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # Create vocabulary
            vocab = set(tokens1 + tokens2)
            
            # Calculate TF-IDF vectors
            vec1 = self._calculate_tfidf_vector(tokens1, vocab)
            vec2 = self._calculate_tfidf_vector(tokens2, vocab)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(vec1, vec2)
            
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    async def find_similar_chunks(self, query: str, chunks: List[Any], 
                                 threshold: float = 0.5) -> List[Tuple[Any, float]]:
        """Find chunks similar to the query."""
        similar_chunks = []
        
        for chunk in chunks:
            # Extract content from chunk (assuming it has a 'content' attribute)
            chunk_content = getattr(chunk, 'content', str(chunk))
            similarity = await self.calculate_similarity(query, chunk_content)
            
            if similarity >= threshold:
                similar_chunks.append((chunk, similarity))
        
        # Sort by similarity score (descending)
        similar_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return similar_chunks
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing and removing stop words."""
        # Simple tokenization (can be enhanced with proper NLP libraries)
        import re
        
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return filtered_words
    
    def _calculate_tfidf_vector(self, tokens: List[str], vocab: set) -> np.ndarray:
        """Calculate TF-IDF vector for tokens."""
        # Calculate term frequency
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        # Normalize by document length
        doc_length = len(tokens)
        for token in tf:
            tf[token] = tf[token] / doc_length
        
        # Create vector (simplified IDF - just using TF for now)
        vector = np.zeros(len(vocab))
        vocab_list = list(vocab)
        
        for i, word in enumerate(vocab_list):
            if word in tf:
                vector[i] = tf[word]
        
        return vector
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class EmbeddingSemanticMapper(SemanticMapper):
    """Semantic mapper using embeddings (placeholder for future implementation)."""
    
    def __init__(self, embedding_model=None):
        super().__init__()
        self.embedding_model = embedding_model
        self.logger.warning("EmbeddingSemanticMapper is a placeholder - using basic similarity")
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using embeddings (fallback to basic for now)."""
        # Fallback to basic similarity for now
        basic_mapper = BasicSemanticMapper()
        return await basic_mapper.calculate_similarity(text1, text2)
    
    async def find_similar_chunks(self, query: str, chunks: List[Any], 
                                 threshold: float = 0.5) -> List[Tuple[Any, float]]:
        """Find similar chunks using embeddings (fallback to basic for now)."""
        # Fallback to basic similarity for now
        basic_mapper = BasicSemanticMapper()
        return await basic_mapper.find_similar_chunks(query, chunks, threshold)

def create_semantic_mapper(mapper_type: str = "basic", **kwargs) -> SemanticMapper:
    """Factory function to create semantic mapper instances."""
    if mapper_type == "basic":
        return BasicSemanticMapper()
    elif mapper_type == "embedding":
        return EmbeddingSemanticMapper(**kwargs)
    else:
        logger.warning(f"Unknown mapper type {mapper_type}, using basic")
        return BasicSemanticMapper()

# Convenience function for backward compatibility
async def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using basic mapper."""
    mapper = BasicSemanticMapper()
    return await mapper.calculate_similarity(text1, text2)