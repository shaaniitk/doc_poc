import os
import openai
import numpy as np
import logging
import time
from typing import List, Union, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import cohere
from config import SEMANTIC_MAPPING_CONFIG, LLM_CONFIG
from .error_handler import (
    ProcessingError, EmbeddingError,
    EmbeddingAPIError, EmbeddingModelError, EmbeddingFallbackError,
    robust_embedding_call, EmbeddingRateLimiter, EmbeddingFallbackManager
)

class UnifiedEmbeddingClient:
    """
    A unified client for handling different embedding providers with fallback support and robust error handling.
    Supports both SentenceTransformer models and OpenAI embeddings.
    """
    
    def __init__(self, config=None):
        self.config = config or SEMANTIC_MAPPING_CONFIG
        self.provider = self.config.get('provider', 'sentence_transformer')
        self.model_name = self.config['model']
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize error handling components
        self.rate_limiter = EmbeddingRateLimiter()
        self.fallback_manager = EmbeddingFallbackManager()
        
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate embedding model based on provider."""
        try:
            if self.provider == 'openai':
                # Verify OpenAI API key is set
                if not os.getenv('OPENAI_API_KEY'):
                    raise ProcessingError("OPENAI_API_KEY environment variable not set")
                openai.api_key = os.getenv('OPENAI_API_KEY')
                self.model = None  # OpenAI doesn't need a local model instance
            else:
                # Default to SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise ProcessingError(f"Failed to load embedding model '{self.model_name}': {e}")
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode texts into embeddings using the configured provider.
        
        Args:
            texts: Single text string or list of text strings
            **kwargs: Additional arguments (batch_size, etc.)
        
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            if self.provider == 'openai':
                return self._encode_openai(texts, **kwargs)
            else:
                return self._encode_sentence_transformer(texts, **kwargs)
        except Exception as e:
            raise ProcessingError(f"Failed to encode texts: {e}")
    
    def _encode_openai(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts using OpenAI's embedding API."""
        try:
            # Use the new OpenAI client format
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except Exception as e:
            raise ProcessingError(f"OpenAI embedding API error: {e}")
    
    def _encode_sentence_transformer(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts using SentenceTransformer."""
        return self.model.encode(texts, **kwargs)
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
        
        Returns:
            Similarity matrix
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings1, embeddings2)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension size
        """
        if self.provider == 'openai':
            # OpenAI text-embedding-3-large has 3072 dimensions
            if self.model_name == 'text-embedding-3-large':
                return 3072
            elif self.model_name == 'text-embedding-3-small':
                return 1536
            elif self.model_name == 'text-embedding-ada-002':
                return 1536
            else:
                # Default fallback - could also make an API call to determine
                return 1536
        else:
            return self.model.get_sentence_embedding_dimension()


class LangChainEmbeddingWrapper:
    """
    A LangChain-compatible wrapper for UnifiedEmbeddingClient.
    This allows our embedding client to work with LangChain's vector stores.
    """
    
    def __init__(self, embedding_client: UnifiedEmbeddingClient):
        self.embedding_client = embedding_client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.embedding_client.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        embedding = self.embedding_client.encode([text])
        return embedding[0].tolist()
    
    def __call__(self, text: str) -> List[float]:
        """Make the wrapper callable - delegates to embed_query."""
        return self.embed_query(text)

    @robust_embedding_call(max_retries=3, backoff_delay=2.0)
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts with robust error handling.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        try:
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            embeddings = self.embedding_client.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {e}")
            # Try fallback if available
            fallback_result = self.fallback_manager.try_fallback('get_embeddings', texts=texts)
            if fallback_result is not None:
                return fallback_result
            raise EmbeddingAPIError(f"Failed to get embeddings: {e}")
    
    @robust_embedding_call(max_retries=2, backoff_delay=1.5)
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Get embeddings for texts in batches with robust error handling.
        
        Args:
            texts: List of text strings to embed
            batch_size: Size of each batch for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        all_embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                self.rate_limiter.wait_if_needed()
                
                batch_embeddings = self.get_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Error in batch embedding processing: {e}")
            fallback_result = self.fallback_manager.try_fallback('get_embeddings_batch', texts=texts, batch_size=batch_size)
            if fallback_result is not None:
                return fallback_result
            raise EmbeddingAPIError(f"Failed to process embeddings in batches: {e}")
    
    def calculate_cohesion_scores(self, embeddings: List[List[float]], window_size: int = 3) -> List[float]:
        """
        Calculate cohesion scores between consecutive embeddings.
        
        Args:
            embeddings: List of embedding vectors
            window_size: Size of the sliding window for cohesion calculation
            
        Returns:
            List of cohesion scores
        """
        if len(embeddings) < 2:
            return [1.0] * len(embeddings)
            
        try:
            embeddings_array = np.array(embeddings)
            cohesion_scores = []
            
            for i in range(len(embeddings)):
                if i == 0:
                    cohesion_scores.append(1.0)
                    continue
                    
                # Calculate similarity with previous embeddings in window
                start_idx = max(0, i - window_size)
                window_embeddings = embeddings_array[start_idx:i]
                current_embedding = embeddings_array[i]
                
                # Calculate cosine similarities
                similarities = []
                for prev_embedding in window_embeddings:
                    similarity = np.dot(current_embedding, prev_embedding) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(prev_embedding)
                    )
                    similarities.append(similarity)
                
                # Average similarity as cohesion score
                cohesion_score = np.mean(similarities) if similarities else 0.0
                cohesion_scores.append(max(0.0, cohesion_score))
                
            return cohesion_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating cohesion scores: {e}")
            # Return default scores on error
            return [0.5] * len(embeddings)
    
    def detect_topic_boundaries(self, embeddings: List[List[float]], threshold: float = 0.3) -> List[int]:
        """
        Detect topic boundaries based on embedding similarity drops.
        
        Args:
            embeddings: List of embedding vectors
            threshold: Similarity threshold for boundary detection
            
        Returns:
            List of indices where topic boundaries are detected
        """
        if len(embeddings) < 2:
            return []
            
        try:
            cohesion_scores = self.calculate_cohesion_scores(embeddings)
            boundaries = []
            
            for i, score in enumerate(cohesion_scores[1:], 1):
                if score < threshold:
                    boundaries.append(i)
                    
            return boundaries
            
        except Exception as e:
            self.logger.error(f"Error detecting topic boundaries: {e}")
            return []