import os
import openai
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import SEMANTIC_MAPPING_CONFIG
from .error_handler import ProcessingError

class UnifiedEmbeddingClient:
    """
    A unified client for handling different embedding providers.
    Supports both SentenceTransformer models and OpenAI embeddings.
    """
    
    def __init__(self, config=None):
        self.config = config or SEMANTIC_MAPPING_CONFIG
        self.provider = self.config.get('provider', 'sentence_transformer')
        self.model_name = self.config['model']
        self.model = None
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