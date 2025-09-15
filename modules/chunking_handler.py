"""Chunking handler for document processing"""
from typing import List, Dict, Any
from .chunker import AdaptiveChunker
from .error_handler import ChunkingError, validate_chunk

class ChunkingHandler:
    """Handler for document chunking operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.chunker = AdaptiveChunker()
    
    def process_chunks(self, content: str, **kwargs) -> List[Dict[str, Any]]:
        """Process content into chunks"""
        try:
            # Use the adaptive chunker to create chunks
            chunks = self.chunker.chunk_content(content, **kwargs)
            
            # Validate each chunk
            validated_chunks = []
            for chunk in chunks:
                if validate_chunk(chunk):
                    validated_chunks.append(chunk)
                else:
                    raise ChunkingError(f"Invalid chunk structure: {chunk}")
            
            return validated_chunks
            
        except Exception as e:
            raise ChunkingError(f"Chunking failed: {str(e)}")
    
    def get_chunk_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a chunk"""
        return {
            'type': chunk.get('type', 'unknown'),
            'length': len(str(chunk.get('content', ''))),
            'parent_section': chunk.get('parent_section', 'unknown')
        }