#!/usr/bin/env python3
"""
Debug script to test semantic similarity calculations.
"""

import asyncio
from core.semantic_mapper import BasicSemanticMapper

async def test_similarity():
    """Test similarity calculations with sample texts."""
    mapper = BasicSemanticMapper()
    
    # Test with section titles and content
    section_title = "introduction"
    chunk_content = "Bitcoin is a peer-to-peer electronic cash system that allows online payments to be sent directly from one party to another without going through a financial institution."
    
    similarity = await mapper.calculate_similarity(section_title, chunk_content)
    print(f"Similarity between '{section_title}' and chunk: {similarity:.4f}")
    
    # Test with more similar texts
    text1 = "Bitcoin electronic cash system"
    text2 = "Bitcoin is an electronic cash system for peer-to-peer payments"
    
    similarity2 = await mapper.calculate_similarity(text1, text2)
    print(f"Similarity between similar texts: {similarity2:.4f}")
    
    # Test with identical words
    text3 = "bitcoin payment system"
    text4 = "bitcoin payment network system"
    
    similarity3 = await mapper.calculate_similarity(text3, text4)
    print(f"Similarity with overlapping words: {similarity3:.4f}")
    
    # Test preprocessing
    tokens1 = mapper._preprocess_text(section_title)
    tokens2 = mapper._preprocess_text(chunk_content)
    
    print(f"\nPreprocessed tokens:")
    print(f"Section title: {tokens1}")
    print(f"Chunk content: {tokens2[:10]}...")  # First 10 tokens
    
    # Test with lower threshold
    chunks = [type('Chunk', (), {'content': chunk_content})()]
    similar_chunks_05 = await mapper.find_similar_chunks(section_title, chunks, threshold=0.5)
    similar_chunks_01 = await mapper.find_similar_chunks(section_title, chunks, threshold=0.1)
    similar_chunks_001 = await mapper.find_similar_chunks(section_title, chunks, threshold=0.01)
    
    print(f"\nMatches with threshold 0.5: {len(similar_chunks_05)}")
    print(f"Matches with threshold 0.1: {len(similar_chunks_01)}")
    print(f"Matches with threshold 0.01: {len(similar_chunks_001)}")
    
    if similar_chunks_001:
        print(f"Best match score: {similar_chunks_001[0][1]:.4f}")

if __name__ == "__main__":
    asyncio.run(test_similarity())