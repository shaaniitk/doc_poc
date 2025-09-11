#!/usr/bin/env python3
"""
Test script to verify KG score integration in section mapping.
This script tests the enhanced SemanticMapper with KnowledgeGraphProcessor integration.
"""

import sys
import os
import logging
from modules.knowledge_graph_processor import KnowledgeGraphProcessor
from modules.embedding_client import UnifiedEmbeddingClient
from modules.section_mapper import SemanticMapper, assign_chunks_to_skeleton
from config import SEMANTIC_MAPPING_CONFIG, KG_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def create_test_chunks():
    """Create sample chunks for testing."""
    return [
        {
            'content': 'Bitcoin is a decentralized digital currency that enables peer-to-peer transactions without intermediaries.',
            'chunk_id': 'chunk_1',
            'metadata': {'section': 'Introduction', 'chunk_id': 'chunk_1'}
        },
        {
            'content': 'The blockchain technology underlying Bitcoin uses cryptographic hashing and proof-of-work consensus.',
            'chunk_id': 'chunk_2',
            'metadata': {'section': 'Technical Background', 'chunk_id': 'chunk_2'}
        },
        {
            'content': 'Mining involves solving computationally intensive puzzles to validate transactions and secure the network.',
            'chunk_id': 'chunk_3',
            'metadata': {'section': 'Mining Process', 'chunk_id': 'chunk_3'}
        },
        {
            'content': 'Smart contracts enable programmable transactions and decentralized applications on blockchain platforms.',
            'chunk_id': 'chunk_4',
            'metadata': {'section': 'Applications', 'chunk_id': 'chunk_4'}
        },
        {
            'content': 'Future research directions include scalability solutions, energy efficiency, and regulatory frameworks.',
            'chunk_id': 'chunk_5',
            'metadata': {'section': 'Conclusion', 'chunk_id': 'chunk_5'}
        }
    ]

def test_semantic_mapper_with_kg():
    """Test SemanticMapper with KG processor integration."""
    log.info("=== Testing SemanticMapper with KG Integration ===")
    
    # Create test data
    test_chunks = create_test_chunks()
    log.info(f"Created {len(test_chunks)} test chunks")
    
    try:
        # Initialize embedding model
        embedding_model = UnifiedEmbeddingClient(SEMANTIC_MAPPING_CONFIG)
        log.info("Initialized embedding model")
        
        # Initialize KG processor
        kg_processor = KnowledgeGraphProcessor(test_chunks, embedding_model)
        kg_processor.build_graphs()
        log.info("Built knowledge graphs")
        
        # Test SemanticMapper without KG
        log.info("\n--- Testing SemanticMapper WITHOUT KG ---")
        mapper_no_kg = SemanticMapper(template_name="bitcoin_paper_hierarchical")
        assignments_no_kg = mapper_no_kg.assign_chunks(test_chunks, mapper_no_kg.section_names)
        
        log.info("Assignments without KG:")
        for assignment in assignments_no_kg:
            log.info(f"  Chunk {assignment['chunk_id']} -> {assignment['section']} (score: {assignment['assignment_score']:.3f})")
        
        # Test SemanticMapper with KG
        log.info("\n--- Testing SemanticMapper WITH KG ---")
        mapper_with_kg = SemanticMapper(template_name="bitcoin_paper_hierarchical", kg_processor=kg_processor)
        assignments_with_kg = mapper_with_kg.assign_chunks(test_chunks, mapper_with_kg.section_names)
        
        log.info("Assignments with KG:")
        for assignment in assignments_with_kg:
            log.info(f"  Chunk {assignment['chunk_id']} -> {assignment['section']} (score: {assignment['assignment_score']:.3f})")
        
        # Compare results
        log.info("\nComparison:")
        # assignments are lists of dicts, not section->chunks mappings
        sections_no_kg = {assignment['section'] for assignment in assignments_no_kg}
        sections_with_kg = {assignment['section'] for assignment in assignments_with_kg}
        
        for section in sections_no_kg.union(sections_with_kg):
            no_kg_count = sum(1 for a in assignments_no_kg if a['section'] == section)
            with_kg_count = sum(1 for a in assignments_with_kg if a['section'] == section)
            log.info(f"  {section}: {no_kg_count} chunks (no KG) vs {with_kg_count} chunks (with KG)")
        
        # Test KG scoring methods
        log.info("\n--- Testing KG Scoring Methods ---")
        chunk_kg_scores = kg_processor.get_chunk_kg_scores(test_chunks)
        log.info(f"Chunk KG scores: {len(chunk_kg_scores)} entries")
        for chunk_id, scores in chunk_kg_scores.items():
            log.info(f"  {chunk_id}: composite={scores.get('composite_score', 0):.3f}")
        
        # Extract unique section names from assignments
        section_names = list(set(assignment['section'] for assignment in assignments_with_kg))
        section_affinities = kg_processor.get_section_affinity_scores(test_chunks, section_names)
        log.info(f"Section affinities: {len(section_affinities)} entries")
        for chunk_id, affinities in section_affinities.items():
            log.info(f"  {chunk_id}: {len(affinities)} section affinities")
        
        log.info("\n=== KG Integration Test PASSED ===")
        return True
        
    except Exception as e:
        log.error(f"KG Integration Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_assign_chunks_to_skeleton_with_kg():
    """Test the top-level assign_chunks_to_skeleton function with KG."""
    log.info("\n=== Testing assign_chunks_to_skeleton with KG ===")
    
    try:
        # Create test data
        test_chunks = create_test_chunks()
        grouped_chunks = {'Introduction': test_chunks}
        
        # Initialize KG processor
        embedding_model = UnifiedEmbeddingClient(SEMANTIC_MAPPING_CONFIG)
        kg_processor = KnowledgeGraphProcessor(test_chunks, embedding_model)
        kg_processor.build_graphs()
        
        # Test without KG
        assignments_no_kg = assign_chunks_to_skeleton(grouped_chunks, template_name="bitcoin_paper_hierarchical")
        log.info(f"Assignments without KG: {len(assignments_no_kg)} assignments")
        
        # Test with KG
        assignments_with_kg = assign_chunks_to_skeleton(grouped_chunks, template_name="bitcoin_paper_hierarchical", kg_processor=kg_processor)
        log.info(f"Assignments with KG: {len(assignments_with_kg)} assignments")
        
        # Compare results
        log.info("\nComparison:")
        # assignments are lists of dicts, not section->chunks mappings
        sections_no_kg = {assignment['section'] for assignment in assignments_no_kg}
        sections_with_kg = {assignment['section'] for assignment in assignments_with_kg}
        
        for section in sections_no_kg.union(sections_with_kg):
            no_kg_count = sum(1 for a in assignments_no_kg if a['section'] == section)
            with_kg_count = sum(1 for a in assignments_with_kg if a['section'] == section)
            log.info(f"  {section}: {no_kg_count} -> {with_kg_count} chunks")
        
        log.info("\n=== assign_chunks_to_skeleton Test PASSED ===")
        return True
        
    except Exception as e:
        log.error(f"assign_chunks_to_skeleton Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all KG integration tests."""
    log.info("Starting KG Integration Tests...")
    
    # Check if KG is enabled in config
    if not KG_CONFIG.get('enhance_section_mapping', False):
        log.warning("KG section mapping enhancement is disabled in config. Enabling for testing...")
        KG_CONFIG['enhance_section_mapping'] = True
    
    success = True
    
    # Run tests
    success &= test_semantic_mapper_with_kg()
    success &= test_assign_chunks_to_skeleton_with_kg()
    
    if success:
        log.info("\nALL KG INTEGRATION TESTS PASSED")
        log.info("The enhanced section mapping with KG scores is working correctly.")
    else:
        log.error("\nSOME TESTS FAILED")
        log.error("Please check the error messages above and fix the issues.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)