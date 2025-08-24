"""Unit tests for semantic validation and context management"""
import unittest
from unittest.mock import Mock, patch
import numpy as np
from modules.semantic_validator import SemanticValidator, SmartContextManager


class TestSemanticValidator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.validator = SemanticValidator()
    
    def test_validate_content_quality_valid_content(self):
        """Test content quality validation with valid content"""
        content = "This is a well-structured academic paper discussing machine learning algorithms and their applications in data science."
        is_valid, score = self.validator.validate_content_quality(content)
        
        self.assertTrue(is_valid)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)
    
    def test_validate_content_quality_invalid_content(self):
        """Test content quality validation with invalid content"""
        # Test empty content
        is_valid, score = self.validator.validate_content_quality("")
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)
        
        # Test very short content
        is_valid, score = self.validator.validate_content_quality("Hi")
        self.assertFalse(is_valid)
        self.assertLess(score, 0.5)
        
        # Test repetitive content
        repetitive = "test " * 100
        is_valid, score = self.validator.validate_content_quality(repetitive)
        self.assertFalse(is_valid)
        self.assertLess(score, 0.5)
    
    def test_validate_content_coherence_similar_content(self):
        """Test coherence validation with similar content"""
        content1 = "Machine learning is a subset of artificial intelligence that focuses on algorithms."
        content2 = "Artificial intelligence includes machine learning algorithms and deep learning techniques."
        
        coherence = self.validator.validate_content_coherence(content1, content2)
        
        self.assertGreater(coherence, 0.3)
        self.assertLessEqual(coherence, 1.0)
    
    def test_validate_content_coherence_dissimilar_content(self):
        """Test coherence validation with dissimilar content"""
        content1 = "Machine learning algorithms are used for data analysis."
        content2 = "Cooking recipes require precise measurements and timing."
        
        coherence = self.validator.validate_content_coherence(content1, content2)
        
        self.assertLess(coherence, 0.5)
        self.assertGreaterEqual(coherence, 0.0)
    
    def test_validate_content_coherence_identical_content(self):
        """Test coherence validation with identical content"""
        content = "This is identical content for testing purposes."
        
        coherence = self.validator.validate_content_coherence(content, content)
        
        self.assertGreater(coherence, 0.9)
    
    def test_validate_content_coherence_empty_content(self):
        """Test coherence validation with empty content"""
        coherence = self.validator.validate_content_coherence("", "some content")
        self.assertEqual(coherence, 0.0)
        
        coherence = self.validator.validate_content_coherence("some content", "")
        self.assertEqual(coherence, 0.0)
        
        coherence = self.validator.validate_content_coherence("", "")
        self.assertEqual(coherence, 0.0)


class TestSmartContextManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.context_manager = SmartContextManager()
    
    def test_build_context_single_query(self):
        """Test context building with single query"""
        queries = ["machine learning"]
        documents = [
            "Machine learning is a method of data analysis.",
            "Deep learning uses neural networks.",
            "Cooking involves preparing food."
        ]
        
        context = self.context_manager.build_context(queries, documents)
        
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)
        # Should prioritize ML-related content
        self.assertIn("machine learning", context.lower())
    
    def test_build_context_multiple_queries(self):
        """Test context building with multiple queries"""
        queries = ["neural networks", "deep learning"]
        documents = [
            "Neural networks are computational models.",
            "Deep learning uses multiple layers.",
            "Traditional algorithms are rule-based."
        ]
        
        context = self.context_manager.build_context(queries, documents)
        
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)
        # Should include both neural networks and deep learning content
        self.assertTrue(
            "neural" in context.lower() or "deep" in context.lower()
        )
    
    def test_build_context_empty_inputs(self):
        """Test context building with empty inputs"""
        # Empty queries
        context = self.context_manager.build_context([], ["some document"])
        self.assertEqual(context, "")
        
        # Empty documents
        context = self.context_manager.build_context(["query"], [])
        self.assertEqual(context, "")
        
        # Both empty
        context = self.context_manager.build_context([], [])
        self.assertEqual(context, "")
    
    def test_build_context_max_length(self):
        """Test context building respects max length"""
        queries = ["test"]
        long_document = "This is a very long document. " * 200
        documents = [long_document]
        
        context = self.context_manager.build_context(queries, documents, max_length=500)
        
        self.assertLessEqual(len(context), 500)
        self.assertGreater(len(context), 0)
    
    def test_create_section_summary_normal_content(self):
        """Test section summary creation with normal content"""
        section_name = "Introduction"
        content = "This paper introduces machine learning concepts and their applications in various domains. We discuss supervised and unsupervised learning approaches."
        
        summary = self.context_manager.create_section_summary(section_name, content)
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLessEqual(len(summary), len(content))
    
    def test_create_section_summary_long_content(self):
        """Test section summary creation with long content"""
        section_name = "Methodology"
        long_content = "This is a very detailed methodology section. " * 100
        
        summary = self.context_manager.create_section_summary(section_name, long_content)
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        # Summary should be shorter than original for long content
        self.assertLess(len(summary), len(long_content))
    
    def test_create_section_summary_short_content(self):
        """Test section summary creation with short content"""
        section_name = "Conclusion"
        short_content = "Brief conclusion."
        
        summary = self.context_manager.create_section_summary(section_name, short_content)
        
        self.assertIsInstance(summary, str)
        self.assertEqual(summary, short_content)  # Should return as-is for short content
    
    def test_create_section_summary_empty_content(self):
        """Test section summary creation with empty content"""
        summary = self.context_manager.create_section_summary("Test", "")
        self.assertEqual(summary, "")


class TestSemanticValidatorIntegration(unittest.TestCase):
    """Integration tests for semantic validation components"""
    
    def setUp(self):
        self.validator = SemanticValidator()
        self.context_manager = SmartContextManager()
    
    def test_quality_and_coherence_workflow(self):
        """Test complete workflow of quality validation and coherence checking"""
        # Original content
        original = "Machine learning algorithms require large datasets for training and validation."
        
        # Enhanced content (should be coherent)
        enhanced = "Machine learning algorithms need extensive datasets for proper training, validation, and testing phases."
        
        # Validate quality of both
        orig_valid, orig_score = self.validator.validate_content_quality(original)
        enh_valid, enh_score = self.validator.validate_content_quality(enhanced)
        
        self.assertTrue(orig_valid)
        self.assertTrue(enh_valid)
        
        # Check coherence
        coherence = self.validator.validate_content_coherence(original, enhanced)
        
        self.assertGreater(coherence, 0.5)  # Should be coherent
    
    def test_context_building_with_validation(self):
        """Test context building combined with content validation"""
        queries = ["data science"]
        documents = [
            "Data science combines statistics, programming, and domain expertise.",
            "Invalid content: xyz abc def.",  # Low quality
            "Data scientists use various tools for analysis and visualization."
        ]
        
        # Filter documents by quality before building context
        valid_docs = []
        for doc in documents:
            is_valid, score = self.validator.validate_content_quality(doc)
            if is_valid:
                valid_docs.append(doc)
        
        context = self.context_manager.build_context(queries, valid_docs)
        
        self.assertGreater(len(context), 0)
        # Should not contain the invalid content
        self.assertNotIn("xyz abc def", context)


if __name__ == '__main__':
    unittest.main()