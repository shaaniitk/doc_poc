import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from modules.format_enforcer import FormatEnforcer
from modules.error_handler import ProcessingError, ChunkingError
from config import OUTPUT_FORMATS
from langgraph_state import (
    PipelineState, OutputData, ProcessingStage, ErrorInfo, AnalyticsData, create_initial_state
)
from langgraph_config import ProcessingConfiguration

def test_validate_output_detects_markdown_in_latex():
    fe = FormatEnforcer(output_format="latex")
    issues = fe.validate_output("This is **bold** text")
    assert any("textbf" in msg for msg in issues)


def test_post_process_converts_markdown_to_latex():
    fe = FormatEnforcer(output_format="latex")
    fixed = fe.post_process_output("This is **bold** and *italic*\n- item")
    assert "\\textbf{" in fixed
    assert "\\textit{" in fixed
    assert "\\item" in fixed


class TestFormatEnforcer:
    """Test class for FormatEnforcer with LangGraph state integration."""
    
    @pytest.fixture
    def enforcer(self):
        """Create a FormatEnforcer instance for testing."""
        return FormatEnforcer()
    
    def test_enforce_json_format_success(self, enforcer):
        """Test successful JSON format enforcement."""
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        
        result = enforcer.enforce_format(data, "json")
        
        assert result is not None
        assert isinstance(result, str)
        
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed == data
    
    def test_format_enforcement_with_langgraph_state(self, enforcer):
        """Test format enforcement with LangGraph state integration."""
        # Create initial pipeline state
        config = ProcessingConfiguration(
            output_format="json",
            format_validation_enabled=True
        )
        state = create_initial_state(
            document_path="format_test.txt",
            processing_config=config
        )
        
        # Test data to format
        data = {
            "document_analysis": {
                "chunks": 5,
                "total_tokens": 1500,
                "key_concepts": ["AI", "ML", "NLP"]
            },
            "processing_metadata": {
                "timestamp": "2024-01-01T00:00:00Z",
                "version": "1.0"
            }
        }
        
        # Update state to output formatting stage
        state.current_stage = ProcessingStage.OUTPUT_GENERATION
        
        # Perform format enforcement
        formatted_result = enforcer.enforce_format(data, "json")
        
        # Create OutputInfo from result
        output_info = OutputInfo(
            format_type="json",
            content=formatted_result,
            file_path="output.json",
            size_bytes=len(formatted_result.encode('utf-8')),
            validation_passed=True,
            metadata={
                "schema_version": "1.0",
                "content_type": "application/json",
                "encoding": "utf-8"
            }
        )
        
        # Update state with output
        state.output_info = output_info
        state.current_stage = ProcessingStage.OUTPUT_COMPLETE
        
        # Verify state integration
        assert state.output_info is not None
        assert state.output_info.format_type == "json"
        assert state.output_info.validation_passed is True
        assert state.current_stage == ProcessingStage.OUTPUT_COMPLETE
        
        # Verify JSON validity
        parsed = json.loads(formatted_result)
        assert parsed == data
    
    def test_enforce_format_validation_error(self, enforcer):
        """Test format enforcement with validation error."""
        # Create data that might cause validation issues
        invalid_data = {"circular_ref": None}
        invalid_data["circular_ref"] = invalid_data  # Circular reference
        
        with pytest.raises(FormatError):
            enforcer.enforce_format(invalid_data, "json")
    
    def test_format_validation_error_with_state(self, enforcer):
        """Test format validation error with state tracking."""
        # Create initial state
        config = ProcessingConfiguration(
            output_format="json",
            format_validation_enabled=True
        )
        state = create_initial_state(
            document_path="validation_error_test.txt",
            processing_config=config
        )
        
        # Create problematic data
        invalid_data = {"circular_ref": None}
        invalid_data["circular_ref"] = invalid_data
        
        try:
            enforcer.enforce_format(invalid_data, "json")
        except FormatError as e:
            # Create error info
            error_info = ErrorInfo(
                error_type="FormatError",
                message=str(e),
                stage=ProcessingStage.OUTPUT_GENERATION,
                recoverable=False,
                context={
                    "format_type": "json",
                    "validation_enabled": True,
                    "error_cause": "circular_reference"
                }
            )
            
            # Add error to state
            state.errors.append(error_info)
            state.current_stage = ProcessingStage.ERROR
            
            # Verify error tracking
            assert len(state.errors) == 1
            assert state.errors[0].error_type == "FormatError"
            assert state.errors[0].recoverable is False
            assert state.current_stage == ProcessingStage.ERROR
    
    def test_format_validation_comprehensive(self, enforcer):
        """Test comprehensive format validation."""
        test_cases = [
            ({"simple": "data"}, "json", True),
            ({"nested": {"data": [1, 2, 3]}}, "json", True),
            ("<root><item>test</item></root>", "xml", True),
            ("Simple text content", "text", True)
        ]
        
        for data, format_type, should_succeed in test_cases:
            if should_succeed:
                result = enforcer.enforce_format(data, format_type)
                assert result is not None
                assert isinstance(result, str)
            else:
                with pytest.raises((FormatError, ValidationError)):
                    enforcer.enforce_format(data, format_type)
    
    def test_multiple_format_enforcement_with_state(self, enforcer):
        """Test multiple format enforcement operations with state tracking."""
        # Create initial state
        config = ProcessingConfiguration(
            output_format="multiple",
            format_validation_enabled=True,
            supported_formats=["json", "xml", "text"]
        )
        state = create_initial_state(
            document_path="multi_format_test.txt",
            processing_config=config
        )
        
        # Test data
        data = {
            "analysis_results": {
                "sentiment": "positive",
                "confidence": 0.85,
                "keywords": ["innovation", "technology", "future"]
            }
        }
        
        # Test multiple formats
        formats_to_test = ["json", "xml", "text"]
        output_results = []
        
        state.current_stage = ProcessingStage.OUTPUT_GENERATION
        
        for format_type in formats_to_test:
            try:
                formatted_result = enforcer.enforce_format(data, format_type)
                
                output_info = OutputInfo(
                    format_type=format_type,
                    content=formatted_result,
                    file_path=f"output.{format_type}",
                    size_bytes=len(formatted_result.encode('utf-8')),
                    validation_passed=True,
                    metadata={
                        "format_index": len(output_results),
                        "content_type": f"application/{format_type}"
                    }
                )
                output_results.append(output_info)
                
            except Exception as e:
                error_info = ErrorInfo(
                    error_type=type(e).__name__,
                    message=str(e),
                    stage=ProcessingStage.OUTPUT_GENERATION,
                    recoverable=True,
                    context={"format_type": format_type}
                )
                state.errors.append(error_info)
        
        # Update state with results
        if output_results:
            state.output_info = output_results[0]  # Primary format
            state.current_stage = ProcessingStage.OUTPUT_COMPLETE
        
        # Create analytics for format enforcement
        analytics = AnalyticsInfo(
            processing_time=0.3,
            memory_usage=512,
            tokens_processed=0,  # Not applicable for formatting
            chunks_created=0,    # Not applicable for formatting
            average_chunk_size=0,
            processing_stage=ProcessingStage.OUTPUT_GENERATION,
            metadata={
                "formats_processed": len(formats_to_test),
                "successful_formats": len(output_results),
                "failed_formats": len(state.errors),
                "primary_format": output_results[0].format_type if output_results else None
            }
        )
        state.analytics.append(analytics)
        
        # Verify results
        assert len(output_results) > 0, "At least one format should succeed"
        assert state.current_stage == ProcessingStage.OUTPUT_COMPLETE
        assert len(state.analytics) == 1
        assert state.analytics[0].metadata["formats_processed"] == len(formats_to_test)
    
    def test_format_enforcement_pipeline_workflow(self, enforcer):
        """Test complete format enforcement pipeline workflow."""
        # Create comprehensive pipeline state
        config = ProcessingConfiguration(
            output_format="json",
            format_validation_enabled=True,
            output_compression=False,
            include_metadata=True
        )
        state = create_initial_state(
            document_path="pipeline_format_test.txt",
            processing_config=config
        )
        
        # Simulate processed document data
        processed_data = {
            "document_metadata": {
                "title": "Test Document",
                "author": "Test Author",
                "processing_date": "2024-01-01"
            },
            "content_analysis": {
                "total_chunks": 10,
                "total_tokens": 2500,
                "key_topics": ["AI", "Machine Learning", "Data Science"],
                "sentiment_score": 0.75
            },
            "processing_statistics": {
                "processing_time_seconds": 45.2,
                "memory_usage_mb": 128,
                "success_rate": 1.0
            }
        }
        
        # Execute format enforcement workflow
        state.current_stage = ProcessingStage.OUTPUT_GENERATION
        
        try:
            # Enforce format
            formatted_output = enforcer.enforce_format(processed_data, config.output_format)
            
            # Validate the formatted output
            if config.format_validation_enabled:
                json.loads(formatted_output)  # Validate JSON
            
            # Create comprehensive output info
            output_info = OutputInfo(
                format_type=config.output_format,
                content=formatted_output,
                file_path=f"final_output.{config.output_format}",
                size_bytes=len(formatted_output.encode('utf-8')),
                validation_passed=True,
                metadata={
                    "schema_version": "2.0",
                    "compression_enabled": config.output_compression,
                    "metadata_included": config.include_metadata,
                    "validation_enabled": config.format_validation_enabled,
                    "content_hash": hash(formatted_output)
                }
            )
            
            # Update state
            state.output_info = output_info
            state.current_stage = ProcessingStage.COMPLETE
            
            # Add final analytics
            analytics = AnalyticsInfo(
                processing_time=0.8,
                memory_usage=256,
                tokens_processed=processed_data["content_analysis"]["total_tokens"],
                chunks_created=processed_data["content_analysis"]["total_chunks"],
                average_chunk_size=processed_data["content_analysis"]["total_tokens"] / processed_data["content_analysis"]["total_chunks"],
                processing_stage=ProcessingStage.OUTPUT_GENERATION,
                metadata={
                    "output_format": config.output_format,
                    "output_size_bytes": output_info.size_bytes,
                    "validation_passed": output_info.validation_passed,
                    "pipeline_complete": True
                }
            )
            state.analytics.append(analytics)
            
            # Verify complete workflow
            assert state.current_stage == ProcessingStage.COMPLETE
            assert state.output_info.validation_passed is True
            assert state.output_info.format_type == config.output_format
            assert len(state.analytics) == 1
            assert state.analytics[0].metadata["pipeline_complete"] is True
            
            # Verify output content
            parsed_output = json.loads(formatted_output)
            assert "document_metadata" in parsed_output
            assert "content_analysis" in parsed_output
            assert "processing_statistics" in parsed_output
            
        except Exception as e:
            # Handle workflow errors
            error_info = ErrorInfo(
                error_type=type(e).__name__,
                message=str(e),
                stage=ProcessingStage.OUTPUT_GENERATION,
                recoverable=False,
                context={
                    "format_type": config.output_format,
                    "data_size": len(str(processed_data))
                }
            )
            
            state.errors.append(error_info)
            state.current_stage = ProcessingStage.ERROR
            
            pytest.fail(f"Format enforcement workflow failed: {e}")