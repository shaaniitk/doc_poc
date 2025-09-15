#!/usr/bin/env python3
"""Enhanced Main Orchestrator for LangGraph Document Processing

This module provides the main entry point for the enhanced document processing
system built with LangGraph orchestration, featuring performance optimization,
intelligent error handling, and comprehensive monitoring.
"""

import asyncio
import logging
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import traceback
from contextlib import asynccontextmanager
import signal
import os

# Core components
from core.state_manager import CentralizedStateManager, ProcessingStage
from core.langgraph_orchestrator import LangGraphOrchestrator, NodeConfig
from langgraph_state import PipelineState
from core.document_parser import DocumentParser
from core.chunking_processor import ChunkingProcessor, ChunkingConfig, ChunkingStrategy
from core.llm_handler import LLMHandler, LLMConfig, ProcessingMode
from core.knowledge_graph_processor import KnowledgeGraphProcessor, ExtractionConfig
from core.output_generator import OutputGenerator, OutputConfig, OutputFormat, TemplateType

# Performance and monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not available. Install psutil for system monitoring.")

try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    logging.warning("memory_profiler not available. Install memory_profiler for memory monitoring.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_processing.log')
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Main processing configuration."""
    # Input/Output
    source_path: Path
    output_path: Optional[Path] = None
    output_format: OutputFormat = OutputFormat.MARKDOWN
    template_type: Optional[TemplateType] = None
    
    # Processing options
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_AWARE
    chunk_size: int = 1000
    chunk_overlap: int = 200
    processing_mode: ProcessingMode = ProcessingMode.PARALLEL
    max_concurrent: int = 5
    
    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000
    
    # Knowledge graph options
    enable_knowledge_graph: bool = True
    kg_extraction_method: str = "hybrid"
    
    # Performance options
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_compression: bool = True
    memory_limit_mb: int = 2048
    
    # Monitoring options
    enable_monitoring: bool = True
    enable_profiling: bool = False
    log_level: str = "INFO"
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    fail_fast: bool = False
    
    # Quality settings
    min_chunk_quality: float = 0.5
    min_llm_confidence: float = 0.6
    enable_quality_gates: bool = True


@dataclass
class ProcessingMetrics:
    """Processing performance metrics."""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    
    # Document metrics
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    
    # Processing stages
    parsing_time: float = 0.0
    chunking_time: float = 0.0
    llm_processing_time: float = 0.0
    kg_processing_time: float = 0.0
    output_generation_time: float = 0.0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # Quality metrics
    avg_chunk_quality: float = 0.0
    avg_llm_confidence: float = 0.0
    
    # Error metrics
    total_errors: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        self.end_time = datetime.now(timezone.utc)
        self.total_duration = (self.end_time - self.start_time).total_seconds()


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and HAS_PSUTIL
        self.process = psutil.Process() if self.enabled else None
        self.initial_memory = self.get_memory_usage() if self.enabled else 0
        self.peak_memory = 0
        self.cpu_samples = []
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.enabled:
            return 0
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if not self.enabled:
            return 0
        return self.process.cpu_percent()
    
    def update_metrics(self, metrics: ProcessingMetrics) -> None:
        """Update processing metrics with current system stats."""
        if not self.enabled:
            return
        
        current_memory = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        cpu_usage = self.get_cpu_usage()
        self.cpu_samples.append(cpu_usage)
        
        metrics.peak_memory_mb = self.peak_memory
        metrics.avg_cpu_percent = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0


class EnhancedDocumentProcessor:
    """Enhanced document processor with LangGraph orchestration."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.metrics = ProcessingMetrics()
        self.monitor = SystemMonitor(config.enable_monitoring)
        
        # Initialize components
        self.state_manager = None
        self.orchestrator = None
        self.document_parser = None
        self.chunking_processor = None
        self.llm_handler = None
        self.kg_processor = None
        self.output_generator = None
        
        # Shutdown handling
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        logger.info("EnhancedDocumentProcessor initialized")
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
    
    async def initialize_async(self) -> None:
        """Initialize all components asynchronously."""
        try:
            logger.info("Initializing processing components...")
            
            # Initialize state manager
            self.state_manager = CentralizedStateManager()
            await self.state_manager.initialize_async()
            
            # Initialize document parser
            self.document_parser = DocumentParser(
                enable_metadata_extraction=True,
                enable_structure_analysis=True,
                max_file_size_mb=100
            )
            
            # Initialize chunking processor
            chunking_config = ChunkingConfig(
                strategy=self.config.chunking_strategy,
                chunk_size=self.config.chunk_size,
                overlap_size=self.config.chunk_overlap,
                min_chunk_size=100,
                enable_quality_scoring=self.config.enable_quality_gates,
                min_quality_score=self.config.min_chunk_quality
            )
            self.chunking_processor = ChunkingProcessor(chunking_config)
            
            # Initialize LLM handler
            llm_config = LLMConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                enable_caching=self.config.enable_caching,
                cache_ttl=self.config.cache_ttl,
                max_retries=self.config.max_retries,
                retry_delay=self.config.retry_delay
            )
            self.llm_handler = LLMHandler(llm_config)
            await self.llm_handler.initialize_async()
            
            # Initialize knowledge graph processor
            if self.config.enable_knowledge_graph:
                kg_config = ExtractionConfig(
                    extraction_method=self.config.kg_extraction_method,
                    enable_entity_linking=self.config.enable_caching,
                    min_confidence=0.5
                )
                self.kg_processor = KnowledgeGraphProcessor(kg_config)
                await self.kg_processor.initialize_async()
            
            # Initialize output generator
            output_config = OutputConfig(
                output_format=self.config.output_format,
                include_metadata=True,
                include_statistics=True,
                include_visualizations=True
            )
            
            if self.config.template_type:
                from core.output_generator import create_output_generator
                self.output_generator = create_output_generator(
                    self.config.output_format,
                    self.config.template_type
                )
            else:
                self.output_generator = OutputGenerator(output_config)
            
            # Initialize LangGraph orchestrator
            workflow_config = NodeConfig(
                name="main_workflow",
                parallel_execution=True,
                max_retries=self.config.max_retries,
                timeout_seconds=300.0,
                error_recovery_strategy="retry"
            )
            
            self.orchestrator = LangGraphOrchestrator(
                state_manager=self.state_manager,
                document_parser=self.document_parser,
                chunking_processor=self.chunking_processor,
                llm_handler=self.llm_handler,
                kg_processor=self.kg_processor,
                output_generator=self.output_generator,
                config=workflow_config
            )
            
            await self.orchestrator.initialize_async()
            
            logger.info("All components initialized successfully")
        
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    async def process_document_async(self) -> Dict[str, Any]:
        """Process document with full orchestration."""
        try:
            logger.info(f"Starting document processing: {self.config.source_path}")
            
            # Check if shutdown was requested
            if self._shutdown_requested:
                raise KeyboardInterrupt("Processing cancelled by user")
            
            # Initialize processing state
            initial_state = PipelineState(
                stage=ProcessingStage.PARSING,
                source_path=str(self.config.source_path),
                output_path=str(self.config.output_path) if self.config.output_path else None,
                config=self.config.__dict__
            )
            
            # Start orchestrated processing
            start_time = time.time()
            
            final_state = await self.orchestrator.process_document_async(
                initial_state,
                progress_callback=self._progress_callback
            )
            
            # Update metrics
            self.metrics.total_duration = time.time() - start_time
            self.monitor.update_metrics(self.metrics)
            self.metrics.finalize()
            
            # Prepare results
            results = {
                "success": final_state.stage == ProcessingStage.COMPLETED,
                "final_state": final_state,
                "metrics": self.metrics,
                "output_path": final_state.output_path,
                "processing_time": self.metrics.total_duration,
                "chunks_processed": self.metrics.processed_chunks,
                "errors": final_state.errors
            }
            
            if results["success"]:
                logger.info(f"Document processing completed successfully in {self.metrics.total_duration:.2f}s")
            else:
                logger.error(f"Document processing failed: {final_state.errors}")
            
            return results
        
        except KeyboardInterrupt:
            logger.info("Processing cancelled by user")
            return {
                "success": False,
                "error": "Processing cancelled by user",
                "metrics": self.metrics
            }
        
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            logger.error(traceback.format_exc())
            
            self.metrics.total_errors += 1
            error_type = type(e).__name__
            self.metrics.error_types[error_type] = self.metrics.error_types.get(error_type, 0) + 1
            
            return {
                "success": False,
                "error": str(e),
                "error_type": error_type,
                "metrics": self.metrics,
                "traceback": traceback.format_exc()
            }
    
    async def _progress_callback(self, state: PipelineState) -> None:
        """Handle progress updates from orchestrator."""
        # Update metrics based on current stage
        if state.stage == ProcessingStage.PARSING:
            logger.info("Document parsing in progress...")
        
        elif state.stage == ProcessingStage.CHUNKING:
            logger.info(f"Document chunking in progress... ({len(state.chunks)} chunks created)")
            self.metrics.total_chunks = len(state.chunks)
        
        elif state.stage == ProcessingStage.LLM_PROCESSING:
            logger.info("LLM processing in progress...")
            if state.llm_results:
                processed = len([r for r in state.llm_results.results if r.success])
                self.metrics.processed_chunks = processed
                self.metrics.failed_chunks = len(state.llm_results.results) - processed
        
        elif state.stage == ProcessingStage.KNOWLEDGE_GRAPH:
            logger.info("Knowledge graph processing in progress...")
        
        elif state.stage == ProcessingStage.OUTPUT_GENERATION:
            logger.info("Output generation in progress...")
        
        elif state.stage == ProcessingStage.COMPLETED:
            logger.info("Processing completed successfully")
        
        elif state.stage == ProcessingStage.ERROR:
            logger.error(f"Processing error: {state.errors}")
        
        # Update system metrics
        self.monitor.update_metrics(self.metrics)
        
        # Check for shutdown request
        if self._shutdown_requested:
            raise KeyboardInterrupt("Processing cancelled by user")
    
    async def cleanup_async(self) -> None:
        """Clean up resources."""
        try:
            logger.info("Cleaning up resources...")
            
            if self.orchestrator:
                await self.orchestrator.cleanup_async()
            
            if self.llm_handler:
                await self.llm_handler.cleanup_async()
            
            if self.kg_processor:
                await self.kg_processor.cleanup_async()
            
            if self.state_manager:
                await self.state_manager.cleanup_async()
            
            logger.info("Cleanup completed")
        
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def print_metrics(self) -> None:
        """Print processing metrics."""
        print("\n" + "="*60)
        print("PROCESSING METRICS")
        print("="*60)
        
        print(f"Total Duration: {self.metrics.total_duration:.2f}s")
        print(f"Chunks Processed: {self.metrics.processed_chunks}/{self.metrics.total_chunks}")
        print(f"Success Rate: {(self.metrics.processed_chunks/max(self.metrics.total_chunks, 1)*100):.1f}%")
        
        if self.monitor.enabled:
            print(f"Peak Memory: {self.metrics.peak_memory_mb:.1f} MB")
            print(f"Average CPU: {self.metrics.avg_cpu_percent:.1f}%")
        
        if self.metrics.total_errors > 0:
            print(f"Total Errors: {self.metrics.total_errors}")
            for error_type, count in self.metrics.error_types.items():
                print(f"  {error_type}: {count}")
        
        print("\nStage Timings:")
        print(f"  Parsing: {self.metrics.parsing_time:.2f}s")
        print(f"  Chunking: {self.metrics.chunking_time:.2f}s")
        print(f"  LLM Processing: {self.metrics.llm_processing_time:.2f}s")
        print(f"  Knowledge Graph: {self.metrics.kg_processing_time:.2f}s")
        print(f"  Output Generation: {self.metrics.output_generation_time:.2f}s")
        
        print("="*60)


@asynccontextmanager
async def create_processor(config: ProcessingConfig):
    """Context manager for processor lifecycle."""
    processor = EnhancedDocumentProcessor(config)
    try:
        await processor.initialize_async()
        yield processor
    finally:
        await processor.cleanup_async()


def create_config_from_args(args) -> ProcessingConfig:
    """Create processing configuration from command line arguments."""
    config = ProcessingConfig(
        source_path=Path(args.source),
        output_path=Path(args.output) if args.output else None,
        output_format=OutputFormat[args.format.upper()],
        chunking_strategy=ChunkingStrategy[args.chunking_strategy.upper()],
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        processing_mode=ProcessingMode[args.processing_mode.upper()],
        max_concurrent=args.max_concurrent,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_temperature=args.temperature,
        llm_max_tokens=args.max_tokens,
        enable_knowledge_graph=args.enable_kg,
        kg_extraction_method=args.kg_method,
        enable_caching=args.enable_caching,
        enable_monitoring=args.enable_monitoring,
        enable_profiling=args.enable_profiling,
        max_retries=args.max_retries,
        fail_fast=args.fail_fast,
        min_chunk_quality=args.min_chunk_quality,
        min_llm_confidence=args.min_llm_confidence,
        enable_quality_gates=args.enable_quality_gates
    )
    
    if args.template:
        config.template_type = TemplateType[args.template.upper()]
    
    return config


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced Document Processing with LangGraph Orchestration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "source",
        help="Source document path"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        help="Output file path (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=[f.name.lower() for f in OutputFormat],
        default="markdown",
        help="Output format"
    )
    
    parser.add_argument(
        "-t", "--template",
        choices=[t.name.lower() for t in TemplateType],
        help="Output template type"
    )
    
    # Processing options
    parser.add_argument(
        "--chunking-strategy",
        choices=[s.name.lower() for s in ChunkingStrategy],
        default="sentence_aware",
        help="Chunking strategy"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters"
    )
    
    parser.add_argument(
        "--processing-mode",
        choices=[m.name.lower() for m in ProcessingMode],
        default="parallel",
        help="LLM processing mode"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent operations"
    )
    
    # LLM options
    parser.add_argument(
        "--llm-provider",
        default="openai",
        help="LLM provider"
    )
    
    parser.add_argument(
        "--llm-model",
        default="gpt-3.5-turbo",
        help="LLM model name"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens per LLM request"
    )
    
    # Knowledge graph options
    parser.add_argument(
        "--enable-kg",
        action="store_true",
        default=True,
        help="Enable knowledge graph processing"
    )
    
    parser.add_argument(
        "--kg-method",
        default="hybrid",
        help="Knowledge graph extraction method"
    )
    
    # Performance options
    parser.add_argument(
        "--enable-caching",
        action="store_true",
        default=True,
        help="Enable caching"
    )
    
    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        default=True,
        help="Enable system monitoring"
    )
    
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable performance profiling"
    )
    
    # Error handling
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Fail fast on first error"
    )
    
    # Quality options
    parser.add_argument(
        "--min-chunk-quality",
        type=float,
        default=0.5,
        help="Minimum chunk quality score"
    )
    
    parser.add_argument(
        "--min-llm-confidence",
        type=float,
        default=0.6,
        help="Minimum LLM confidence score"
    )
    
    parser.add_argument(
        "--enable-quality-gates",
        action="store_true",
        default=True,
        help="Enable quality gates"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--config",
        help="Load configuration from JSON file"
    )
    
    return parser


async def main_async() -> int:
    """Main async entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            # Override with command line arguments
            for key, value in vars(args).items():
                if value is not None:
                    config_data[key] = value
            config = ProcessingConfig(**config_data)
        else:
            config = create_config_from_args(args)
        
        # Validate source file
        if not config.source_path.exists():
            logger.error(f"Source file not found: {config.source_path}")
            return 1
        
        # Generate output path if not specified
        if not config.output_path:
            output_name = f"{config.source_path.stem}_processed.{config.output_format.name.lower()}"
            config.output_path = config.source_path.parent / output_name
        
        logger.info(f"Processing: {config.source_path} -> {config.output_path}")
        
        # Process document
        async with create_processor(config) as processor:
            results = await processor.process_document_async()
            
            # Print results
            if results["success"]:
                print(f"\n✅ Processing completed successfully!")
                print(f"Output saved to: {results['output_path']}")
                print(f"Processing time: {results['processing_time']:.2f}s")
                print(f"Chunks processed: {results['chunks_processed']}")
            else:
                print(f"\n❌ Processing failed: {results['error']}")
                if 'traceback' in results:
                    logger.debug(results['traceback'])
            
            # Print detailed metrics
            processor.print_metrics()
            
            return 0 if results["success"] else 1
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return 1


def main() -> int:
    """Main entry point."""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())