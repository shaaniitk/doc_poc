
#!/usr/bin/env python3
"""Example Usage of Enhanced Core Components

This script demonstrates how to use all enhanced core components together
to process a document from raw text to final formatted output.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Import enhanced components
from core.enhanced_chunking_processor import (
    create_enhanced_chunking_processor, ChunkingStrategy
)
from core.enhanced_semantic_mapper import (
    create_enhanced_semantic_mapper, MappingStrategy
)
from core.enhanced_aggregation_engine import (
    create_enhanced_aggregation_engine, AggregationStrategy
)
from core.enhanced_output_generator import (
    create_enhanced_output_generator, OutputFormat, TemplateType
)
from core.chunking_processor import DocumentChunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_document_example():
    """Example of processing a document through the complete pipeline."""
    
    # Sample document content
    document_content = """
# Artificial Intelligence in Healthcare

Artificial Intelligence (AI) is revolutionizing healthcare by providing innovative solutions for diagnosis, treatment, and patient care. This technology leverages machine learning algorithms and data analytics to improve medical outcomes.

## Applications in Medical Diagnosis

AI systems can analyze medical images, such as X-rays, MRIs, and CT scans, with remarkable accuracy. Deep learning models trained on vast datasets can detect anomalies that might be missed by human radiologists.

### Image Recognition
Convolutional neural networks excel at identifying patterns in medical imagery, enabling early detection of diseases like cancer, fractures, and neurological conditions.

### Predictive Analytics
Machine learning algorithms can predict patient outcomes by analyzing electronic health records, lab results, and vital signs, helping healthcare providers make informed decisions.

## Drug Discovery and Development

AI accelerates the drug discovery process by identifying potential compounds, predicting their effectiveness, and optimizing clinical trial designs. This reduces the time and cost associated with bringing new medications to market.

## Challenges and Ethical Considerations

Despite its potential, AI in healthcare faces challenges including data privacy, algorithmic bias, regulatory compliance, and the need for clinical validation. Ensuring patient safety and maintaining trust in AI systems is paramount.

## Future Prospects

The future of AI in healthcare looks promising, with developments in personalized medicine, robotic surgery, and telemedicine. As technology continues to advance, we can expect more sophisticated and accessible healthcare solutions.
    """
    
    logger.info("Starting document processing example")
    
    # Step 1: Enhanced Chunking
    logger.info("Step 1: Processing with enhanced chunking")
    chunking_processor = create_enhanced_chunking_processor(
        enable_workflow=True,
        enable_adaptive=True,
        enable_semantic_analysis=True
    )
    
    # Create ParsedDocument object
    from core.document_parser import ParsedDocument, DocumentMetadata, DocumentFormat
    from pathlib import Path
    
    document = ParsedDocument(
        content=document_content,
        metadata=DocumentMetadata(
            file_path=Path("ai_healthcare.md"),
            format=DocumentFormat.MARKDOWN,
            file_size=len(document_content.encode('utf-8')),
            creation_time=datetime.now(),
            modification_time=datetime.now()
        )
    )
    
    chunking_result = await chunking_processor.process_document_async(document)
    
    chunks = chunking_result.chunks
    logger.info(f"Generated {len(chunks)} chunks")
    
    # Step 2: Enhanced Semantic Mapping
    logger.info("Step 2: Creating semantic mappings")
    semantic_mapper = create_enhanced_semantic_mapper(
        enable_workflow=True,
        primary_strategy=MappingStrategy.HYBRID,
        enable_relationship_detection=True,
        enable_clustering=True
    )
    
    mapping_result = await semantic_mapper.create_semantic_mappings_async(chunks)
    logger.info(f"Generated {len(mapping_result['semantic_relationships'])} relationships")
    logger.info(f"Generated {len(mapping_result['semantic_clusters'])} clusters")
    
    # Step 3: Enhanced Aggregation
    logger.info("Step 3: Aggregating content intelligently")
    aggregation_engine = create_enhanced_aggregation_engine(
        enable_workflow=True,
        primary_strategy=AggregationStrategy.ADAPTIVE,
        enable_llm_synthesis=True,
        enable_hierarchical_organization=True
    )
    
    aggregation_result = await aggregation_engine.aggregate_content_async(
        chunks,
        mapping_result
    )
    logger.info(f"Generated {len(aggregation_result['aggregated_content'])} aggregated sections")
    
    # Step 4: Enhanced Output Generation
    logger.info("Step 4: Generating final document")
    output_generator = create_enhanced_output_generator(
        format=OutputFormat.LATEX,
        template_type=TemplateType.TECHNICAL_REPORT,
        title="AI in Healthcare: A Comprehensive Overview",
        author="Document Processing System",
        abstract="This document provides a comprehensive overview of artificial intelligence applications in healthcare, generated through automated document processing.",
        keywords=["artificial intelligence", "healthcare", "machine learning", "medical diagnosis"],
        include_toc=True,
        output_directory="./example_output"
    )
    
    final_document = await output_generator.generate_document_async(aggregation_result)
    
    logger.info(f"Final document generated: {final_document.file_path}")
    logger.info(f"Document length: {len(final_document.content)} characters")
    
    # Display summary
    print("\n" + "="*60)
    print("DOCUMENT PROCESSING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Input document: {len(document_content)} characters")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Semantic relationships: {len(mapping_result['semantic_relationships'])}")
    print(f"Semantic clusters: {len(mapping_result['semantic_clusters'])}")
    print(f"Aggregated sections: {len(aggregation_result['aggregated_content'])}")
    print(f"Final document: {final_document.file_path}")
    print(f"Output format: {final_document.format.value}")
    print(f"Template type: {final_document.template_type.value}")
    print("="*60)
    
    return final_document


if __name__ == "__main__":
    # Run the example
    asyncio.run(process_document_example())
