"""Main modular document refactoring script"""
import sys
import os

from modules.file_loader import load_latex_file
from modules.chunker import extract_latex_sections, group_chunks_by_section, llm_enhance_chunking
from modules.section_mapper import assign_chunks_to_skeleton, get_section_prompt
from modules.contribution_tracker import ContributionTracker
from modules.llm_handler import ContextualLLMHandler
from modules.advanced_processing import AdvancedProcessor
from modules.intelligent_aggregation import IntelligentAggregator
from modules.output_manager import OutputManager
from modules.output_formatter import OutputFormatter
from config import CHUNKING_STRATEGIES, DOCUMENT_TEMPLATES

def main(source=None, source2=None, combine_strategy="smart_merge", chunking_strategy="semantic", output_format="latex", template="bitcoin_paper", analysis=None):
    # Handle document combination
    if source2:
        from modules.robust_combiner import RobustDocumentCombiner
        combiner = RobustDocumentCombiner(output_format)
        
        print(f"Combining documents: {source} + {source2}")
        combined_content, format_issues = combiner.combine_documents(
            source, source2, combine_strategy, output_format
        )
        
        if format_issues:
            print(f"Format issues detected: {format_issues}")
        
        # Save combined document
        output_manager = OutputManager()
        combined_path = output_manager.save_section_output("combined_document", combined_content)
        
        print(f"Combined document saved: {combined_path}")
        return
    
    # Configuration for single document processing
    if source and (source.startswith('http') or source.startswith('arxiv:')):
        from modules.tex_downloader import TexDownloader
        downloader = TexDownloader()
        
        if source.startswith('arxiv:'):
            arxiv_id = source.replace('arxiv:', '')
            source_file = downloader.download_arxiv_source(arxiv_id)
        else:
            source_file = downloader.download_tex_file(source)
        
        # Preprocess downloaded file
        source_file = downloader.preprocess_tex(source_file)
        print(f"Downloaded and preprocessed: {source_file}")
    else:
        source_file = source or "unstructured_document.tex"
    
    # Initialize modules
    llm_handler = ContextualLLMHandler(output_format=output_format)
    advanced_processor = AdvancedProcessor()
    aggregator = IntelligentAggregator()
    output_manager = OutputManager()
    formatter = OutputFormatter(output_format)
    contribution_tracker = ContributionTracker()
    log_entries = []
    
    try:
        # 1. Load file
        print("Loading LaTeX file...")
        content = load_latex_file(source_file)
        log_entries.append(f"Loaded file: {source_file}")
        
        # 2. Extract and chunk
        print("Extracting and chunking content...")
        chunks = extract_latex_sections(content)
        
        # 2.5. Skip LLM enhancement for now
        print("Using basic chunking...")
        enhanced_chunks = chunks
        
        grouped_chunks = group_chunks_by_section(enhanced_chunks)
        log_entries.append(f"Extracted {len(chunks)} chunks, enhanced to {len(enhanced_chunks)} chunks into {len(grouped_chunks)} sections")
        
        # 3. Assign to skeleton with tracking
        print("Assigning chunks to document skeleton...")
        assignments = assign_chunks_to_skeleton(grouped_chunks)
        
        # Track chunk contributions
        chunk_id = 0
        for section_name, section_chunks in assignments.items():
            for chunk in section_chunks:
                contribution_tracker.track_chunk_assignment(
                    chunk_id, 
                    chunk.get('parent_section', 'Unknown'),
                    section_name,
                    chunk['content']
                )
                chunk_id += 1
        
        log_entries.append("Assigned chunks to document skeleton")
        
        # 4. Process with LLM
        print("Processing sections with LLM...")
        processed_sections = {}
        
        for section_name, section_chunks in assignments.items():
            if section_chunks:
                print(f"  Processing: {section_name}")
                
                # Combine content for this section
                combined_content = '\n\n'.join([chunk['content'] for chunk in section_chunks])
                
                # Get section prompt
                prompt = get_section_prompt(section_name)
                
                # Simple LLM processing (bypass advanced for now)
                combined_content = '\n\n'.join([chunk['content'] for chunk in section_chunks])
                result = llm_handler.process_section(section_name, combined_content, prompt)
                
                # Save section output
                section_path = output_manager.save_section_output(section_name, result)
                processed_sections[section_name] = result
                
                log_entries.append(f"Processed section: {section_name} -> {section_path}")
            else:
                log_entries.append(f"Skipped empty section: {section_name}")
        
        # 5. Simple aggregation (bypass advanced for now)
        print("Aggregating final document...")
        final_path = output_manager.aggregate_document(processed_sections)
        log_entries.append(f"Final document saved: {final_path}")
        
        # 6. Save processing log and contribution report
        log_path = output_manager.save_processing_log(log_entries)
        contribution_path = contribution_tracker.save_contribution_report(
            os.path.join(output_manager.session_path, "chunk_contributions.md")
        )
        
        print(f"\nProcessing complete!")
        print(f"Final document: {final_path}")
        print(f"Processing log: {log_path}")
        print(f"Contribution report: {contribution_path}")
        print(f"Session outputs: {output_manager.session_path}")
        print(f"Processing completed successfully")
        
        # Run analysis if requested
        if analysis:
            run_analysis(analysis)
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        log_entries.append(error_msg)
        output_manager.save_processing_log(log_entries)

def run_analysis(analysis_type):
    """Run specified analysis using modular framework"""
    print(f"\n{'='*60}")
    print(f"RUNNING {analysis_type.upper()} ANALYSIS")
    print(f"{'='*60}")
    
    try:
        if analysis_type == "comprehensive":
            from modules.comprehensive_analysis import main as comp_main
            comp_main()
        elif analysis_type == "comparison":
            from modules.comparison_analysis import main as comp_main
            comp_main()
        elif analysis_type == "both":
            from modules.comprehensive_analysis import main as comp_main
            comp_main()
            print(f"\n{'='*60}")
            print("COMPARISON ANALYSIS")
            print(f"{'='*60}")
            from modules.comparison_analysis import main as comp_main2
            comp_main2()
        else:
            print(f"Unknown analysis type: {analysis_type}")
            print("Available: comprehensive, comparison, both")
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    import sys
    
    # Command line usage examples:
    # python modular_refactor.py
    # python modular_refactor.py --source https://example.com/paper.tex
    # python modular_refactor.py --source arxiv:2301.12345
    # python modular_refactor.py --analysis comprehensive
    # python modular_refactor.py --analysis comparison
    # python modular_refactor.py --analysis both
    # python modular_refactor.py --source bitcoin_whitepaper.tex --source2 blockchain_security.tex --combine smart_merge --analysis comprehensive
    
    source = None
    source2 = None
    combine_strategy = "smart_merge"
    analysis = None
    
    if len(sys.argv) > 1:
        if '--source' in sys.argv:
            idx = sys.argv.index('--source')
            if idx + 1 < len(sys.argv):
                source = sys.argv[idx + 1]
        
        if '--source2' in sys.argv:
            idx = sys.argv.index('--source2')
            if idx + 1 < len(sys.argv):
                source2 = sys.argv[idx + 1]
        
        if '--combine' in sys.argv:
            idx = sys.argv.index('--combine')
            if idx + 1 < len(sys.argv):
                combine_strategy = sys.argv[idx + 1]
        
        if '--analysis' in sys.argv:
            idx = sys.argv.index('--analysis')
            if idx + 1 < len(sys.argv):
                analysis = sys.argv[idx + 1]
    
    main(source=source, source2=source2, combine_strategy=combine_strategy, analysis=analysis)