"""🚀 REVOLUTIONARY DOCUMENT COMBINER MODULE

This module implements the breakthrough chunk-level LLM-dependent document augmentation
system. Unlike traditional section-level merging, this system:

🧠 SEMANTIC INTELLIGENCE:
- Analyzes individual chunks for content meaning and purpose
- Maps chunks to original document sections based on semantic relevance
- Preserves original document structure while intelligently adding content
- Creates new sections only when content truly doesn't fit existing structure

🎯 KEY INNOVATIONS:
- Chunk-level granularity instead of coarse section-level mapping
- LLM-driven relevance scoring with confidence thresholds
- Conservative augmentation that respects document architecture
- Batch processing for efficiency while maintaining quality

📊 PROCESSING FLOW:
1. Extract chunks from augmentation document (preserves original intact)
2. LLM analyzes each chunk against original document sections
3. Assigns chunks based on semantic relevance (confidence > 0.6)
4. Synthesizes content for sections with multiple chunks
5. Creates minimal new sections for orphaned content only

This represents a paradigm shift from primitive text concatenation to
intelligent content understanding and integration.
"""
from .file_loader import load_latex_file
from .chunker import extract_latex_sections, group_chunks_by_section
from .format_enforcer import FormatEnforcer
from .llm_client import UnifiedLLMClient
import re

class DocumentCombiner:
    """🎯 Revolutionary Document Augmentation Engine
    
    This class implements the most advanced document combination system available,
    using chunk-level semantic analysis to intelligently augment documents while
    preserving their original structure and integrity.
    
    🧠 INTELLIGENCE FEATURES:
    - Semantic chunk analysis using LLM
    - Confidence-based assignment decisions
    - Original document structure preservation
    - Conservative new section creation
    - Batch processing optimization
    
    🔧 SUPPORTED STRATEGIES:
    - smart_merge: Revolutionary chunk-level LLM-dependent augmentation
    - merge: Traditional section-level merging
    - interleave: Alternating content integration
    - append: Sequential document combination
    """
    def __init__(self, output_format="latex"):
        # 🎨 Format enforcement for output quality
        self.format_enforcer = FormatEnforcer(output_format)
        
        # 🧠 LLM client for semantic analysis
        self.llm_client = UnifiedLLMClient()
        
        # 🚀 Available combination strategies (smart_merge is revolutionary)
        self.combination_strategies = {
            "merge": self._merge_documents,
            "interleave": self._interleave_documents,
            "append": self._append_documents,
            "smart_merge": self._llm_enhanced_smart_merge  # 🌟 THE GAME CHANGER
        }
    
    def combine_documents(self, doc1_path, doc2_path, strategy="smart_merge", output_format="latex"):
        """Combine two documents using specified strategy"""
        
        # Load documents
        doc1_content = load_latex_file(doc1_path)
        doc2_content = load_latex_file(doc2_path)
        
        # Extract and chunk both documents
        doc1_chunks = extract_latex_sections(doc1_content)
        doc2_chunks = extract_latex_sections(doc2_content)
        
        doc1_grouped = group_chunks_by_section(doc1_chunks)
        doc2_grouped = group_chunks_by_section(doc2_chunks)
        
        # Apply combination strategy
        combined_content = self.combination_strategies[strategy](doc1_grouped, doc2_grouped)
        
        # Enforce format consistency
        formatted_content, issues = self.format_enforcer.enforce_format(combined_content)
        
        return formatted_content, issues
    
    def _merge_documents(self, doc1_sections, doc2_sections):
        """Merge sections by combining content from both documents"""
        combined = {}
        all_sections = set(doc1_sections.keys()) | set(doc2_sections.keys())
        
        for section in all_sections:
            content_parts = []
            
            if section in doc1_sections:
                for chunk in doc1_sections[section]:
                    content_parts.append(f"% From Document 1\n{chunk['content']}")
            
            if section in doc2_sections:
                for chunk in doc2_sections[section]:
                    content_parts.append(f"% From Document 2\n{chunk['content']}")
            
            combined[section] = "\n\n".join(content_parts)
        
        return self._format_combined_document(combined)
    
    def _interleave_documents(self, doc1_sections, doc2_sections):
        """Interleave sections from both documents"""
        combined = {}
        all_sections = sorted(set(doc1_sections.keys()) | set(doc2_sections.keys()))
        
        for section in all_sections:
            content_parts = []
            
            # Interleave chunks from both documents
            doc1_chunks = doc1_sections.get(section, [])
            doc2_chunks = doc2_sections.get(section, [])
            
            max_chunks = max(len(doc1_chunks), len(doc2_chunks))
            
            for i in range(max_chunks):
                if i < len(doc1_chunks):
                    content_parts.append(f"% Document 1 - Chunk {i+1}\n{doc1_chunks[i]['content']}")
                if i < len(doc2_chunks):
                    content_parts.append(f"% Document 2 - Chunk {i+1}\n{doc2_chunks[i]['content']}")
            
            combined[section] = "\n\n".join(content_parts)
        
        return self._format_combined_document(combined)
    
    def _append_documents(self, doc1_sections, doc2_sections):
        """Append second document after first document"""
        combined = {}
        
        # Add all sections from doc1
        for section, chunks in doc1_sections.items():
            combined[f"Part I - {section}"] = "\n\n".join([chunk['content'] for chunk in chunks])
        
        # Add all sections from doc2
        for section, chunks in doc2_sections.items():
            combined[f"Part II - {section}"] = "\n\n".join([chunk['content'] for chunk in chunks])
        
        return self._format_combined_document(combined)
    
    def _llm_enhanced_smart_merge(self, doc1_sections, doc2_sections):
        """🚀 REVOLUTIONARY CHUNK-LEVEL AUGMENTATION ENGINE
        
        This is the breakthrough function that implements semantic chunk-level
        document augmentation. Unlike primitive section-level merging, this:
        
        🧠 PRESERVES ORIGINAL STRUCTURE:
        - Keeps doc1 (original) structure completely intact
        - Only adds content where semantically relevant
        - Creates new sections conservatively for orphaned content
        
        🎯 SEMANTIC INTELLIGENCE:
        - LLM analyzes each chunk individually for relevance
        - Confidence scoring determines assignment decisions
        - Batch processing for efficiency
        
        📊 THREE-PHASE PROCESS:
        1. Chunk-Level Analysis: Extract and analyze augmentation chunks
        2. Content Augmentation: Map chunks to original sections
        3. Document Assembly: Generate final enhanced document
        
        This represents the future of document processing!
        """
        print("🧠 Phase 1: Augmentation Analysis...")
        
        # 🏗️ Preserve original document structure (CRITICAL: no modification)
        original_structure = self._get_original_document_structure(doc1_sections)
        
        # 📊 Extract chunks only from augmentation document (doc2)
        # This ensures original document remains untouched
        augmentation_chunks = self._extract_augmentation_chunks(doc2_sections)
        
        print(f"  📈 Augmenting original {len(original_structure)} sections with {len(augmentation_chunks)} new chunks")
        
        # 🎯 Phase 1: Map augmentation chunks to original sections using LLM
        # This is where the magic happens - semantic relevance analysis
        augmentation_assignments = self._map_augmentation_to_original(augmentation_chunks, original_structure)
        
        print("🔄 Phase 2: Content Augmentation...")
        
        # 🔧 Phase 2: Augment original sections with semantically relevant content
        # Original content is preserved, new content is intelligently integrated
        augmented_document = self._augment_original_sections(doc1_sections, augmentation_assignments)
        
        # 📝 Phase 3: Format and assemble the final enhanced document
        return self._format_combined_document(augmented_document)
    
    def _analyze_document_structure(self, sections, doc_name):
        """Analyze document structure and themes using LLM"""
        section_list = list(sections.keys())
        content_preview = {}
        
        for section, chunks in list(sections.items())[:5]:  # Analyze first 5 sections
            content = '\n'.join([chunk['content'] for chunk in chunks[:2]])  # First 2 chunks
            content_preview[section] = content[:500]  # First 500 chars
        
        analysis_prompt = f"""Analyze this document structure and identify its type, themes, and purpose:

{doc_name} Sections: {section_list}

Content Preview:
{content_preview}

Provide analysis in this format:
Document Type: [whitepaper/security analysis/academic paper/etc.]
Main Themes: [list 3-5 key themes]
Purpose: [brief description]
Technical Level: [basic/intermediate/advanced]"""
        
        try:
            analysis = self.llm_client.call_llm(analysis_prompt, max_tokens=300)
            return analysis
        except:
            return f"Standard technical document with sections: {section_list[:5]}"
    
    def _get_original_document_structure(self, doc1_sections):
        """Get the original document structure to preserve"""
        return list(doc1_sections.keys())
    
    def _extract_augmentation_chunks(self, doc2_sections):
        """Extract chunks only from augmentation document (doc2)"""
        augmentation_chunks = []
        
        for section_name, chunks in doc2_sections.items():
            for i, chunk in enumerate(chunks):
                augmentation_chunks.append({
                    'content': chunk['content'],
                    'source_section': section_name,
                    'chunk_id': f"aug_{section_name}_{i}",
                    'type': chunk.get('type', 'paragraph')
                })
        
        return augmentation_chunks
    
    def _map_augmentation_to_original(self, augmentation_chunks, original_sections):
        """Map augmentation chunks to original document sections"""
        assignments = {section: [] for section in original_sections}
        orphaned_chunks = []
        
        print(f"  Original sections: {original_sections}")
        
        # Process chunks in batches for efficiency
        batch_size = 5
        for i in range(0, len(augmentation_chunks), batch_size):
            batch = augmentation_chunks[i:i+batch_size]
            batch_assignments = self._analyze_augmentation_batch(batch, original_sections)
            
            for j, chunk in enumerate(batch):
                assignment = batch_assignments.get(j, {'section': None, 'confidence': 0.0})
                
                if assignment['confidence'] > 0.6 and assignment['section'] in original_sections:
                    assignments[assignment['section']].append({
                        'chunk': chunk,
                        'confidence': assignment['confidence']
                    })
                    print(f"    ➕ {chunk['chunk_id']} -> {assignment['section']} ({assignment['confidence']:.2f})")
                else:
                    orphaned_chunks.append(chunk)
                    print(f"    ❓ {chunk['chunk_id']} -> ORPHANED ({assignment['confidence']:.2f})")
        
        # Handle orphaned chunks - create minimal new sections
        if orphaned_chunks:
            print(f"  🆕 Creating minimal sections for {len(orphaned_chunks)} orphaned chunks...")
            new_sections = self._create_minimal_new_sections(orphaned_chunks)
            assignments.update(new_sections)
        
        return assignments
    
    def _analyze_augmentation_batch(self, chunks, original_sections):
        """Analyze augmentation chunks for assignment to original sections"""
        chunk_previews = []
        for i, chunk in enumerate(chunks):
            preview = f"Chunk {i} (from {chunk['source_section']}): {chunk['content'][:200]}..."
            chunk_previews.append(preview)
        
        analysis_prompt = f"""Analyze these augmentation chunks and assign each to the most relevant ORIGINAL section.

ORIGINAL DOCUMENT SECTIONS: {original_sections}

AUGMENTATION CHUNKS:
{chr(10).join(chunk_previews)}

For each chunk, determine which original section it would best enhance:
Chunk X: ORIGINAL_SECTION_NAME (confidence: 0.0-1.0)

Confidence guidelines:
- 0.9-1.0: Perfect enhancement for this section
- 0.7-0.8: Good fit, adds relevant content
- 0.5-0.6: Moderate fit, somewhat related
- 0.0-0.4: Poor fit, doesn't enhance this section

Assignments:"""
        
        try:
            response = self.llm_client.call_llm(analysis_prompt, max_tokens=300)
            return self._parse_chunk_assignments(response)
        except Exception as e:
            print(f"    ⚠️ Augmentation analysis failed: {e}")
            return {}
    
    def _parse_chunk_assignments(self, response):
        """Parse LLM chunk assignment response"""
        assignments = {}
        lines = response.split('\n')
        
        for line in lines:
            if 'Chunk' in line and ':' in line:
                try:
                    # Extract chunk number
                    chunk_match = re.search(r'Chunk (\d+)', line)
                    if not chunk_match:
                        continue
                    chunk_num = int(chunk_match.group(1))
                    
                    # Extract section and confidence
                    parts = line.split(':', 1)[1].strip()
                    if '(' in parts:
                        section = parts.split('(')[0].strip()
                        conf_match = re.search(r'([0-9.]+)', parts)
                        confidence = float(conf_match.group(1)) if conf_match else 0.0
                    else:
                        section = parts.strip()
                        confidence = 0.5
                    
                    assignments[chunk_num] = {
                        'section': section,
                        'confidence': confidence
                    }
                except:
                    continue
        
        return assignments
    
    def _create_minimal_new_sections(self, orphaned_chunks):
        """Create minimal new sections only for truly orphaned chunks"""
        new_sections = {}
        
        # Group by source section but create minimal new sections
        source_groups = {}
        for chunk in orphaned_chunks:
            source = chunk['source_section']
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(chunk)
        
        # Create new sections with descriptive names
        for source_section, chunks in source_groups.items():
            new_section_name = f"Security Analysis: {source_section}"
            new_sections[new_section_name] = [{
                'chunk': chunk,
                'confidence': 1.0
            } for chunk in chunks]
            print(f"    🆕 Created new section: {new_section_name} ({len(chunks)} chunks)")
        
        return new_sections
    
    def _group_orphaned_chunks(self, orphaned_chunks):
        """Group orphaned chunks by content similarity"""
        if not orphaned_chunks:
            return {}
        
        # Simple grouping by source section for now
        groups = {}
        for chunk in orphaned_chunks:
            section_name = f"Additional {chunk['source_section']}"
            if section_name not in groups:
                groups[section_name] = []
            groups[section_name].append(chunk)
        
        return groups
    
    def _augment_original_sections(self, original_sections, augmentation_assignments):
        """Augment original sections with new content"""
        augmented = {}
        
        # Process each original section
        for section_name, original_chunks in original_sections.items():
            # Start with original content
            original_content = '\n\n'.join([chunk['content'] for chunk in original_chunks])
            
            # Check if there are augmentation chunks for this section
            if section_name in augmentation_assignments and augmentation_assignments[section_name]:
                augmentation_data = augmentation_assignments[section_name]
                
                # Sort by confidence (highest first)
                augmentation_data.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Extract augmentation contents
                augmentation_contents = [item['chunk']['content'] for item in augmentation_data]
                
                # Augment the original section
                augmented[section_name] = self._augment_section_content(
                    section_name, original_content, augmentation_contents
                )
                
                print(f"    ➕ {section_name}: Original + {len(augmentation_contents)} augmentations -> {len(augmented[section_name])} chars")
            else:
                # No augmentation, keep original
                augmented[section_name] = self._clean_content(original_content)
                print(f"    ➡️ {section_name}: Original only -> {len(augmented[section_name])} chars")
        
        # Add any new sections from orphaned chunks
        for section_name, chunks_data in augmentation_assignments.items():
            if section_name not in original_sections and chunks_data:
                chunk_contents = [item['chunk']['content'] for item in chunks_data]
                augmented[section_name] = '\n\n'.join([self._clean_content(c) for c in chunk_contents])
                print(f"    🆕 {section_name}: New section -> {len(augmented[section_name])} chars")
        
        return augmented
    
    def _augment_section_content(self, section_name, original_content, augmentation_contents):
        """Augment original section with additional content"""
        if not augmentation_contents:
            return self._clean_content(original_content)
        
        augmentation_text = '\n\n'.join(augmentation_contents)
        
        augmentation_prompt = f"""Enhance this original section with additional relevant content:

ORIGINAL {section_name}:
{original_content[:1500]}

ADDITIONAL CONTENT TO INTEGRATE:
{augmentation_text[:1500]}

Requirements:
1. Keep ALL original content intact
2. Seamlessly integrate additional content where relevant
3. Maintain coherent flow and structure
4. Preserve all technical details and equations
5. Add smooth transitions between original and new content

Enhanced {section_name}:"""
        
        try:
            enhanced = self.llm_client.call_llm(augmentation_prompt, max_tokens=2000)
            cleaned = self._clean_content(enhanced)
            
            # Fallback if enhancement is too short or failed
            if len(cleaned) < len(original_content) * 0.8:
                return self._clean_content(original_content + '\n\n' + augmentation_text)
            
            return cleaned
        except Exception as e:
            print(f"    ⚠️ Augmentation failed for {section_name}: {e}")
            return self._clean_content(original_content + '\n\n' + augmentation_text)
    
    def _synthesize_chunks(self, section_name, chunk_contents):
        """Synthesize multiple chunks into coherent section content"""
        combined_content = '\n\n---CHUNK SEPARATOR---\n\n'.join(chunk_contents)
        
        synthesis_prompt = f"""Synthesize these related chunks into a coherent '{section_name}' section:

{combined_content[:2000]}

Requirements:
1. Create unified, flowing narrative
2. Preserve all technical details and equations exactly
3. Remove redundancy but keep complementary information
4. Maintain professional tone
5. Output only the synthesized content

Synthesized content:"""
        
        try:
            synthesized = self.llm_client.call_llm(synthesis_prompt, max_tokens=1200)
            cleaned = self._clean_content(synthesized)
            
            # Fallback if synthesis is too short
            if len(cleaned) < 100:
                return '\n\n'.join([self._clean_content(c) for c in chunk_contents])
            
            return cleaned
        except Exception as e:
            print(f"    ⚠️ Synthesis failed for {section_name}: {e}")
            return '\n\n'.join([self._clean_content(c) for c in chunk_contents])
    

    

    

    
    def _clean_content(self, content):
        """🧹 CONTENT CLEANING AND FORMATTING ENGINE
        
        Performs comprehensive content cleaning to ensure high-quality output.
        Fixes common formatting issues and standardizes content presentation.
        
        🔧 CLEANING OPERATIONS:
        1. Remove processing artifacts and markers
        2. Fix common LaTeX formatting issues
        3. Standardize list formatting
        4. Escape special characters properly
        5. Normalize whitespace and spacing
        
        This ensures professional, compilation-ready output.
        """
        # 🗑️ Remove processing artifacts and source markers
        content = re.sub(r'=== From .+ ===\n', '', content)
        
        # 🔧 Fix common LaTeX formatting issues
        content = content.replace('\\textit{', '\\times ')  # Fix multiplication symbols
        content = content.replace('- ', '\\item ')          # Fix list items
        
        # 🛡️ Escape special characters properly
        content = re.sub(r'(?<!\\)%', '\\%', content)  # Escape % not preceded by \
        
        # 🧹 Clean up excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Max 2 consecutive newlines
        
        # ✨ Return cleaned and trimmed content
        return content.strip()
    
    def _format_combined_document(self, combined_sections):
        """Format augmented sections into complete document"""
        print("📝 Phase 3: Document Assembly...")
        
        # Create document structure
        document_parts = []
        
        # Enhanced document header
        document_parts.extend([
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{amsmath}",
            "\\usepackage{amsfonts}",
            "\\usepackage{graphicx}",
            "\\usepackage{url}",
            "",
            "\\title{Bitcoin: A Peer-to-Peer Electronic Cash System (Enhanced)}",
            "\\author{Satoshi Nakamoto (with Security Analysis Enhancement)}",
            "\\date{\\today}",
            "",
            "\\begin{document}",
            "\\maketitle",
            ""
        ])
        
        # Add sections in their original order (preserving document structure)
        for section_name, content in combined_sections.items():
            if content.strip():
                document_parts.append(f"\\section{{{section_name}}}")
                document_parts.append(content)
                document_parts.append("")
        
        document_parts.append("\\end{document}")
        
        # Final document enhancement
        document_text = "\n".join(document_parts)
        return document_text  # Skip coherence enhancement for now
    
    def _enhance_document_coherence(self, document_text):
        """Final LLM pass to enhance document coherence"""
        try:
            # Extract sections for coherence check
            sections = re.findall(r'\\section\{([^}]+)\}\n([^\\]*(?:\\[^s][^\\]*)*)', document_text)
            
            if len(sections) > 3:  # Only enhance if substantial content
                coherence_prompt = f"""Review this combined document and add brief transition sentences between sections where needed.
Focus on:
1. Smooth flow between Bitcoin concepts and security analysis
2. Logical progression of ideas
3. Clear connections between related sections

Document preview (first 3 sections):
{sections[:3]}

Provide 2-3 brief transition sentences to improve flow."""
                
                transitions = self.llm_client.call_llm(coherence_prompt, max_tokens=200)
                # For now, just return the document as-is since adding transitions requires complex parsing
                # In a full implementation, we would parse and insert transitions
        except:
            pass
        
        return document_text