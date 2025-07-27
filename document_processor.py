"""
Advanced Document Processing System
Complete class-based implementation for document refactoring and analysis
"""

import os
import re
import json
import yaml
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentLoader:
    """Handle document loading from various sources"""
    
    def __init__(self):
        self.supported_formats = ['.tex', '.txt', '.md']
    
    def load_document(self, source: str) -> str:
        """Load document from file path, URL, or arXiv"""
        if source.startswith('http'):
            return self._download_from_url(source)
        elif source.startswith('arxiv:'):
            return self._download_from_arxiv(source.replace('arxiv:', ''))
        else:
            return self._load_from_file(source)
    
    def _load_from_file(self, file_path: str) -> str:
        """Load document from local file"""
        encodings = ['utf-8', 'latin-1', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file {file_path}")
    
    def _download_from_url(self, url: str) -> str:
        """Download document from URL"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    
    def _download_from_arxiv(self, arxiv_id: str) -> str:
        """Download from arXiv"""
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        return self._download_from_url(url)

class DocumentChunker:
    """Intelligent document chunking with LaTeX awareness"""
    
    def __init__(self):
        self.latex_environments = [
            'equation', 'align', 'gather', 'multline', 'split',
            'enumerate', 'itemize', 'verbatim', 'lstlisting',
            'figure', 'table', 'abstract'
        ]
    
    def extract_chunks(self, content: str) -> List[Dict[str, Any]]:
        """Extract structured chunks from document"""
        chunks = []
        
        # Extract sections first
        sections = self._extract_sections(content)
        
        for section_name, section_content in sections.items():
            section_chunks = self._chunk_section(section_content, section_name)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from LaTeX document"""
        sections = {}
        
        # Pattern for section markers
        section_pattern = r'% --- (.+?) ---\n(.*?)(?=% --- |$)'
        matches = re.findall(section_pattern, content, re.DOTALL)
        
        for section_name, section_content in matches:
            sections[section_name.strip()] = section_content.strip()
        
        # If no section markers, treat as single section
        if not sections:
            sections['Main Content'] = content
        
        return sections
    
    def _chunk_section(self, content: str, section_name: str) -> List[Dict[str, Any]]:
        """Chunk individual section content"""
        chunks = []
        
        # Split by LaTeX environments
        env_pattern = r'(\\begin\{(' + '|'.join(self.latex_environments) + r')\}.*?\\end\{\2\})'
        parts = re.split(env_pattern, content, flags=re.DOTALL)
        
        chunk_id = 0
        for i, part in enumerate(parts):
            if part.strip():
                chunk_type = self._classify_content(part)
                chunks.append({
                    'id': chunk_id,
                    'content': part.strip(),
                    'type': chunk_type,
                    'parent_section': section_name,
                    'position': i
                })
                chunk_id += 1
        
        return chunks
    
    def _classify_content(self, content: str) -> str:
        """Classify chunk content type"""
        if re.search(r'\\begin\{equation\}', content):
            return 'equation'
        elif re.search(r'\\begin\{enumerate\}|\\begin\{itemize\}', content):
            return 'list'
        elif re.search(r'\\begin\{verbatim\}|\\begin\{lstlisting\}', content):
            return 'code'
        elif re.search(r'\\begin\{figure\}|\\begin\{table\}', content):
            return 'figure'
        else:
            return 'paragraph'

class LLMHandler:
    """Handle LLM interactions with multiple providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('provider', 'mistral')
        self.model = config.get('model', 'mistral-small-latest')
        self.api_keys = {
            'mistral': os.getenv('MISTRAL_API_KEY'),
            'openai': os.getenv('OPENAI_API_KEY'),
            'huggingface': os.getenv('HUGGINGFACE_TOKEN')
        }
        self.context_memory = {}
    
    def process_content(self, content: str, prompt: str, section_name: str = "") -> str:
        """Process content with LLM"""
        system_prompt = self._get_system_prompt()
        full_prompt = self._build_contextual_prompt(content, prompt, section_name)
        
        try:
            response = self._call_llm(full_prompt, system_prompt)
            self._update_context(section_name, response)
            return self._post_process_response(response)
        except Exception as e:
            print(f"LLM processing error: {e}")
            return content  # Return original on error
    
    def _get_system_prompt(self) -> str:
        """Get format-specific system prompt"""
        output_format = self.config.get('output_format', 'latex')
        
        if output_format == 'latex':
            return """You are a LaTeX expert. Output ONLY valid LaTeX content.
STRICT RULES:
- Use \\section{Title} for sections, NOT ### Title
- Use \\textbf{text} for bold, NOT **text**
- Use \\begin{itemize} for lists, NOT - item
- Preserve all equations exactly
- NO Markdown syntax allowed
- NO document structure (\\documentclass, \\begin{document})"""
        else:
            return "You are a technical writer. Maintain the original format and structure."
    
    def _build_contextual_prompt(self, content: str, prompt: str, section_name: str) -> str:
        """Build prompt with context"""
        context_parts = []
        
        if section_name in self.context_memory:
            context_parts.append(f"Previous context: {self.context_memory[section_name]}")
        
        context = "\n".join(context_parts)
        
        return f"{context}\n\nTASK: {prompt}\n\nCONTENT:\n{content}"
    
    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Call LLM API"""
        if self.provider == 'mistral':
            return self._call_mistral(prompt, system_prompt)
        elif self.provider == 'openai':
            return self._call_openai(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_mistral(self, prompt: str, system_prompt: str) -> str:
        """Call Mistral API"""
        api_key = self.api_keys['mistral']
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set")
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.get('max_tokens', 2048),
                "temperature": self.config.get('temperature', 0.2)
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API Error: {response.status_code}")
    
    def _call_openai(self, prompt: str, system_prompt: str) -> str:
        """Call OpenAI API"""
        api_key = self.api_keys['openai']
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.config.get('max_tokens', 2048),
                "temperature": self.config.get('temperature', 0.2)
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API Error: {response.status_code}")
    
    def _post_process_response(self, response: str) -> str:
        """Post-process LLM response for format consistency"""
        # Fix common Markdown to LaTeX issues
        response = re.sub(r'^#{3}\s*(.+)$', r'\\section{\1}', response, flags=re.MULTILINE)
        response = re.sub(r'^#{4}\s*(.+)$', r'\\subsection{\1}', response, flags=re.MULTILINE)
        response = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', response)
        response = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', response)
        
        return response
    
    def _update_context(self, section_name: str, content: str):
        """Update context memory"""
        if section_name:
            summary = content[:200] + "..." if len(content) > 200 else content
            self.context_memory[section_name] = summary

class OutputHandler:
    """Handle document output and analysis"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = os.path.join(output_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        
        self.processing_log = []
        self.chunk_tracking = {}
    
    def save_section(self, section_name: str, content: str) -> str:
        """Save individual section"""
        safe_name = re.sub(r'[^\w\-_\.]', '_', section_name)
        file_path = os.path.join(self.session_path, f"{safe_name}.tex")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.log(f"Saved section: {section_name} -> {file_path}")
        return file_path
    
    def assemble_document(self, sections: Dict[str, str], template: str = "academic") -> str:
        """Assemble final document"""
        document_parts = []
        
        # Document header
        document_parts.extend([
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{amsmath}",
            "\\usepackage{graphicx}",
            "",
            "\\title{Processed Document}",
            "\\author{Document Processing System}",
            "\\date{\\today}",
            "",
            "\\begin{document}",
            "\\maketitle",
            ""
        ])
        
        # Add sections in order
        section_order = self._get_section_order(template)
        
        for section_name in section_order:
            if section_name in sections and sections[section_name].strip():
                document_parts.append(sections[section_name])
                document_parts.append("")
        
        document_parts.append("\\end{document}")
        
        # Save final document
        final_content = "\n".join(document_parts)
        final_path = os.path.join(self.session_path, "final_document.tex")
        
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        self.log(f"Final document saved: {final_path}")
        return final_path
    
    def _get_section_order(self, template: str) -> List[str]:
        """Get section order for template"""
        templates = {
            "academic": ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion"],
            "bitcoin": ["Summary", "Abstract", "Introduction", "Transactions", "Timestamp Server", 
                       "Proof-of-Work", "Network", "Incentive", "Calculations", "Conclusion"]
        }
        return templates.get(template, templates["academic"])
    
    def track_chunk(self, chunk_id: int, original_section: str, target_section: str, content_preview: str):
        """Track chunk transformation"""
        self.chunk_tracking[chunk_id] = {
            'original_section': original_section,
            'target_section': target_section,
            'content_preview': content_preview[:100] + "..."
        }
    
    def generate_analysis_report(self, original_file: str, final_file: str) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        analysis = {
            'processing_summary': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'total_chunks': len(self.chunk_tracking),
                'processing_steps': len(self.processing_log)
            },
            'content_analysis': self._analyze_content_preservation(original_file, final_file),
            'chunk_tracking': self.chunk_tracking,
            'processing_log': self.processing_log
        }
        
        # Save analysis report
        report_path = os.path.join(self.session_path, "analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _analyze_content_preservation(self, original_file: str, final_file: str) -> Dict[str, Any]:
        """Analyze content preservation"""
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(final_file, 'r', encoding='utf-8') as f:
                final = f.read()
            
            # Clean content for comparison
            original_clean = self._clean_content(original)
            final_clean = self._clean_content(final)
            
            # Count LaTeX environments
            equations_orig = len(re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', original, re.DOTALL))
            equations_final = len(re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', final, re.DOTALL))
            
            return {
                'character_count': {
                    'original': len(original_clean),
                    'final': len(final_clean),
                    'change_percent': ((len(final_clean) - len(original_clean)) / len(original_clean)) * 100
                },
                'equations_preserved': {
                    'original': equations_orig,
                    'final': equations_final,
                    'preserved': equations_orig == equations_final
                },
                'content_quality': 'preserved' if abs(len(final_clean) - len(original_clean)) < len(original_clean) * 0.1 else 'modified'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _clean_content(self, content: str) -> str:
        """Clean content for analysis"""
        # Remove LaTeX structure
        content = re.sub(r'\\documentclass.*?\\begin\{document\}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\end\{document\}', '', content)
        content = re.sub(r'\\title\{.*?\}', '', content)
        content = re.sub(r'% .+', '', content)  # Remove comments
        return re.sub(r'\n\s*\n', '\n', content).strip()
    
    def log(self, message: str):
        """Add entry to processing log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        print(log_entry)

class DocumentProcessor:
    """Main document processing orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.llm_handler = LLMHandler(self.config['llm'])
        self.output_handler = OutputHandler(self.config['output']['directory'])
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default config
            return {
                'llm': {
                    'provider': 'mistral',
                    'model': 'mistral-small-latest',
                    'max_tokens': 2048,
                    'temperature': 0.2,
                    'output_format': 'latex'
                },
                'processing': {
                    'template': 'academic',
                    'enable_enhancement': True,
                    'chunk_strategy': 'semantic'
                },
                'output': {
                    'directory': 'outputs',
                    'format': 'latex',
                    'generate_analysis': True
                }
            }
    
    def process_document(self, source: str, template: Optional[str] = None) -> Dict[str, Any]:
        """Process single document"""
        self.output_handler.log(f"Starting document processing: {source}")
        
        # Load document
        content = self.loader.load_document(source)
        self.output_handler.log(f"Loaded document: {len(content)} characters")
        
        # Extract chunks
        chunks = self.chunker.extract_chunks(content)
        self.output_handler.log(f"Extracted {len(chunks)} chunks")
        
        # Group chunks by section
        sections = self._group_chunks_by_section(chunks)
        
        # Process sections with LLM
        processed_sections = {}
        template_name = template or self.config['processing']['template']
        
        for section_name, section_chunks in sections.items():
            if section_chunks:
                self.output_handler.log(f"Processing section: {section_name}")
                
                # Combine chunk content
                combined_content = "\n\n".join([chunk['content'] for chunk in section_chunks])
                
                # Get section prompt
                prompt = self._get_section_prompt(section_name, template_name)
                
                # Process with LLM
                if self.config['processing']['enable_enhancement']:
                    processed_content = self.llm_handler.process_content(
                        combined_content, prompt, section_name
                    )
                else:
                    processed_content = combined_content
                
                processed_sections[section_name] = processed_content
                
                # Save individual section
                self.output_handler.save_section(section_name, processed_content)
                
                # Track chunks
                for chunk in section_chunks:
                    self.output_handler.track_chunk(
                        chunk['id'], chunk['parent_section'], 
                        section_name, chunk['content']
                    )
        
        # Assemble final document
        final_path = self.output_handler.assemble_document(processed_sections, template_name)
        
        # Generate analysis report
        analysis = self.output_handler.generate_analysis_report(source, final_path)
        
        return {
            'final_document': final_path,
            'session_path': self.output_handler.session_path,
            'analysis': analysis,
            'processed_sections': len(processed_sections)
        }
    
    def _group_chunks_by_section(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by their parent section"""
        sections = {}
        for chunk in chunks:
            section = chunk['parent_section']
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk)
        return sections
    
    def _get_section_prompt(self, section_name: str, template: str) -> str:
        """Get processing prompt for section"""
        prompts = {
            'Abstract': 'Rewrite as a concise abstract summarizing the main contributions.',
            'Introduction': 'Rewrite as a compelling introduction that motivates the work.',
            'Methodology': 'Describe the methodology clearly and systematically.',
            'Results': 'Present the results clearly with proper analysis.',
            'Discussion': 'Discuss the implications and significance of the results.',
            'Conclusion': 'Summarize the main contributions and future work.'
        }
        
        return prompts.get(section_name, 'Process this content appropriately for the document.')