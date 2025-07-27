"""
Advanced Document Processing System - Configuration-Driven Version
All hardcoded values moved to configuration files
"""

import os
import re
import json
import yaml
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.patterns = self.config.get('patterns', {})
        self.prompts = self.config.get('prompts', {})
        self.api_configs = self.config.get('api_configs', {})
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_pattern(self, pattern_name: str) -> str:
        """Get regex pattern from config"""
        return self.patterns.get(pattern_name, '')
    
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get prompt template from config with variable substitution"""
        prompt_template = self.prompts.get(prompt_name, '')
        return prompt_template.format(**kwargs)
    
    def get_api_config(self, provider: str) -> Dict[str, Any]:
        """Get API configuration for provider"""
        return self.api_configs.get(provider, {})

class DocumentLoader:
    """Handle document loading from various sources"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.supported_formats = self.config.config.get('supported_formats', ['.tex', '.txt', '.md'])
    
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
        encodings = self.config.config.get('file_encodings', ['utf-8', 'latin-1', 'ascii'])
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file {file_path}")
    
    def _download_from_url(self, url: str) -> str:
        """Download document from URL"""
        timeout = self.config.config.get('download_timeout', 30)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    
    def _download_from_arxiv(self, arxiv_id: str) -> str:
        """Download from arXiv"""
        arxiv_url_template = self.config.config.get('arxiv_url_template', 'https://arxiv.org/e-print/{arxiv_id}')
        url = arxiv_url_template.format(arxiv_id=arxiv_id)
        return self._download_from_url(url)

class DocumentChunker:
    """Intelligent document chunking with configurable patterns"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.latex_environments = self.config.config.get('latex_environments', [])
        self.content_classifiers = self.config.config.get('content_classifiers', {})
    
    def extract_chunks(self, content: str) -> List[Dict[str, Any]]:
        """Extract structured chunks from document"""
        chunks = []
        
        # Extract sections using configurable pattern
        sections = self._extract_sections(content)
        
        for section_name, section_content in sections.items():
            section_chunks = self._chunk_section(section_content, section_name)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections using configurable pattern"""
        sections = {}
        
        # Get section pattern from config
        section_pattern = self.config.get_pattern('section_extraction')
        if not section_pattern:
            sections['Main Content'] = content
            return sections
        
        matches = re.findall(section_pattern, content, re.DOTALL)
        
        for section_name, section_content in matches:
            sections[section_name.strip()] = section_content.strip()
        
        # If no sections found, treat as single section
        if not sections:
            sections['Main Content'] = content
        
        return sections
    
    def _chunk_section(self, content: str, section_name: str) -> List[Dict[str, Any]]:
        """Chunk individual section content using configurable patterns"""
        chunks = []
        
        # Build environment pattern from config
        if self.latex_environments:
            env_pattern = self.config.get_pattern('environment_extraction').format(
                environments='|'.join(self.latex_environments)
            )
            parts = re.split(env_pattern, content, flags=re.DOTALL)
        else:
            parts = [content]
        
        chunk_id = 0
        for i, part in enumerate(parts):
            if part and part.strip():
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
        """Classify chunk content type using configurable patterns"""
        for content_type, pattern in self.content_classifiers.items():
            if re.search(pattern, content):
                return content_type
        
        return self.config.config.get('default_content_type', 'paragraph')

class GenericLLMClient:
    """Generic LLM client that works with any API through configuration"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.api_configs = self.config.api_configs
    
    def call_llm(self, provider: str, prompt: str, system_prompt: str = "", **kwargs) -> str:
        """Generic LLM calling interface"""
        api_config = self.config.get_api_config(provider)
        if not api_config:
            raise ValueError(f"No configuration found for provider: {provider}")
        
        # Get API key from environment
        api_key_env = api_config.get('api_key_env')
        api_key = os.getenv(api_key_env) if api_key_env else None
        
        if not api_key:
            raise ValueError(f"API key not found for {provider}. Set {api_key_env}")
        
        # Build request using configuration
        request_data = self._build_request(api_config, prompt, system_prompt, api_key, **kwargs)
        
        # Make API call
        return self._make_api_call(api_config, request_data)
    
    def _build_request(self, api_config: Dict[str, Any], prompt: str, system_prompt: str, api_key: str, **kwargs) -> Dict[str, Any]:
        """Build API request using configuration template"""
        # Get request template from config
        request_template = api_config.get('request_template', {})
        
        # Build headers
        headers = {}
        header_template = api_config.get('headers', {})
        for key, value_template in header_template.items():
            headers[key] = value_template.format(api_key=api_key)
        
        # Build request body
        body = {}
        for key, value in request_template.items():
            if isinstance(value, str):
                # Handle template substitution
                if '{model}' in value:
                    body[key] = value.format(model=kwargs.get('model', api_config.get('default_model')))
                elif '{messages}' in value:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    body[key] = messages
                else:
                    body[key] = value
            elif key in ['max_tokens', 'temperature', 'timeout']:
                body[key] = kwargs.get(key, api_config.get(f'default_{key}', value))
            else:
                body[key] = value
        
        return {
            'url': api_config['url'],
            'headers': headers,
            'json': body,
            'timeout': kwargs.get('timeout', api_config.get('default_timeout', 30))
        }
    
    def _make_api_call(self, api_config: Dict[str, Any], request_data: Dict[str, Any]) -> str:
        """Make the actual API call"""
        try:
            response = requests.post(**request_data)
            
            if response.status_code == 200:
                # Extract response using configurable path
                response_data = response.json()
                response_path = api_config.get('response_path', ['choices', 0, 'message', 'content'])
                
                # Navigate response structure
                result = response_data
                for key in response_path:
                    if isinstance(key, int):
                        result = result[key]
                    else:
                        result = result.get(key, '')
                
                return result
            else:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

class LLMHandler:
    """Handle LLM interactions with configurable processing"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.llm_client = GenericLLMClient(config_manager)
        self.context_memory = {}
        self.post_processors = self._load_post_processors()
    
    def _load_post_processors(self) -> List[Tuple[str, str]]:
        """Load post-processing rules from config"""
        return self.config.config.get('post_processing_rules', [])
    
    def process_content(self, content: str, prompt_name: str, section_name: str = "", **kwargs) -> str:
        """Process content with LLM using configurable prompts"""
        # Get provider and model from config
        llm_config = self.config.config.get('llm', {})
        provider = llm_config.get('provider')
        model = llm_config.get('model')
        
        # Build prompts from config
        system_prompt = self.config.get_prompt('system_prompt', 
                                               output_format=llm_config.get('output_format', 'latex'))
        task_prompt = self.config.get_prompt(prompt_name, section_name=section_name, **kwargs)
        
        # Build contextual prompt
        full_prompt = self._build_contextual_prompt(content, task_prompt, section_name)
        
        try:
            # Call LLM
            response = self.llm_client.call_llm(
                provider=provider,
                prompt=full_prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=llm_config.get('max_tokens', 2048),
                temperature=llm_config.get('temperature', 0.2)
            )
            
            # Post-process response
            processed_response = self._post_process_response(response)
            
            # Update context
            self._update_context(section_name, processed_response)
            
            return processed_response
            
        except Exception as e:
            print(f"LLM processing error: {e}")
            return content  # Return original on error
    
    def _build_contextual_prompt(self, content: str, prompt: str, section_name: str) -> str:
        """Build prompt with context using configurable template"""
        context_template = self.config.get_prompt('context_template')
        
        context_parts = []
        if section_name in self.context_memory:
            context_parts.append(self.context_memory[section_name])
        
        context = "\n".join(context_parts)
        
        return context_template.format(
            context=context,
            task=prompt,
            content=content
        )
    
    def _post_process_response(self, response: str) -> str:
        """Post-process LLM response using configurable rules"""
        processed = response
        
        for pattern, replacement in self.post_processors:
            processed = re.sub(pattern, replacement, processed, flags=re.MULTILINE)
        
        return processed
    
    def _update_context(self, section_name: str, content: str):
        """Update context memory"""
        if section_name:
            max_context_length = self.config.config.get('max_context_length', 200)
            summary = content[:max_context_length] + "..." if len(content) > max_context_length else content
            self.context_memory[section_name] = summary

class OutputHandler:
    """Handle document output and analysis with configurable templates"""
    
    def __init__(self, config_manager: ConfigManager, output_dir: str = None):
        self.config = config_manager
        self.output_dir = output_dir or self.config.config.get('output', {}).get('directory', 'outputs')
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_path = os.path.join(self.output_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        
        self.processing_log = []
        self.chunk_tracking = {}
        self.analysis_patterns = self.config.config.get('analysis_patterns', {})
    
    def save_section(self, section_name: str, content: str) -> str:
        """Save individual section"""
        safe_name = re.sub(r'[^\w\-_\.]', '_', section_name)
        extension = self.config.config.get('output', {}).get('extension', '.tex')
        file_path = os.path.join(self.session_path, f"{safe_name}{extension}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.log(f"Saved section: {section_name} -> {file_path}")
        return file_path
    
    def assemble_document(self, sections: Dict[str, str], template_name: str = None) -> str:
        """Enhanced document assembly with dynamic template generation"""
        template_name = template_name or self.config.config.get('processing', {}).get('template', 'academic')
        
        # Get document template from config
        document_template = self.config.config.get('document_templates', {}).get(template_name, {})
        
        document_parts = []
        
        # Add header from template
        header = document_template.get('header', [])
        document_parts.extend(header)
        
        # Enhanced section ordering with dynamic template
        section_order = self._generate_dynamic_section_order(sections, document_template)
        
        # Add sections with proper LaTeX structure
        for section_name in section_order:
            if section_name in sections and sections[section_name].strip():
                # Add proper section header if not present
                content = sections[section_name]
                if not content.strip().startswith('\\section'):
                    # Clean section name for LaTeX
                    clean_name = re.sub(r'[^\w\s-]', '', section_name)
                    document_parts.append(f"\\section{{{clean_name}}}")
                
                document_parts.append(content)
                document_parts.append("")
        
        # Add footer from template (ensure single \end{document})
        footer = document_template.get('footer', [])
        # Remove any existing \end{document} from content
        final_content_so_far = "\n".join(document_parts)
        final_content_so_far = re.sub(r'\\end\{document\}.*$', '', final_content_so_far, flags=re.MULTILINE | re.DOTALL)
        
        document_parts = [final_content_so_far]
        document_parts.extend(footer)
        
        # Save final document
        final_content = "\n".join(document_parts)
        
        # Clean up structural issues
        final_content = self._clean_document_structure(final_content)
        
        extension = self.config.config.get('output', {}).get('extension', '.tex')
        final_path = os.path.join(self.session_path, f"final_document{extension}")
        
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        self.log(f"Final document saved: {final_path}")
        return final_path
    
    def _generate_dynamic_section_order(self, sections: Dict[str, str], template: Dict) -> List[str]:
        """Generate dynamic section order based on available sections"""
        # Get template order as base
        template_order = template.get('section_order', [])
        
        # Find sections that match template order
        ordered_sections = []
        remaining_sections = set(sections.keys())
        
        # First, add sections that match template order
        for template_section in template_order:
            for section in list(remaining_sections):
                if self._sections_match_for_ordering(template_section, section):
                    ordered_sections.append(section)
                    remaining_sections.remove(section)
                    break
        
        # Add remaining sections in alphabetical order
        ordered_sections.extend(sorted(remaining_sections))
        
        return ordered_sections
    
    def _sections_match_for_ordering(self, template_section: str, actual_section: str) -> bool:
        """Check if sections match for ordering purposes"""
        template_lower = template_section.lower().strip()
        actual_lower = actual_section.lower().strip()
        
        # Remove numbers and special characters for comparison
        template_clean = re.sub(r'^\d+\.?\s*', '', template_lower)
        actual_clean = re.sub(r'^\d+\.?\s*', '', actual_lower)
        
        return template_clean == actual_clean or template_clean in actual_clean or actual_clean in template_clean
    
    def _clean_document_structure(self, content: str) -> str:
        """Clean up document structure issues"""
        # Remove orphaned LaTeX environment names
        content = re.sub(r'^(equation|enumerate|verbatim)\s*$', '', content, flags=re.MULTILINE)
        
        # Ensure single \end{document}
        content = re.sub(r'\\end\{document\}', '', content)
        content = content.rstrip() + '\n\n\\end{document}'
        
        # Clean up excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content
    
    def generate_analysis_report(self, original_file: str, final_file: str) -> Dict[str, Any]:
        """Generate analysis report using configurable patterns"""
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
        """Analyze content preservation using configurable patterns"""
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(final_file, 'r', encoding='utf-8') as f:
                final = f.read()
            
            # Clean content using configurable patterns
            original_clean = self._clean_content(original)
            final_clean = self._clean_content(final)
            
            # Count elements using configurable patterns
            analysis_results = {}
            for element_name, pattern in self.analysis_patterns.items():
                original_count = len(re.findall(pattern, original, re.DOTALL))
                final_count = len(re.findall(pattern, final, re.DOTALL))
                
                analysis_results[f"{element_name}_preserved"] = {
                    'original': original_count,
                    'final': final_count,
                    'preserved': original_count == final_count
                }
            
            analysis_results['character_count'] = {
                'original': len(original_clean),
                'final': len(final_clean),
                'change_percent': ((len(final_clean) - len(original_clean)) / len(original_clean)) * 100 if len(original_clean) > 0 else 0
            }
            
            return analysis_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def _clean_content(self, content: str) -> str:
        """Clean content using configurable patterns"""
        cleaned = content
        
        # Apply cleaning patterns from config
        cleaning_patterns = self.config.config.get('cleaning_patterns', [])
        for pattern in cleaning_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        return re.sub(r'\n\s*\n', '\n', cleaned).strip()
    
    def track_chunk(self, chunk_id: int, original_section: str, target_section: str, content_preview: str):
        """Track chunk transformation"""
        preview_length = self.config.config.get('preview_length', 100)
        self.chunk_tracking[chunk_id] = {
            'original_section': original_section,
            'target_section': target_section,
            'content_preview': content_preview[:preview_length] + "..." if len(content_preview) > preview_length else content_preview
        }
    
    def log(self, message: str):
        """Add entry to processing log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        
        # Print if verbose mode enabled
        if self.config.config.get('logging', {}).get('verbose_output', True):
            print(log_entry)

class DocumentCombiner:
    """Handle document combination with configurable strategies"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.combination_strategies = self.config.config.get('combination_strategies', {})
    
    def combine_documents(self, doc1_sections: Dict[str, List[Dict[str, Any]]], 
                         doc2_sections: Dict[str, List[Dict[str, Any]]], 
                         strategy: str = "smart_merge") -> Dict[str, str]:
        """Combine two documents using configurable strategy"""
        
        strategy_config = self.combination_strategies.get(strategy, {})
        method_name = strategy_config.get('method', 'smart_merge')
        
        if method_name == 'merge':
            return self._merge_documents(doc1_sections, doc2_sections, strategy_config)
        elif method_name == 'interleave':
            return self._interleave_documents(doc1_sections, doc2_sections, strategy_config)
        elif method_name == 'append':
            return self._append_documents(doc1_sections, doc2_sections, strategy_config)
        else:  # smart_merge
            return self._smart_merge_documents(doc1_sections, doc2_sections, strategy_config)
    
    def _merge_documents(self, doc1_sections: Dict, doc2_sections: Dict, config: Dict) -> Dict[str, str]:
        """Merge sections by combining content from both documents"""
        combined = {}
        all_sections = set(doc1_sections.keys()) | set(doc2_sections.keys())
        
        merge_template = config.get('merge_template', '% From Document 1\n{doc1_content}\n\n% From Document 2\n{doc2_content}')
        
        for section in all_sections:
            content_parts = []
            
            doc1_content = '\n\n'.join([chunk['content'] for chunk in doc1_sections.get(section, [])])
            doc2_content = '\n\n'.join([chunk['content'] for chunk in doc2_sections.get(section, [])])
            
            if doc1_content and doc2_content:
                combined[section] = merge_template.format(
                    doc1_content=doc1_content,
                    doc2_content=doc2_content
                )
            elif doc1_content:
                combined[section] = doc1_content
            elif doc2_content:
                combined[section] = doc2_content
        
        return combined
    
    def _interleave_documents(self, doc1_sections: Dict, doc2_sections: Dict, config: Dict) -> Dict[str, str]:
        """Interleave sections from both documents"""
        combined = {}
        all_sections = sorted(set(doc1_sections.keys()) | set(doc2_sections.keys()))
        
        interleave_template = config.get('interleave_template', '% Document 1 - Chunk {chunk_num}\n{content}')
        
        for section in all_sections:
            content_parts = []
            
            doc1_chunks = doc1_sections.get(section, [])
            doc2_chunks = doc2_sections.get(section, [])
            
            max_chunks = max(len(doc1_chunks), len(doc2_chunks))
            
            for i in range(max_chunks):
                if i < len(doc1_chunks):
                    content_parts.append(interleave_template.format(
                        chunk_num=i+1, content=doc1_chunks[i]['content']
                    ).replace('Document 1', 'Document 1'))
                if i < len(doc2_chunks):
                    content_parts.append(interleave_template.format(
                        chunk_num=i+1, content=doc2_chunks[i]['content']
                    ).replace('Document 1', 'Document 2'))
            
            combined[section] = '\n\n'.join(content_parts)
        
        return combined
    
    def _append_documents(self, doc1_sections: Dict, doc2_sections: Dict, config: Dict) -> Dict[str, str]:
        """Append second document after first document"""
        combined = {}
        
        part_template = config.get('part_template', 'Part {part_num} - {section}')
        
        # Add all sections from doc1
        for section, chunks in doc1_sections.items():
            section_name = part_template.format(part_num='I', section=section)
            combined[section_name] = '\n\n'.join([chunk['content'] for chunk in chunks])
        
        # Add all sections from doc2
        for section, chunks in doc2_sections.items():
            section_name = part_template.format(part_num='II', section=section)
            combined[section_name] = '\n\n'.join([chunk['content'] for chunk in chunks])
        
        return combined
    
    def _smart_merge_documents(self, doc1_sections: Dict, doc2_sections: Dict, config: Dict) -> Dict[str, str]:
        """Enhanced intelligent merge with content preservation"""
        combined = {}
        
        # Get all sections from both documents
        all_sections = set(doc1_sections.keys()) | set(doc2_sections.keys())
        
        # Get section priority from config
        section_priority = config.get('section_priority', [
            'Abstract', 'Introduction', 'Background', 'Methodology', 'Methods',
            'Results', 'Discussion', 'Conclusion', 'References'
        ])
        
        # Enhanced content preservation settings
        preserve_all_content = config.get('preserve_all_content', True)
        merge_equations = config.get('merge_equations', True)
        merge_lists = config.get('merge_lists', True)
        
        # Step 1: Handle priority sections with intelligent merging
        handled_sections = set()
        
        for priority_section in section_priority:
            matching_sections = []
            
            # Find matching sections in both documents (fuzzy matching)
            for section in doc1_sections.keys():
                if self._sections_match(priority_section, section):
                    matching_sections.append(('doc1', section))
                    handled_sections.add(section)
            
            for section in doc2_sections.keys():
                if self._sections_match(priority_section, section):
                    matching_sections.append(('doc2', section))
                    handled_sections.add(section)
            
            if matching_sections:
                content_parts = []
                equations = []
                lists = []
                
                for doc_id, section in matching_sections:
                    source_sections = doc1_sections if doc_id == 'doc1' else doc2_sections
                    for chunk in source_sections[section]:
                        content = chunk['content']
                        
                        # Extract and preserve equations
                        if merge_equations:
                            chunk_equations = re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', content, re.DOTALL)
                            equations.extend(chunk_equations)
                            # Remove equations from content to avoid duplication
                            for eq in chunk_equations:
                                content = content.replace(eq, '')
                        
                        # Extract and preserve lists
                        if merge_lists:
                            chunk_lists = re.findall(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', content, re.DOTALL)
                            lists.extend(chunk_lists)
                            # Remove lists from content to avoid duplication
                            for lst in chunk_lists:
                                content = content.replace(lst, '')
                        
                        content_parts.append(content.strip())
                
                # Combine content with preserved elements
                final_content = '\n\n'.join([part for part in content_parts if part.strip()])
                
                # Add back equations
                if equations:
                    final_content += '\n\n' + '\n\n'.join(equations)
                
                # Add back lists (merge if multiple)
                if lists:
                    if len(lists) > 1:
                        # Merge multiple lists into one
                        merged_list = self._merge_lists(lists)
                        final_content += '\n\n' + merged_list
                    else:
                        final_content += '\n\n' + lists[0]
                
                combined[priority_section] = final_content
        
        # Step 2: Handle remaining sections (preserve all content)
        remaining_sections = all_sections - handled_sections
        
        for section in remaining_sections:
            content_parts = []
            
            if section in doc1_sections:
                for chunk in doc1_sections[section]:
                    content_parts.append(chunk['content'])
            
            if section in doc2_sections:
                for chunk in doc2_sections[section]:
                    content_parts.append(chunk['content'])
            
            combined[section] = '\n\n'.join(content_parts)
        
        return combined
    
    def _sections_match(self, priority_section: str, actual_section: str) -> bool:
        """Enhanced section matching with fuzzy logic"""
        priority_lower = priority_section.lower()
        actual_lower = actual_section.lower()
        
        # Exact match
        if priority_lower == actual_lower:
            return True
        
        # Substring match
        if priority_lower in actual_lower or actual_lower in priority_lower:
            return True
        
        # Keyword matching for common variations
        keyword_matches = {
            'introduction': ['intro', 'background', 'overview'],
            'methodology': ['methods', 'approach', 'technique'],
            'results': ['findings', 'outcomes', 'analysis'],
            'discussion': ['implications', 'interpretation'],
            'conclusion': ['summary', 'conclusions', 'final']
        }
        
        for key, variations in keyword_matches.items():
            if key in priority_lower:
                for variation in variations:
                    if variation in actual_lower:
                        return True
        
        return False
    
    def _merge_lists(self, lists: List[str]) -> str:
        """Intelligently merge multiple enumerate lists"""
        if not lists:
            return ''
        
        if len(lists) == 1:
            return lists[0]
        
        # Extract items from all lists
        all_items = []
        
        for lst in lists:
            # Extract items between \begin{enumerate} and \end{enumerate}
            items_match = re.search(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}', lst, re.DOTALL)
            if items_match:
                items_content = items_match.group(1)
                # Find all \item entries
                items = re.findall(r'\\item\s+([^\\]+)', items_content)
                all_items.extend([item.strip() for item in items])
        
        # Create merged list
        if all_items:
            merged_content = '\n'.join([f'\\item {item}' for item in all_items])
            return f'\\begin{{enumerate}}\n{merged_content}\n\\end{{enumerate}}'
        
        return lists[0]  # Fallback to first list

class DocumentProcessor:
    """Main document processing orchestrator - fully configuration-driven"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        
        self.loader = DocumentLoader(self.config_manager)
        self.chunker = DocumentChunker(self.config_manager)
        self.llm_handler = LLMHandler(self.config_manager)
        self.output_handler = OutputHandler(self.config_manager)
        self.combiner = DocumentCombiner(self.config_manager)
    
    def combine_documents(self, source1: str, source2: str, strategy: str = "smart_merge", template: Optional[str] = None) -> Dict[str, Any]:
        """Combine two documents using configurable strategy"""
        self.output_handler.log(f"Starting document combination: {source1} + {source2}")
        
        # Load both documents
        content1 = self.loader.load_document(source1)
        content2 = self.loader.load_document(source2)
        self.output_handler.log(f"Loaded documents: {len(content1)} + {len(content2)} characters")
        
        # Extract chunks from both documents
        chunks1 = self.chunker.extract_chunks(content1)
        chunks2 = self.chunker.extract_chunks(content2)
        self.output_handler.log(f"Extracted chunks: {len(chunks1)} + {len(chunks2)}")
        
        # Group chunks by section
        sections1 = self._group_chunks_by_section(chunks1)
        sections2 = self._group_chunks_by_section(chunks2)
        
        # Combine documents using specified strategy
        combined_sections = self.combiner.combine_documents(sections1, sections2, strategy)
        self.output_handler.log(f"Combined using {strategy} strategy: {len(combined_sections)} sections")
        
        # Process combined sections with LLM if enhancement enabled
        processed_sections = {}
        template_name = template or self.config_manager.config.get('processing', {}).get('template', 'academic')
        
        if self.config_manager.config.get('processing', {}).get('enable_enhancement', True):
            # Get section prompts from template configuration
            template_config = self.config_manager.config.get('templates', {}).get(template_name, {})
            section_configs = template_config.get('sections', [])
            
            for section_name, content in combined_sections.items():
                self.output_handler.log(f"Processing combined section: {section_name}")
                
                # Find matching section config
                section_config = next((s for s in section_configs if s['name'] == section_name), None)
                prompt_name = section_config.get('prompt_name', 'default_section_prompt') if section_config else 'default_section_prompt'
                
                # Process with LLM
                processed_content = self.llm_handler.process_content(
                    content, prompt_name, section_name
                )
                
                processed_sections[section_name] = processed_content
                
                # Save individual section
                self.output_handler.save_section(section_name, processed_content)
        else:
            processed_sections = combined_sections
            
            # Save sections without LLM processing
            for section_name, content in combined_sections.items():
                self.output_handler.save_section(section_name, content)
        
        # Assemble final combined document
        final_path = self.output_handler.assemble_document(processed_sections, template_name)
        
        # Generate analysis report (comparing with first source as baseline)
        analysis = self.output_handler.generate_analysis_report(source1, final_path)
        
        return {
            'final_document': final_path,
            'session_path': self.output_handler.session_path,
            'analysis': analysis,
            'processed_sections': len(processed_sections),
            'combination_strategy': strategy,
            'source_documents': [source1, source2]
        }
    
    def process_document(self, source: str, template: Optional[str] = None) -> Dict[str, Any]:
        """Process single document using configuration"""
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
        template_name = template or self.config_manager.config.get('processing', {}).get('template', 'academic')
        
        # Get section prompts from template configuration
        template_config = self.config_manager.config.get('templates', {}).get(template_name, {})
        section_configs = template_config.get('sections', [])
        
        for section_name, section_chunks in sections.items():
            if section_chunks:
                self.output_handler.log(f"Processing section: {section_name}")
                
                # Find matching section config
                section_config = next((s for s in section_configs if s['name'] == section_name), None)
                prompt_name = section_config.get('prompt_name', 'default_section_prompt') if section_config else 'default_section_prompt'
                
                # Combine chunk content
                combined_content = "\n\n".join([chunk['content'] for chunk in section_chunks])
                
                # Process with LLM if enhancement enabled
                if self.config_manager.config.get('processing', {}).get('enable_enhancement', True):
                    processed_content = self.llm_handler.process_content(
                        combined_content, prompt_name, section_name
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