"""Download and preprocess .tex files from various sources"""
import requests
import os
import re
from urllib.parse import urlparse

class TexDownloader:
    def __init__(self, download_dir="downloads"):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
    
    def download_tex_file(self, url, filename=None):
        """Download .tex file from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            if not filename:
                filename = self._extract_filename(url)
            
            file_path = os.path.join(self.download_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            return file_path
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")
    
    def download_arxiv_source(self, arxiv_id):
        """Download LaTeX source from arXiv"""
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save as tar.gz and extract
            import tarfile
            import io
            
            tar_path = os.path.join(self.download_dir, f"{arxiv_id}.tar.gz")
            with open(tar_path, 'wb') as f:
                f.write(response.content)
            
            # Extract and find main .tex file
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.download_dir)
            
            # Find main tex file
            tex_files = []
            for root, dirs, files in os.walk(self.download_dir):
                for file in files:
                    if file.endswith('.tex'):
                        tex_files.append(os.path.join(root, file))
            
            main_tex = self._find_main_tex(tex_files)
            return main_tex
            
        except Exception as e:
            raise Exception(f"arXiv download failed: {str(e)}")
    
    def preprocess_tex(self, file_path):
        """Clean and preprocess downloaded .tex file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove comments (but preserve section markers)
        content = re.sub(r'^%(?! ---)(.*?)$', '', content, flags=re.MULTILINE)
        
        # Normalize whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Add section markers if missing
        content = self._add_section_markers(content)
        
        # Save preprocessed version
        preprocessed_path = file_path.replace('.tex', '_preprocessed.tex')
        with open(preprocessed_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return preprocessed_path
    
    def _extract_filename(self, url):
        """Extract filename from URL"""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        
        if not filename.endswith('.tex'):
            filename += '.tex'
        
        return filename
    
    def _find_main_tex(self, tex_files):
        """Find main .tex file from list"""
        # Look for common main file names
        main_names = ['main.tex', 'paper.tex', 'manuscript.tex']
        
        for tex_file in tex_files:
            if os.path.basename(tex_file) in main_names:
                return tex_file
        
        # Return first .tex file if no main found
        return tex_files[0] if tex_files else None
    
    def _add_section_markers(self, content):
        """Add section markers for better chunking"""
        # Find sections and add markers
        sections = [
            (r'\\begin\{abstract\}', 'Abstract'),
            (r'\\section\*?\{[Ii]ntroduction\}', 'Introduction'),
            (r'\\section\*?\{.*?[Mm]ethod', 'Methods'),
            (r'\\section\*?\{.*?[Rr]esult', 'Results'),
            (r'\\section\*?\{.*?[Dd]iscussion\}', 'Discussion'),
            (r'\\section\*?\{.*?[Cc]onclusion\}', 'Conclusion'),
            (r'\\begin\{thebibliography\}', 'References')
        ]
        
        for pattern, marker in sections:
            content = re.sub(pattern, f'% --- {marker} ---\n\\g<0>', content, flags=re.IGNORECASE)
        
        return content