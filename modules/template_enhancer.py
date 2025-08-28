# CREATE THIS NEW FILE: modules/template_enhancer.py

import logging
import copy
from .llm_client import UnifiedLLMClient
from config import PROMPTS

log = logging.getLogger(__name__)

class TemplateEnhancer:
    """
    Dynamically enhances a document template by generating keyword-rich descriptions
    for key sections based on the full content of the source document.
    """
    def __init__(self, llm_client: UnifiedLLMClient):
        self.llm_client = llm_client

    def enhance_template(self, template: dict, full_document_content: str) -> dict:
        """
        Iterates through a template and enriches sections marked with 'dynamic_description'.
        
        Returns:
            A new, enhanced template dictionary.
        """
        log.info("--- Starting Dynamic Template Enhancement ---")
        # Create a deep copy to avoid modifying the original template from config
        enhanced_template = copy.deepcopy(template)
        
        # Use a large excerpt of the document to avoid excessive token usage
        excerpt = full_document_content[:8000]

        # Use a recursive helper to find the dynamic nodes
        self._find_and_enhance_nodes(enhanced_template, excerpt)
        
        return enhanced_template

    def _find_and_enhance_nodes(self, node_level: dict, excerpt: str):
        """Recursively traverses the template tree."""
        for title, data in node_level.items():
            if data.get("dynamic_description"):
                log.info(f"  -> Generating dynamic description for section: '{title}'")
                try:
                    # Generate the new description
                    new_description = self._llm_generate_description(title, excerpt)
                    # Inject it into the template
                    data["description"] = new_description
                    log.info(f"  -> Successfully injected new description for '{title}'.")
                except Exception as e:
                    log.warning(f"  -> Failed to generate dynamic description for '{title}'. Using generic one. Error: {e}")

            if data.get("subsections"):
                self._find_and_enhance_nodes(data["subsections"], excerpt)

    def _llm_generate_description(self, section_title: str, excerpt: str) -> str:
        """Calls the LLM to generate the new description."""
        prompt = PROMPTS['generate_dynamic_description'].format(
            section_title=section_title,
            full_text_excerpt=excerpt
        )
        response = self.llm_client.call_llm([{"role": "user", "content": prompt}])
        return response.strip()