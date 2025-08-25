"""Configuration file for document processing"""

# LLM Configuration
LLM_CONFIG = {
    "provider": "gemini",  # Options: "mistral", "openai", "huggingface", "gemini", "vertexai"
    "model": "gemini-1.5-flash",
    "api_key_env": "GEMINI_API_KEY",
    "max_tokens": 2048,
    "temperature": 0.2,
    "timeout": 30
}

# Semantic Mapping Configuration
# This new section configures the model used for semantic chunk assignment.
SEMANTIC_MAPPING_CONFIG = {
    # A lightweight sentence-transformer model for fast and effective semantic similarity calculation.
    "model": "all-MiniLM-L6-v2",
    # The minimum cosine similarity score required to assign a chunk to a section.
    # This is a tunable parameter; a lower value is more inclusive, a higher value is more strict.
    "similarity_threshold": 0.3
}


# Alternative LLM providers
LLM_PROVIDERS = {
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "models": ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    },
    "huggingface": {
        "url": "https://api-inference.huggingface.co/models/",
        "models": ["mistralai/Mistral-7B-Instruct-v0.1", "google/gemma-7b-it"]
    },
    "gemini": {
        "models": ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro-latest"]
    },
    "vertexai": {
        "models": ["gemini-1.0-pro", "gemini-1.5-pro-latest", "gemini-2.5-pro"]
    }
}

# Chunking Strategies
CHUNKING_STRATEGIES = {
    "semantic": "Semantic-based chunking",
    "section": "Section-based chunking",
    "llm_enhanced": "LLM-enhanced chunking"
}

# Document Templates
# Each template now includes a 'description' field. This is critical for the new
# state-of-the-art semantic mapping, as it provides the text used to generate
# a vector embedding for each target section.
DOCUMENT_TEMPLATES = {
    "bitcoin_paper": [
        {"section": "Summary", "prompt": "Format as LaTeX with section title 'Summary'. Preserve all technical content exactly.", "description": "A high-level summary of the entire document, covering the key problems and solutions."},
        {"section": "Abstract", "prompt": "Format as LaTeX with section title 'Abstract'. Preserve all technical content exactly.", "description": "A concise, formal summary of the paper's main points, typically for an academic audience."},
        {"section": "1. Introduction", "prompt": "Format as LaTeX with section title 'Introduction'. Preserve all technical content exactly.", "description": "The introductory section explaining the background of commerce on the Internet, the problem with the traditional trust-based model, and the purpose of the electronic cash system."},
        {"section": "2. Transactions", "prompt": "Format as LaTeX with section title 'Transactions'. Preserve all technical content exactly.", "description": "Details the definition of an electronic coin as a chain of digital signatures, and the process of transferring ownership."},
        {"section": "3. Timestamp Server", "prompt": "Format as LaTeX with section title 'Timestamp Server'. Preserve all technical content exactly.", "description": "Explains the solution to the double-spending problem by using a distributed timestamp server to create a public history of transactions."},
        {"section": "4. Proof-of-Work", "prompt": "Format as LaTeX with section title 'Proof-of-Work'. Preserve all technical content exactly.", "description": "Describes the implementation of the distributed timestamp server using a proof-of-work system, similar to Adam Back's Hashcash."},
        {"section": "5. Network", "prompt": "Format as LaTeX with section title 'Network'. Preserve all technical content exactly.", "description": "Outlines the steps for running the network, including broadcasting transactions, block creation, and longest-chain validation."},
        {"section": "6. Incentive", "prompt": "Format as LaTeX with section title 'Incentive'. Preserve all technical content exactly.", "description": "Details the incentive system for nodes participating in the network, including the creation of new coins and transaction fees."},
        {"section": "7. Reclaiming Disk Space", "prompt": "Format as LaTeX with section title 'Reclaiming Disk Space'. Preserve all technical content exactly.", "description": "Discusses methods for pruning the blockchain to save disk space once transactions are sufficiently buried."},
        {"section": "8. Simplified Payment Verification", "prompt": "Format as LaTeX with section title 'Simplified Payment Verification'. Preserve all technical content exactly.", "description": "Explains how payment verification can be achieved without running a full network node, by using block headers."},
        {"section": "9. Combining and Splitting Value", "prompt": "Format as LaTeX with section title 'Combining and Splitting Value'. Preserve all technical content exactly.", "description": "Describes how transactions can handle multiple inputs and outputs to combine and split currency value."},
        {"section": "10. Privacy", "prompt": "Format as LaTeX with section title 'Privacy'. Preserve all technical content exactly.", "description": "Addresses the privacy model of the system, where public keys are anonymous but transaction flow can be traced."},
        {"section": "11. Major and Minor Assumptions", "prompt": "Format as LaTeX with section title 'Major and Minor Assumptions'. Preserve all technical content exactly.", "description": "A section to consolidate all the key assumptions, both major and minor, that the system's security and functionality rely on."},
        {"section": "12. Calculations", "prompt": "Format as LaTeX with section title 'Calculations'. Preserve all technical content exactly.", "description": "Presents the mathematical analysis and calculations, particularly regarding the probability of an attacker catching up to the honest chain."},
        {"section": "13. Conclusion", "prompt": "Format as LaTeX with section title 'Conclusion'. Preserve all technical content exactly.", "description": "The concluding section summarizing the benefits of the proposed electronic cash system, such as eliminating the need for trust and protecting sellers."}
    ]
}

# Output Formats
OUTPUT_FORMATS = {
    "latex": {
        "extension": ".tex",
        "header": "\\documentclass{article}\n\\usepackage[pdftex]{graphicx}\n\\usepackage{amsmath}\n\\begin{document}",
        "footer": "\\end{document}"
    },
    "markdown": {
        "extension": ".md",
        "header": "",
        "footer": ""
    },
    "json": {
        "extension": ".json",
        "header": "",
        "footer": ""
    }
}
# Below example for hierarchical document template
DOCUMENT_TEMPLATES = {
    "bitcoin_paper_hierarchical": {
        # Top-level sections are keys
        "1. Introduction": {
            "prompt": "Process the introduction...",
            "description": "The introduction explaining the background...",
            "chunks": [], # Placeholder for chunks belonging directly to this section
            "subsections": {} # Placeholder for nested subsections
        },
        "2. Transactions": {
            "prompt": "Process the main content about transactions...",
            "description": "Details the definition of an electronic coin as a chain of digital signatures...",
            "chunks": [],
            "subsections": {
                # Nested subsections are keys within the 'subsections' dict
                "2.1 Ownership Model": {
                    "prompt": "Focus on the ownership aspect of transactions...",
                    "description": "Explains the model of ownership transfer using digital signatures.",
                    "chunks": [],
                    "subsections": {} # Can be nested further
                },
                "2.2 Double-Spending Problem": {
                    "prompt": "Explain the double-spending problem...",
                    "description": "Describes the fundamental problem of double-spending in a digital cash system.",
                    "chunks": [],
                    "subsections": {}
                }
            }
        },
        # ... other top-level sections ...
        "13. Conclusion": {
            "prompt": "Conclude the paper...",
            "description": "The concluding section summarizing the benefits...",
            "chunks": [],
            "subsections": {}
        }
    }
}

# --- Centralized Prompt Hub ---
# This dictionary contains all the prompts used for LLM interactions.
# Using placeholders like {section_name} allows for dynamic formatting in the code.

PROMPTS = {
    "content_weaving": """
You are an expert technical editor. Your task is to seamlessly weave additional information into an original text for a section titled '{section_name}'.

**Original Content:**
{original_content}

**Additional Content to Integrate:**
{augmentation_content}

**Instructions:**
1.  Rewrite the 'Original Content' to cohesively include the key points, details, and examples from the 'Additional Content'.
2.  **Do not simply append** the new content. Integrate it naturally where it makes the most sense.
3.  Preserve the tone, style, and all technical details of the original text.
4.  Ensure the final output is a single, unified, and coherent narrative.
5.  If the additional content is redundant, discard it. If it conflicts, prioritize the original.

**Final Weaved Content:**
""",

    "structure_grafting_location": """
You are a document structuring expert. Your task is to find the most logical place to insert a new section into an existing document structure.

**Existing Document Structure:**
{tree_summary}

**New Section to Insert:**
- Title: "{new_section_title}"
- Description: "{new_section_description}"

**Instructions:**
Based on the title and description of the new section, identify the best parent section in the existing structure to insert it under.
- Respond with the hierarchical path to the PARENT section.
- Use "->" as a separator for the path.
- If the new section should be at the top level, respond with "ROOT".
- Do not explain your reasoning. Just provide the path.

Example Response: "Section Title -> Subsection Title"
Example Response: "ROOT"

**Parent Path:**
""",
 "hierarchical_refactor": """
You are a world-class technical editor and LaTeX expert. Your task is to refactor the following content from a document.

**Document Context:**
This content is from the section/subsection titled: "{node_path}".
The abstract or summary of the entire document is:
{global_context}

**Parent Section Context:**
The content of the parent section is:
{parent_context}

**Content to Refactor:**
{node_content}

**Instructions:**
1.  Rewrite the provided "Content to Refactor" for maximum clarity, conciseness, and professional academic tone.
2.  Ensure your output is ONLY the refactored content. Do not include titles or sectioning commands.
3.  **Crucially, you must preserve all original LaTeX commands, environments (like `equation`, `figure`), citations (`\cite`), and references (`\ref`) exactly as they appear.**
4.  Maintain all the original technical details and semantic meaning.
5.  Improve the logical flow and transition between ideas.

**Refactored LaTeX Content:**
""",

    # NEW PROMPT for the self-critique pass
    "self_critique_and_refine": """
You are a meticulous quality assurance editor. You will be given a piece of text that has already been refactored once, along with the original context. Your job is to critique the refactored text and then produce a final, improved version.

**Original Context:**
- Document Path: "{node_path}"
- Original Content Summary: A piece of text discussing the main topic of "{node_path}".

**Refactored Text to Review:**
{refactored_text}

**Critique Checklist:**
1.  **Clarity & Conciseness:** Is the text as clear and direct as possible? Is there any jargon that could be simplified? Are there any redundant phrases?
2.  **Logical Flow:** Do the ideas connect smoothly? Is the argument easy to follow?
3.  **Technical Preservation:** Is it plausible that all original LaTeX commands, citations, and technical details were preserved? (You don't have the original, so assess based on structure).
4.  **Tone:** Is the tone appropriate for a formal academic or technical paper?

**Your Task:**
First, write a brief critique of the "Refactored Text to Review" based on the checklist above.
Second, based on your critique, provide a final, polished version of the text.

**Critique:**
[Your brief critique here]

**Final Polished Version:**
""",
# NEW PROMPT for extracting key terms from the entire document
    "term_extraction": """
You are a domain expert analyzing a technical document. Your task is to extract the 10-15 most important and frequently used technical terms and phrases.

**Full Document Content (Excerpt):**
{full_text_excerpt}

**Instructions:**
- Identify the core technical concepts of the document.
- List the terms as a comma-separated list.
- Do not include generic words. Focus on specific terminology.

**Key Technical Terms:**
""",

    # NEW PROMPT for standardizing terminology within a piece of text
    "term_standardization": """
You are a meticulous copy editor. Your task is to ensure consistent terminology in the following text, based on a predefined list of key terms.

**Official Key Terms:**
{key_terms_list}

**Text to Standardize:**
{text_content}

**Instructions:**
- Review the "Text to Standardize" and identify any variations or synonyms for the "Official Key Terms".
- Rewrite the text to use the official terms consistently.
- For example, if "proof of work" is an official term, change instances of "work-proof" or "PoW system" to match.
- Preserve the original meaning, tone, and all LaTeX commands.
- Respond with ONLY the standardized text.

**Standardized Text:**
""",

    # NEW PROMPT for generating a transition sentence between sections
    "section_transition": """
You are a technical writer creating a smooth narrative flow in a document. Your task is to write a single, concise transition sentence that connects the end of a preceding section with the beginning of the current section.

**End of Preceding Section ('{prev_section_title}'):**
...{prev_section_ending}

**Beginning of Current Section ('{current_section_title}'):**
{current_section_beginning}...

**Instructions:**
- Write one sentence that logically links the two sections.
- The sentence should act as a bridge, making the shift in topic feel natural.
- Do not summarize the sections. Just connect them.
- The output should be the transition sentence itself, formatted as a LaTeX paragraph.

**Transition Sentence:**
"""

}

OUTPUT_FORMATS = {
    "latex": {
        "extension": ".tex",
        "header": """\\documentclass{article}
                    \\usepackage[utf8]{inputenc}
                    \\usepackage{amsmath}
                    \\usepackage{graphicx}
                    \\usepackage{hyperref}
                    \\title{Refactored Document}
                    \\author{Automated System}
                    \\date{\\today}
                    \\begin{document}
                    \\maketitle""",
                    "footer": "\\end{document}"
    },
    "markdown": {
        "extension": ".md",
        "header": "",
        "footer": ""
    },
    "json": {
        "extension": ".json",
        "header": "",
        "footer": ""
    }
}

# --- System Prompts for FormatEnforcer ---
# These are specialized system prompts used to guide the LLM's output syntax.
FORMAT_ENFORCER_PROMPTS = {
    "latex": """You are a LaTeX expert. Output ONLY valid LaTeX content.
STRICT RULES:
- Use \\section{Title} for sections, NOT ### Title
- Use \\subsection{Title} for subsections, NOT #### Title
- Use \\textbf{text} for bold, NOT **text**
- Use \\textit{text} for italics, NOT *text*
- Use \\begin{itemize} \\item ... \\end{itemize} for lists, NOT - item
- Use \\begin{enumerate} \\item ... \\end{enumerate} for numbered lists
- Preserve all equations exactly as \\begin{equation} ... \\end{equation}
- NO Markdown syntax allowed
- NO document structure (\\documentclass, \\begin{document}, \\end{document})""",

    "markdown": """You are a Markdown expert. Output ONLY valid Markdown content.
STRICT RULES:
- Use ### for sections, NOT \\section{}
- Use #### for subsections, NOT \\subsection{}
- Use **text** for bold, NOT \\textbf{}
- Use *text* for italics, NOT \\textit{}
- Use - item for lists, NOT \\begin{itemize}
- Use ```language for code blocks
- Convert LaTeX equations to $...$ or $$...$$"""
}