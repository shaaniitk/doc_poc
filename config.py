"""Configuration file for document processing"""

# --- NEW: Configuration for LangChain-based Chunkers ---
LANGCHAIN_CHUNK_CONFIG = {
    'md_chunk_size': 1500,
    'md_chunk_overlap': 150,
}


# LLM Configuration
LLM_CONFIG = {
    "provider": "mistral",  # Options: "mistral", "openai", "huggingface", "gemini", "vertexai"
    "model": "mistral-small-latest",
    "api_key_env": "MISTRAL_API_KEY",
    "max_tokens": 2048,
    "temperature": 0.1,
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

# LLM-Enhanced Chunking Configuration
LLM_CHUNK_CONFIG = {
    # The minimum number of characters a paragraph chunk must have to be considered for a semantic split.
    "SEMANTIC_SPLIT_THRESHOLD": 1500,
    # Whether to enable this feature. Allows for easy toggling for performance.
    "ENABLE_LLM_ENHANCEMENT": True
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
    ],
     "bitcoin_paper_hierarchical": {
       "Abstract": { 
                    "prompt": """
                        You are a technical summarizer. Based on the full document text provided, write a concise, professional summary.

                        **RULES:**
                        1.  Your output MUST be ONLY the summary text itself.
                        2.  Do NOT include any LaTeX preamble, \\documentclass, \\begin{{document}}, \\maketitle, or section commands.
                        3.  Write only the content for the summary.
                    """,
                    "description": "The main summary of the paper, outlining the core problem and solution.",
                    "generative": True,
                    "subsections": {}
                },
        "Introduction": {
            "prompt": "You are a LaTeX expert. Refactor the following introductory content for clarity and logical flow...",
            "description": "This is the instroductory section of the document which introduces the main concepts of bitcoin and its background and should streamlessly create a background for the document and movement to the next section.",
            "chunks": [],
            "subsections": {}
        },
        "Transactions": {
            "prompt": "You are a LaTeX expert. Refactor the following content to precisely define an electronic coin...",
            "description": "This section details the fundamental definition of an electronic coin as a chain of digital signatures...",
            "chunks": [],
            "subsections": {}
        },
        "Timestamp Server": {
            "prompt": "You are a LaTeX expert. Refactor the following content to clearly explain the concept of a distributed timestamp server...",
            "description": "Explains the proposed solution to the double-spending problem by using a distributed timestamp server...",
            "chunks": [],
            "subsections": {}
        },
        "Proof-of-Work": {
            "prompt": "You are a LaTeX expert. Refactor the following content to explain how proof-of-work is used...",
            "description": "Describes the implementation of the distributed timestamp server using a proof-of-work system...",
            "chunks": [],
            "subsections": {}
        },
        "Network": {
            "prompt": "You are a LaTeX expert. Refactor the following steps describing the network's operation...",
            "description": "Outlines the step-by-step process for running the peer-to-peer network...",
            "chunks": [],
            "subsections": {}
        },
        "Incentive": {
            "prompt": "You are a LaTeX expert. Refactor the following content to clearly explain the economic incentives...",
            "description": "Details the incentive system for nodes participating in the network...",
            "chunks": [],
            "subsections": {}
        },
        "Reclaiming Disk Space": {
            "prompt": "You are a LaTeX expert. Refactor the following content to explain the method for reclaiming disk space...",
            "description": "Discusses a method for pruning the blockchain to save disk space...",
            "chunks": [],
            "subsections": {}
        },
        "Simplified Payment Verification": {
            "prompt": "You are a LaTeX expert. Refactor the following content to explain the Simplified Payment Verification (SPV) method...",
            "description": "Explains how payment verification can be achieved without running a full network node...",
            "chunks": [],
            "subsections": {}
        },
        "Combining and Splitting Value": {
            "prompt": "You are a LaTeX expert. Refactor the following content to clearly explain the practical mechanics of transactions...",
            "description": "Describes the practical functionality of how transactions can handle value...",
            "chunks": [],
            "subsections": {}
        },
        "Security Analysis": {
            "prompt": "You are a security expert and LaTeX professional. Refactor the following content to provide an overview of the system's security...",
            "description": "An overarching section that covers the security properties of the system...",
            "chunks": [],
            "subsections": {
                "10.1 Privacy Model": {
                    "prompt": "You are a LaTeX expert. Refactor this content to focus specifically on the privacy model...",
                    "description": "Addresses the privacy model of the system...",
                    "chunks": [],
                    "subsections": {}
                },
                "10.2 Attack Vector Calculations": {
                    "prompt": "You are a LaTeX expert and mathematician. Refactor the following text and equations...",
                    "description": "Presents the mathematical analysis of the system's security against an attacker...",
                    "chunks": [],
                    "subsections": {}
                }
            }
        },
        # --- NEW ASSUMPTIONS SECTION ---
        "11. Assumptions": {
            "prompt": "You are a system analyst and LaTeX expert. Based on the document content, provide a brief introductory paragraph for a section that will outline the core assumptions the system relies on. This paragraph should set the stage for the major and minor subsections.",
            "description": "A section dedicated to explicitly stating the underlying assumptions required for the Bitcoin protocol to function securely and effectively. This includes assumptions about network behavior and participant honesty.",
            "chunks": [],
            "subsections": {
                "11.1 Major Assumptions": {
                    "prompt": "You are a system analyst and LaTeX expert. From the provided text, extract and clearly articulate the most critical assumptions for the system's security. The primary assumption is that honest nodes control a majority of CPU power. Explain the implications of this assumption.",
                    "description": "Details the most critical, foundational assumptions of the system. The foremost assumption is that the majority of the CPU power in the network is controlled by honest nodes that are not conspiring to attack the network.",
                    "chunks": [],
                    "subsections": {}
                },
                "11.2 Minor Assumptions": {
                    "prompt": "You are a system analyst and LaTeX expert. From the provided text, identify and list any secondary or implicit assumptions. This could include assumptions about network latency (nodes receive broadcasts in a timely manner) or participant behavior (nodes are economically rational).",
                    "description": "Outlines other, less critical but still important, assumptions. This includes assumptions such as nodes having reliable network connectivity, the practicality of storing block headers for SPV, and that participants generally act in their own economic self-interest.",
                    "chunks": [],
                    "subsections": {}
                }
            }
        },
        "12. Conclusion": {
            "prompt": "You are a LaTeX expert. Refactor the following content into a strong, formal conclusion...",
            "description": "The concluding section that summarizes the paper's proposal...",
            "chunks": [],
            "subsections": {}
        },
        "13. References": {
            "prompt": "You are a LaTeX expert. Format the following content as a standard 'References' section...",
            "description": "The list of citations and prior work referenced in the paper...",
            "chunks": [],
            "subsections": {}
        },
        "14. System Tests": {
        # --- ADD THIS FLAG ---
        "dynamic_subsections": True,
        # This prompt is used for the *content* of each dynamically created subsection
        "prompt": "You are a QA analyst and LaTeX expert. Refactor the following test case for clarity, focusing on the setup, actions, and expected results. Preserve all technical details and code snippets.",
        # The description helps map the parent section correctly
        "description": "A section detailing the various test cases performed on the system to validate its functionality and security. The subsections will be generated dynamically based on the number of tests found in the content.",
        "chunks": [],
        "subsections": {}
        }
    },

    "dynamic_subsection_identifier": """
    You are a document structuring AI. Your task is to analyze a large block of text from a "{parent_section_title}" section and identify all the distinct, logical subsections within it.
Content to Analyze:
{text_content}
Instructions:
Read the content and identify the natural divisions or sub-topics.
For each distinct sub-topic, create a concise and appropriate subsection title.
Return these titles as a JSON-formatted list of strings.
Do not include any other text or explanation. Your output must be only the JSON list.
Example Response:
["Test Case 1: Valid Single Transaction", "Test Case 2: Double-Spend Attempt", "Test Case 3: Transaction with Multiple Inputs"]
JSON Output:
"""
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
 'hierarchical_refactor': """
    You are a professional technical editor refactoring a document section by section.
    
    You have already processed some sections. Here is a summary of the most relevant section you have already written, to ensure consistency:
    MEMORY OF PREVIOUSLY WRITTEN CONTENT:
    ---
    {memory_context}
    ---

    To improve coherence, consider these semantically related excerpts from other parts of the document:
    RELATED CONTEXT:
    ---
    {semantic_context}
    ---
    
    GLOBAL DOCUMENT CONTEXT: {global_context}
    PARENT SECTION CONTEXT: {parent_context}

    CURRENT CONTENT TO REFACTOR (for section '{node_path}'):
    ---
    {node_content}
    ---
    Your task is to rewrite the CURRENT CONTENT to be clear, professional, and stylistically consistent with your MEMORY and the other contexts provided. Adhere strictly to the specified output format.
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

'term_standardization': """
    You are a silent text processor. Your ONLY job is to rewrite the following text to use the provided key terms consistently.

    **Key Terms List:**
    {key_terms_list}

    **Text to Standardize:**
    ---
    {text_content}
    ---

    **RULES:**
    1.  Your output MUST be ONLY the modified text.
    2.  Do NOT add any explanation, preamble, or conversational filler like "Here is the standardized text...".
    3.  Do NOT wrap the output in markdown code blocks or any other formatting.

    **Standardized Text:**
""",

'section_transition': """
    You are a silent transition writer. Your ONLY job is to write a single, concise transition sentence in plain text.

    **Previous Section Ending:** "...{prev_section_ending}"
    **Next Section Beginning:** "{current_section_beginning}..."

    **RULES:**
    1.  Your output MUST be ONLY the single transition sentence.
    2.  Do NOT add any explanation, preamble, or conversational filler like "Here is the transition sentence...".
    3.  Do NOT wrap the sentence in LaTeX commands or markdown code blocks.

    **Transition Sentence:**
""",

"semantic_split_paragraph": """
You are a text analysis expert. Your task is to identify the natural thematic break points within a long piece of text.
Text to Analyze:
{text_content}
Instructions:
Read the entire text to understand its flow and topics.
Identify the most logical places where the text shifts to a new sub-topic.
For each break point you identify, respond with ONLY the first 5-7 words of the sentence that begins the new sub-topic.
Separate each of these "break point markers" with the unique separator |||---|||.
Do not include the beginning of the very first sentence.
Example Response:
A purely peer-to-peer electronic cash|||---|||An electronic coin is defined|||---|||To address this, payees need
Break Point Markers:
""",


    "llm_map_chunk_to_section": """
You are a document structuring expert. Your task is to determine the single best section for a given chunk of text by understanding its content and the purpose of each available section.

**Available Sections (Path and Description):**
{section_details}

**Chunk of Text to Categorize:**
---
{chunk_content}
---

**Instructions:**
1.  Read the "Chunk of Text" and understand its core topic.
2.  Review the "Available Sections" and their descriptions.
3.  Determine which section is the **single most logical fit** for the chunk.
4.  Respond with ONLY the full, exact path to that section (e.g., "10. Security Analysis -> 10.2 Attack Vector Calculations").
5.  If absolutely no section is a good fit, respond with "UNCATEGORIZED".

**Best Fit Section Path:**
""",

'hierarchical_refactor': """
You are a professional technical editor specializing in academic papers. Your task is to refactor a piece of LaTeX content. Your output MUST BE only the refactored, valid LaTeX content.

**-- CONTEXTUAL INFORMATION --**

1.  **YOUR MEMORY (Previously Written Content):** To ensure consistency, here is the most relevant content you have already written for a previous section. Match its style and terminology.
    ---
    {memory_context}
    ---

2.  **SEMANTIC CONTEXT (Related Document Excerpts):** Here are other parts of the document that are thematically related to the current task. Use them to improve coherence.
    ---
    {semantic_context}
    ---

3.  **HIERARCHICAL CONTEXT:**
    -   **Full Document Abstract:** {global_context}
    -   **Parent Section Content:** {parent_context}

**-- YOUR TASK --**

You are currently working on the section: **"{node_path}"**.

**Content to Refactor:**
```latex
{node_content}

  
-- INSTRUCTIONS & RULES --

1. Rewrite the "Content to Refactor" for maximum clarity, conciseness, and a professional academic tone.
2.  Preserve Core Elements: You MUST preserve all original LaTeX commands, environments (e.g., \\begin{{equation}}...\\end{{equation}}), citations (\\cite{{...}}), and references (\\ref{{...}}) exactly as they appear.
3.  Maintain Meaning: Do NOT alter the original technical details or semantic meaning.
4.  Formatting Rule: Do NOT wrap entire paragraphs or multi-line content in formatting commands (e.g., \\textbf{{...}}). Apply formatting only to specific words or short phrases.
5.  Do NOT add or remove any LaTeX commands, environments, citations, or references.
6.  Output:Your response must contain ONLY the refactored LaTeX content for the section. Do not include section headers (\\section{{...}}) or any other text outside of the rewritten content itself.

Refactored LaTeX Content:
""",

 'semantic_split_paragraph': """
 You are an expert in document analysis. Your task is to split the following text into semantically coherent paragraphs.
 Do not lose any information. The output must be a list of strings in the specified JSON format.

 {format_instructions}

 TEXT TO SPLIT:
 ---
 {text_content}
 ---
 """

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