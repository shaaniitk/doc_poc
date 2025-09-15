"""Configuration file for document processing"""
# Configuration file for document analysis system
# This file contains all configuration parameters for the document processing pipeline
#
# SWITCHING BETWEEN LOCAL AND API MODELS:
# - By default, this configuration uses local models that run offline
# - To use API-based models (OpenAI, Mistral, Cohere, etc.), uncomment the desired
#   API configuration sections and comment out the corresponding local configurations
# - Make sure to set the required API keys as environment variables

class Config:
    """Configuration class providing access to all configuration dictionaries"""
    
    def __init__(self):
        # Initialize with default configurations
        pass
    
    @classmethod
    def get_llm_config(cls):
        return LLM_CONFIG
    
    @classmethod
    def get_embedding_config(cls):
        return EMBEDDING_CONFIG
    
    @classmethod
    def get_semantic_mapping_config(cls):
        return SEMANTIC_MAPPING_CONFIG
    
    @classmethod
    def get_chunking_config(cls):
        return ADAPTIVE_CHUNKING_CONFIG
    
    @classmethod
    def get_all_configs(cls):
        """Return a dictionary of all available configurations"""
        return {
            'llm': LLM_CONFIG,
            'embedding': EMBEDDING_CONFIG,
            'semantic_mapping': SEMANTIC_MAPPING_CONFIG,
            'chunking': ADAPTIVE_CHUNKING_CONFIG,
            'local_model': LOCAL_MODEL_CONFIG,
            'huggingface': HUGGINGFACE_CONFIG
        }

# --- NEW: Configuration for LangChain-based Chunkers ---
LANGCHAIN_CHUNK_CONFIG = {
    'md_chunk_size': 800,
    'md_chunk_overlap': 50,
    
}


# LLM Configuration
# LLM Configuration - Choose between local or API-based models
LLM_CONFIG = {
    "provider": "huggingface_local",  # Local Hugging Face models
    "model": "mistralai/Mistral-7B-v0.3",  # MISTRAL model (registered access)
    "api_key_env": None,  # No API key needed for local models
    "max_tokens": 2048,  # Standard context window for Phi-3
    "temperature": 0.1,
    "timeout": 300,  # Standard timeout for Phi-3
    "device": "auto",  # Auto-detect best device (GPU if available, else CPU)
    "local_model_path": "./models/llm",  # Use cached models from download script
    "cache_folder": "./models",  # Use local models cache
    # Optimized settings for MISTRAL
    "torch_dtype": "float16",  # Optimized precision for MISTRAL
    "device_map": "auto",  # Enable automatic device mapping
    "load_in_8bit": True,  # Enable quantization for MISTRAL
    "load_in_4bit": False,  # Keep 4-bit disabled for stability
    "trust_remote_code": False,  # MISTRAL doesn't need custom code
    "use_cache": True  # Enable KV cache for faster inference
}

# Alternative API-based configurations (uncomment to use):
# OpenAI Configuration
# LLM_CONFIG = {
#     "provider": "openai",
#     "model": "gpt-4",  # or "gpt-3.5-turbo", "gpt-4-turbo"
#     "api_key_env": "OPENAI_API_KEY",
#     "max_tokens": 2048,
#     "temperature": 0.1,
#     "timeout": 30
# }

# Mistral AI Configuration
# LLM_CONFIG = {
#     "provider": "mistral",
#     "model": "mistral-large-latest",  # or "mistral-small-latest", "mistral-medium-latest"
#     "api_key_env": "MISTRAL_API_KEY",
#     "max_tokens": 2048,
#     "temperature": 0.1,
#     "timeout": 30
# }

# Hugging Face API Configuration
# LLM_CONFIG = {
#     "provider": "huggingface",
#     "model": "mistralai/Mistral-7B-Instruct-v0.1",
#     "api_key_env": "HUGGINGFACE_API_KEY",
#     "max_tokens": 2048,
#     "temperature": 0.1,
#     "timeout": 30
# }

# Embedding Model Configuration - Toggle between models by commenting/uncommenting
# Option 1: OpenAI Embedding Model (requires OPENAI_API_KEY)
# SEMANTIC_MAPPING_CONFIG = {
#     "model": "text-embedding-3-large",
#     "provider": "openai",
#     "similarity_threshold": 0.6,
#     "device": "cpu",
#     "batch_size": 32,
#     "top_k_candidates": 3,
#     # Accept borderline matches within this margin below the threshold
#     "soft_accept_margin": 0.05,
#     # Also accept if the top-1 similarity exceeds top-2 by at least this gap
#     "gap_accept_margin": 0.1,
#     # Explicit alias used by output_manager (falls back to soft_accept_margin if absent)
#     "low_confidence_margin": 0.05,
#     # Boosting knobs for intelligent_mapper._run_graph_boost_pass
#     "confidence_threshold": 0.6,
#     "boost_amount": 0.3,
#     "enable_neighbor_window_boost": True,
#     "neighbor_window": 2,
#     "neighbor_boost_amount": 0.05,
#     "neighbor_max_boost": 0.2,
# }

# Option 2: Local SentenceTransformer Model (default - no API key required)
SEMANTIC_MAPPING_CONFIG = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",  # Lightweight embedding model (80MB vs 420MB)
    "provider": "sentence_transformer",
    "similarity_threshold": 0.6,
    "device": "cpu",  # Force CPU for consistent performance
    "batch_size": 16,  # Reduced batch size for lower memory usage
    "top_k_candidates": 3,
    "cache_folder": "./models",  # Use local models cache
    # Accept borderline matches within this margin below the threshold
    "soft_accept_margin": 0.05,
    # Also accept if the top-1 similarity exceeds top-2 by at least this gap
    "gap_accept_margin": 0.1,
    # Explicit alias used by output_manager (falls back to soft_accept_margin if absent)
    "low_confidence_margin": 0.05,
    # Boosting knobs for intelligent_mapper._run_graph_boost_pass
    "confidence_threshold": 0.6,
    "boost_amount": 0.3,
    "enable_neighbor_window_boost": True,
    "neighbor_window": 2,
    "neighbor_boost_amount": 0.05,
    "neighbor_max_boost": 0.2,
}

# LLM-Enhanced Chunking Configuration
LLM_CHUNK_CONFIG = {
    # The minimum number of characters a paragraph chunk must have to be considered for a semantic split.
    "SEMANTIC_SPLIT_THRESHOLD": 800,
    # Whether to enable this feature. Allows for easy toggling for performance.
    "ENABLE_LLM_ENHANCEMENT": False
}

# --- NEW: Embedding-guided cohesion (Phase 1, OFF by default) ---
EMBEDDING_COHESION_CONFIG = {
    "ENABLE": False,                  # When True, chunker can compute per-chunk cohesion_score using embeddings
    "method": "intra_sentence_avg",  # Future-proof: how cohesion is computed
    "min_sentences": 2               # Only compute if chunk has at least this many sentences
}

# --- NEW: Unified Embedding Configuration ---
# Current: Local Sentence Transformer Model (default)
EMBEDDING_CONFIG = {
    "provider": "sentence_transformer",
    "model": "sentence-transformers/all-mpnet-base-v2",
    "device": "auto",
    "batch_size": 32
}

# Alternative API-based embedding configurations (uncomment to use):
# OpenAI Embedding Configuration
# EMBEDDING_CONFIG = {
#     "provider": "openai",
#     "model": "text-embedding-3-large",  # or "text-embedding-ada-002", "text-embedding-3-small"
#     "api_key_env": "OPENAI_API_KEY",
#     "device": "auto",
#     "batch_size": 32
# }

# Cohere Embedding Configuration
# EMBEDDING_CONFIG = {
#     "provider": "cohere",
#     "model": "embed-english-v3.0",  # or "embed-multilingual-v3.0"
#     "api_key_env": "COHERE_API_KEY",
#     "device": "auto",
#     "batch_size": 32
# }

# Hugging Face API Embedding Configuration
# EMBEDDING_CONFIG = {
#     "provider": "huggingface",
#     "model": "sentence-transformers/all-mpnet-base-v2",
#     "api_key_env": "HUGGINGFACE_API_KEY",
#     "device": "auto",
#     "batch_size": 32
# }

# --- NEW: Embedding-guided chunking configuration ---
CHUNKING_EMBEDDING = {
    "enable": True,
    "cohesion_threshold": 0.7,
    "max_tokens_per_chunk": 300,
    "overlap_tokens": 50,
    "fallback_provider": "sentence_transformer",
    "boundary_detection_method": "cohesion_minima",
    "adaptive_sizing": True,
    "smart_overlap": True,
    "local_model": "sentence-transformers/all-mpnet-base-v2",
    "device": "auto"
}

# --- NEW: Adaptive chunking configuration ---
ADAPTIVE_CHUNKING_CONFIG = {
    "enable": True,
    "min_chunk_size": 100,
    "max_chunk_size": 1000,
    "target_chunk_size": 500,
    "overlap_ratio": 0.1,
    "cohesion_threshold": 0.7,
    "adaptive_threshold": 0.8,
    "boundary_detection": "semantic",
    "use_embeddings": True,
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "device": "auto"
}

# --- NEW: Local LLM refinement configuration ---
LOCAL_LLM_REFINEMENT = {
    "enable": True,
    "only_for_low_confidence": True,
    "confidence_threshold": 0.6,
    "max_cases_per_doc": 10,
    "refinement_model": "mistralai/Mistral-7B-v0.3",
    "temperature": 0.1,
    "max_tokens": 1024,
    "device": "auto"
}

# --- Alias for test compatibility ---
MISTRAL_REFINEMENT = LOCAL_LLM_REFINEMENT

# Output format configurations
OUTPUT_FORMATS = {
    "latex": {
        "extension": ".tex",
        "description": "LaTeX document format"
    },
    "markdown": {
        "extension": ".md",
        "description": "Markdown document format"
    },
    "json": {
        "extension": ".json",
        "description": "JSON document format"
    }
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
        {"section": "Major and Minor Assumptions", "prompt": "Format as LaTeX with section title 'Major and Minor Assumptions'. Preserve all technical content exactly.", "description": "A section to consolidate all the key assumptions, both major and minor, that the system's security and functionality rely on."},
        {"section": "Calculations", "prompt": "Format as LaTeX with section title 'Calculations'. Preserve all technical content exactly.", "description": "Presents the mathematical analysis and calculations, particularly regarding the probability of an attacker catching up to the honest chain."},
        {"section": "Conclusion", "prompt": "Format as LaTeX with section title 'Conclusion'. Preserve all technical content exactly.", "description": "The concluding section summarizing the benefits of the proposed electronic cash system, such as eliminating the need for trust and protecting sellers."}
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
        "description": "The introductory section of the document, providing background, problem statement, and proposed solution.",
        "dynamic_description": True,
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
                " Privacy Model": {
                    "prompt": "You are a LaTeX expert. Refactor this content to focus specifically on the privacy model...",
                    "description": "Addresses the privacy model of the system...",
                    "chunks": [],
                    "subsections": {}
                },
                " Attack Vector Calculations": {
                    "prompt": "You are a LaTeX expert and mathematician. Refactor the following text and equations...",
                    "description": "Presents the mathematical analysis of the system's security against an attacker...",
                    "chunks": [],
                    "subsections": {}
                }
            }
        },
        # --- NEW ASSUMPTIONS SECTION ---
        "Assumptions": {
            "prompt": "You are a system analyst and LaTeX expert. Based on the document content, provide a brief introductory paragraph for a section that will outline the core assumptions the system relies on. This paragraph should set the stage for the major and minor subsections.",
            "description": "A section dedicated to explicitly stating the underlying assumptions required for the Bitcoin protocol to function securely and effectively. This includes assumptions about network behavior and participant honesty.",
            "chunks": [],
            "subsections": {
                " Major Assumptions": {
                    "prompt": "You are a system analyst and LaTeX expert. From the provided text, extract and clearly articulate the most critical assumptions for the system's security. The primary assumption is that honest nodes control a majority of CPU power. Explain the implications of this assumption.",
                    "description": "Details the most critical, foundational assumptions of the system. The foremost assumption is that the majority of the CPU power in the network is controlled by honest nodes that are not conspiring to attack the network.",
                    "chunks": [],
                    "subsections": {}
                },
                " Minor Assumptions": {
                    "prompt": "You are a system analyst and LaTeX expert. From the provided text, identify and list any secondary or implicit assumptions. This could include assumptions about network latency (nodes receive broadcasts in a timely manner) or participant behavior (nodes are economically rational).",
                    "description": "Outlines other, less critical but still important, assumptions. This includes assumptions such as nodes having reliable network connectivity, the practicality of storing block headers for SPV, and that participants generally act in their own economic self-interest.",
                    "chunks": [],
                    "subsections": {}
                }
            }
        },
        "Conclusion": {
            "prompt": "You are a LaTeX expert. Refactor the following content into a strong, formal conclusion...",
            "description": "The concluding section that summarizes the paper's proposal...",
            "chunks": [],
            "subsections": {}
        },
        "References": {
            "prompt": "You are a LaTeX expert. Format the following content as a standard 'References' section...",
            "description": "The list of citations and prior work referenced in the paper...",
            "chunks": [],
            "subsections": {}
        },
        "System Tests": {
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

    "quant_finance_research_paper": {
    "1. Abstract": {
        "generative": True,
        "description": "A high-level summary of the paper's methodology, findings, and implications.",
        "persona_prompts": {
            "default": """
                You are a senior quantitative researcher summarizing a complex financial model for an academic journal.
                Focus on the core contribution, the methodology, and the key empirical results.
            """
        }
    },
    "2. Literature Review": {
        "description": "Discusses prior academic work, models like Black-Scholes, GARCH, or Fama-French, and identifies the gap this paper fills.",
        "persona_prompts": {
            "default": """
                You are a finance academic with deep knowledge of historical models.
                Refactor the text to position the paper within the existing literature, highlighting its novel contribution.
            """,
            "consistency_check": """
                You are a meticulous fact-checker. Review the following text. Does it accurately represent the contributions of the cited papers?
                Are there any mischaracterizations of well-known models like Black-Scholes or CAPM?
            """
        }
    },
    "3. Methodology & Model Specification": {
        "description": "The mathematical core of the paper. Details the stochastic processes, equations, and statistical methods used.",
        "persona_prompts": {
            "default": """
                You are a mathematician and statistician. Refactor the following text for maximum precision and clarity.
                Ensure all equations are correctly referenced and the derivation of the model is logically sound.
            """,
            "notation_check": """
                You are a LaTeX typesetting expert specializing in mathematical notation.
                Review the following text. Is the notation for variables (e.g., σ for volatility, r for risk-free rate) consistent and standard?
                Correct any inconsistencies.
            """
        }
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
You are a professional technical editor. Refactor the following LaTeX content for maximum clarity, conciseness, and professional academic tone.

**Context for consistency:**
Previous content: {memory_context}
Related excerpts: {semantic_context}
Document abstract: {global_context}
Parent section: {parent_context}

**Section:** "{node_path}"

**LaTeX content to refactor:**
{node_content}

**Requirements:**
- Preserve ALL LaTeX commands, environments, citations, and references exactly
- Maintain original technical meaning
- Apply formatting only to specific words, not entire paragraphs
- Output ONLY the refactored LaTeX content with no additional text

""",

    # NEW PROMPT for the self-critique pass
    "self_critique_and_refine": """
You are a quality assurance editor. Improve the following refactored text for maximum clarity, conciseness, and professional academic tone.

**Document Section:** "{node_path}"

**Text to Improve:**
{refactored_text}

**Instructions:**
1. Enhance clarity and remove redundant phrases
2. Ensure smooth logical flow between ideas
3. Preserve all LaTeX commands, citations, and technical details exactly
4. Maintain formal academic tone
5. Output ONLY the improved text with no explanations or preamble

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
     
     **Official Document Glossary:**
    {domain_glossary}
    
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
You are a professional technical editor. Refactor the following LaTeX content for maximum clarity, conciseness, and professional academic tone.

**Context for consistency:**
Previous content: {memory_context}
Related excerpts: {semantic_context}
Document abstract: {global_context}
Parent section: {parent_context}

**Section:** "{node_path}"

**LaTeX content to refactor:**
{node_content}

**Requirements:**
- Preserve ALL LaTeX commands, environments, citations, and references exactly
- Maintain original technical meaning
- Apply formatting only to specific words, not entire paragraphs
- Output ONLY the refactored LaTeX content with no additional text

""",

 'semantic_split_paragraph': """
 You are an expert in document analysis. Your task is to split the following text into semantically coherent paragraphs.
 Do not lose any information. The output must be a list of strings in the specified JSON format.

 {format_instructions}

 TEXT TO SPLIT:
 ---
 {text_content}
 ---
 """,
  'generate_dynamic_description': """
        You are an expert document analyst. I will provide you with the full text of a document. Your task is to write a dense, keyword-rich, one-paragraph description for the section titled '{section_title}'.

        This description will be used by an AI to perform a semantic search, so it is critical that it contains the core concepts, specific terminology, problems, and solutions mentioned in that part of the document.

        Do NOT describe the section in general terms. Analyze the provided text and extract the key semantic themes. Respond with ONLY the descriptive paragraph and nothing else.

        **Full Document Text:**
        ---
        {full_text_excerpt}
        ---

        **Generated Description for '{section_title}':**
    """,

'content_augmentation_contrast': """
    You are a research analyst comparing two academic papers on the same topic.
    
    **Content from the Base Document (Document A) on '{section_name}':**
    ---
    {original_content}
    ---

    **Content from the Augmentation Document (Document B) on '{section_name}':**
    ---
    {augmentation_content}
    ---

    **Your Task:**
    Rewrite the content for this section by integrating the insights from both documents.
    Your primary goal is to **highlight the differences**. Start by presenting the view from Document A, then introduce the perspective from Document B using phrases like "In contrast," "An alternative approach suggests," or "However, [Author B] proposes...".
    Synthesize the two viewpoints into a coherent, academic discussion.
""",

'suggest_cross_reference': """
    You are a helpful academic editor.
    The following two text excerpts from a document are highly related in meaning but are not explicitly linked.
    
    **Source Excerpt (from section '{source_section}'):**
    ---
    {source_content}
    ---

    **Related Excerpt (from section '{target_section}'):**
    ---
    {target_content}
    ---

    Your task is to rewrite the final sentence of the "Source Excerpt" to include a natural, academic-style cross-reference to the target section.
    For example: "...which is closely related to the methodology discussed in Section {target_section}."
    Respond with ONLY the rewritten sentence.
""",

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



QUANT_FINANCE_GLOSSARY = {
    "key_terms": [
        "Stochastic Volatility", "Monte Carlo Simulation", "Risk-Neutral Pricing",
        "Arbitrage Opportunity", "Efficient Frontier", "Capital Asset Pricing Model (CAPM)",
        "Black-Scholes-Merton (BSM)", "Geometric Brownian Motion (GBM)"
    ],
    "acronyms": {
        "VaR": "Value at Risk",
        "CVaR": "Conditional Value at Risk",
        "HFT": "High-Frequency Trading",
        "APT": "Arbitrage Pricing Theory"
    }
}

# --- NEW: Knowledge Graph (KG) unified view configuration (Phase 2) ---
KG_CONFIG = {
    "ENABLE_KG_UNIFIED": True,        # Build unified KG when requested
    "SEMANTIC_TOP_K": 5,              # Top-K semantic neighbors per node
    "SEMANTIC_THRESHOLD": 0.80,       # Threshold for semantic edges
    "DUPLICATE_THRESHOLD": 0.95,      # Stricter threshold for duplicate edges
    "DUMP_ON_ANALYSIS": True,         # When running run_analysis, also dump the KG to disk
    "DUMP_FORMAT": "json",           # Options: "json" (node-link) or "graphml"
    "DUMP_FILENAME": "knowledge_graph_unified.json",  # File name for dump within the session folder
    
    # KG-Enhanced Section Mapping
    "enhance_section_mapping": True,
    "kg_weight": 0.4,  # Weight for KG scores (0.0-1.0)
    "embedding_weight": 0.6,  # Weight for embedding similarity (should sum to 1.0 with kg_weight)
    
    # KG Score Components (weights for composite score)
    "centrality_weight": 0.3,
    "cohesion_weight": 0.2, 
    "semantic_connectivity_weight": 0.3,
    "structural_importance_weight": 0.2,
    
    # Section Affinity Parameters
    "use_section_affinity": True,
    "affinity_boost_factor": 1.2  # Multiplier for KG-based section affinity
}

# === LOCAL MODEL CONFIGURATION ===
# Configuration for running models locally without external APIs
LOCAL_MODEL_CONFIG = {
    "profile": "performance",  # Performance profile for MISTRAL
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # Lightweight embedding model
    "llm_model": "mistralai/Mistral-7B-v0.3",  # MISTRAL model
    "device": "auto",  # Auto-detect best device (GPU if available, else CPU)
    "batch_size": 1,  # Conservative batch size for MISTRAL
    "cache_dir": "./models",  # Local model cache directory
    "download_timeout": 600,  # Extended timeout for MISTRAL
    "enable_gpu_if_available": True,  # Enable GPU for better performance with MISTRAL
    "memory_optimization": True
}

# Hugging Face Configuration for local models
HUGGINGFACE_CONFIG = {
    "model_name": "mistralai/Mistral-7B-v0.3",  # MISTRAL model
    "max_tokens": 2048,
    "timeout": 300,
    "torch_dtype": "float16",  # Optimized for MISTRAL
    "load_in_8bit": True,  # Enable quantization for MISTRAL
    "profile": "performance",  # Performance profile for MISTRAL
    "batch_size": 1,  # Conservative batch size for MISTRAL
    "download_timeout": 600,  # 10 minutes for MISTRAL
}

# Hardware requirements for different profiles
HARDWARE_REQUIREMENTS = {
    "lightweight": {
        "min_ram_gb": 4,
        "min_disk_gb": 2,
        "gpu_required": False,
        "description": "Runs on most systems, basic performance"
    },
    "balanced": {
        "min_ram_gb": 8,
        "min_disk_gb": 5,
        "gpu_required": False,
        "gpu_recommended": True,
        "description": "Good balance of performance and resource usage"
    },
    "high_quality": {
        "min_ram_gb": 16,
        "min_disk_gb": 10,
        "gpu_required": True,
        "description": "Best performance, requires dedicated GPU"
    }
}