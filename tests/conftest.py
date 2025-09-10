import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import types

# Stub heavy external dependencies to keep test environment lean.
# sentence-transformers
st = types.ModuleType('sentence_transformers')
class _DummySTModel:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Simple deterministic mapping: vector length equals token count
        import numpy as np
        vecs = []
        for t in texts:
            # crude token count
            n = max(1, len(str(t).split()))
            vecs.append(np.ones(8) * n)
        return np.vstack(vecs)

def SentenceTransformer(*args, **kwargs):
    return _DummySTModel()

st.SentenceTransformer = SentenceTransformer
sys.modules['sentence_transformers'] = st

# langchain stubs
langchain_mod = types.ModuleType('langchain')
text_splitter_mod = types.ModuleType('langchain.text_splitter')
class _DummySplitter:
    def __init__(self, *args, **kwargs):
        pass
    def split_text(self, text):
        return [text]
text_splitter_mod.MarkdownHeaderTextSplitter = _DummySplitter
sys.modules['langchain'] = langchain_mod
sys.modules['langchain.text_splitter'] = text_splitter_mod

# Add langchain.prompts and langchain.chains stubs
prompts_mod = types.ModuleType('langchain.prompts')
class PromptTemplate:
    def __init__(self, input_variables=None, template=""):
        self.input_variables = input_variables or []
        self.template = template
    def format(self, **kwargs):
        return self.template.format(**kwargs)
prompts_mod.PromptTemplate = PromptTemplate
chains_mod = types.ModuleType('langchain.chains')
class LLMChain:
    def __init__(self, llm=None, prompt=None):
        self.llm = llm
        self.prompt = prompt
    def invoke(self, inputs):
        # Return empty mapping to avoid affecting logic
        return {"text": ""}
chains_mod.LLMChain = LLMChain
sys.modules['langchain.prompts'] = prompts_mod
sys.modules['langchain.chains'] = chains_mod

# docx stub
sys.modules.setdefault('docx', types.ModuleType('docx'))

# sklearn stubs if needed
sk_mod = types.ModuleType('sklearn')
cluster_mod = types.ModuleType('sklearn.cluster')
class _DummyKMeans:
    def __init__(self, *args, **kwargs):
        self.labels_ = []
    def fit(self, X):
        import numpy as np
        n = len(X)
        self.labels_ = [0] * n
        return self
cluster_mod.KMeans = _DummyKMeans
# Add sklearn.metrics.pairwise.cosine_similarity stub
metrics_mod = types.ModuleType('sklearn.metrics')
pairwise_mod = types.ModuleType('sklearn.metrics.pairwise')

def _cosine_similarity(X, Y):
    import numpy as np
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9
    Xn = X / X_norm
    Yn = Y / Y_norm
    return Xn @ Yn.T

pairwise_mod.cosine_similarity = _cosine_similarity
sys.modules['sklearn'] = sk_mod
sys.modules['sklearn.cluster'] = cluster_mod
sys.modules['sklearn.metrics'] = metrics_mod
sys.modules['sklearn.metrics.pairwise'] = pairwise_mod

# modules.llm_client stub
llm_client_mod = types.ModuleType('modules.llm_client')
class _DummyLLMClient:
    def __init__(self, *args, **kwargs):
        pass
    def complete(self, *args, **kwargs):
        return ""
# Also provide UnifiedLLMClient and LangChainLLM names used in code
class UnifiedLLMClient(_DummyLLMClient):
    pass
class LangChainLLM:
    def __init__(self, client=None):
        self.client = client
    def __call__(self, prompt):
        return ""
llm_client_mod.LLMClient = _DummyLLMClient
llm_client_mod.UnifiedLLMClient = UnifiedLLMClient
llm_client_mod.LangChainLLM = LangChainLLM
sys.modules['modules.llm_client'] = llm_client_mod

# jinja2 stub (minimal) to satisfy imports if needed
jinja2_mod = types.ModuleType('jinja2')
class _DummyTemplate:
    def __init__(self, name):
        self.name = name
    def render(self, **kwargs):
        # Very naive rendering used only if code insists on Jinja2
        processed_tree = kwargs.get('processed_tree', {})
        # If latex template
        if self.name.endswith('.tex.j2'):
            body = []
            for k, v in processed_tree.items():
                body.append(f"\\section{{{k}}}\n{v if isinstance(v, str) else ''}")
            return "\\begin{document}\n" + "\n".join(body) + "\n\\end{document}"
        # If markdown template
        if self.name.endswith('.md.j2'):
            body = []
            for k, v in processed_tree.items():
                body.append(f"# {k}\n{v if isinstance(v, str) else ''}")
            return "\n\n".join(body)
        return ""
class Environment:
    def __init__(self, loader=None):
        self.loader = loader
        # Allow attributes assignment used by code
        self.block_start_string = '{%'
        self.block_end_string = '%}'
        self.variable_start_string = '{{'
        self.variable_end_string = '}}'
        self.comment_start_string = '{#'
        self.comment_end_string = '#}'
    def get_template(self, name):
        return _DummyTemplate(name)
class FileSystemLoader:
    def __init__(self, *args, **kwargs):
        pass
jinja2_mod.Environment = Environment
jinja2_mod.FileSystemLoader = FileSystemLoader
sys.modules['jinja2'] = jinja2_mod

# networkx stub (minimal Graph API)
nx_mod = types.ModuleType('networkx')
class _DummyGraph:
    def __init__(self):
        self._adj = {}
    def add_edge(self, u, v):
        self._adj.setdefault(u, set()).add(v)
        self._adj.setdefault(v, set()).add(u)
    def neighbors(self, u):
        return list(self._adj.get(u, []))
class Graph(_DummyGraph):
    pass
class DiGraph(_DummyGraph):
    pass
nx_mod.Graph = Graph
nx_mod.DiGraph = DiGraph
sys.modules['networkx'] = nx_mod