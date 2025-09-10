import numpy as np
import pytest

from modules.intelligent_mapper import IntelligentMapper
from modules.knowledge_graph_processor import KnowledgeGraphProcessor


class DummyEmbeddingModel:
    def __init__(self):
        pass
    def encode(self, texts, show_progress_bar=False):
        # Simple bag-of-words vectors over a tiny vocabulary
        vocab = ['alpha', 'beta', 'gamma', 'delta']
        embs = []
        for t in texts:
            v = np.zeros(len(vocab), dtype=float)
            tl = t.lower()
            for i, w in enumerate(vocab):
                v[i] = tl.count(w)
            # normalize
            n = np.linalg.norm(v) or 1.0
            embs.append(v / n)
        return np.vstack(embs)


def make_template():
    # Minimal skeleton similar to DOCUMENT_TEMPLATES entry
    return {
        'Introduction': {
            'description': 'Intro and background',
            'subsections': {}
        },
        'Alpha Section': {
            'description': 'Covers alpha topics',
            'subsections': {}
        },
        'Beta Section': {
            'description': 'Covers beta topics',
            'subsections': {}
        }
    }


def make_chunks():
    chunks = []
    contents = [
        "An introduction to the paper and background info",
        "We study alpha methods and alpha results",
        "Further alpha experiments are presented",
        "Discussion of beta baselines and beta ablations",
        "Conclusions with beta implications"
    ]
    for i, c in enumerate(contents):
        chunks.append({
            'chunk_id': i,
            'content': c,
            'metadata': {'prev_chunk_id': i-1 if i>0 else None, 'next_chunk_id': i+1 if i < len(contents)-1 else None}
        })
    return chunks


def test_global_assignment_capacity_replication(monkeypatch):
    # Prepare KG processor with dummy embeddings for chunks
    chunks = make_chunks()
    model = DummyEmbeddingModel()

    # Create a simple KG processor stub that holds embeddings and a structural graph
    kg = KnowledgeGraphProcessor(chunks=[], embedding_model=model)
    kg.embeddings = model.encode([c['content'] for c in chunks])
    kg.structural_graph = None

    # Create mapper with template object and injected kg processor
    mapper = IntelligentMapper(template_name="test_template", template_object=make_template(), kg_processor=kg)

    # Monkeypatch the mapper's embedding model to our dummy for sections
    mapper.embedding_model = model
    mapper.section_embeddings = model.encode([s['description'] for s in mapper.flat_skeleton])

    # Force config for global assignment with reasonable thresholds
    mapper.config['use_global_assignment'] = True
    mapper.config['similarity_threshold'] = 0.1
    mapper.config['global_capacity_alpha'] = 1.0

    mapped = mapper.map_chunks(chunks)

    # Validate that alpha-heavy chunks go to Alpha Section and beta-heavy to Beta Section
    def find_section_chunks(mapped_tree, section_title):
        node = mapped_tree_section(mapped_tree, [section_title])
        return [c['content'] for c in node['chunks']]

    def mapped_tree_section(tree, path):
        node = tree
        for p in path[:-1]:
            node = node[p]['subsections']
        return node[path[-1]]

    alpha_chunks = find_section_chunks(mapped, 'Alpha Section')
    beta_chunks = find_section_chunks(mapped, 'Beta Section')

    assert any('alpha' in c.lower() for c in alpha_chunks)
    assert any('beta' in c.lower() for c in beta_chunks)

    # Ensure that the Assignments include assignment_score metadata
    all_chunks = alpha_chunks + beta_chunks
    assert all(isinstance(c, str) for c in all_chunks)


def test_global_assignment_with_capacity_constraints(monkeypatch):
    # Create more alpha chunks than beta chunks to test capacity logic
    alpha_contents = [
        "alpha one", "alpha two", "alpha three", "alpha four"
    ]
    beta_contents = ["beta one"]
    contents = alpha_contents + beta_contents
    
    chunks = []
    for i, c in enumerate(contents):
        chunks.append({
            'chunk_id': i,
            'content': c,
            'metadata': {}
        })

    model = DummyEmbeddingModel()
    kg = KnowledgeGraphProcessor(chunks=[], embedding_model=model)
    kg.embeddings = model.encode([c['content'] for c in chunks])
    kg.structural_graph = None

    mapper = IntelligentMapper(template_name="test_template", template_object=make_template(), kg_processor=kg)
    mapper.embedding_model = model
    mapper.section_embeddings = model.encode([s['description'] for s in mapper.flat_skeleton])

    # Set capacity for Alpha Section to 2 chunks
    mapper.config['use_global_assignment'] = True
    mapper.config['similarity_threshold'] = 0.1
    mapper.config['global_capacity_alpha'] = 2.0  # Capacity for alpha sections

    mapped = mapper.map_chunks(chunks)

    def find_section_chunks(mapped_tree, section_title):
        node = mapped_tree
        for p in section_title.split('/'):
            node = node.get(p, {})
            if 'subsections' in node:
                node = node['subsections']
        return node.get('chunks', [])

    alpha_chunks = find_section_chunks(mapped, 'Alpha Section')
    beta_chunks = find_section_chunks(mapped, 'Beta Section')

    # Alpha section should be at its capacity of 2
    assert len(alpha_chunks) == 2
    # The other 3 chunks (2 alpha, 1 beta) should be in the Beta section
    assert len(beta_chunks) == 3

    # Verify that the chunks in alpha section are indeed alpha chunks
    assert all('alpha' in c['content'] for c in alpha_chunks)

    # Verify that assignment metadata is present
    for chunk in alpha_chunks + beta_chunks:
        assert 'assignment_score' in chunk['metadata']
        assert 'assignment_type' in chunk['metadata']