import pytest
from modules.chunker import AdaptiveChunker


def test_very_long_unbreakable_line():
    """Tests that a long line with no natural breaks is still chunked correctly."""
    long_word = "a" * 500
    text = f"Start. {long_word} End."
    ch = AdaptiveChunker()
    chunks = ch.chunk(text, doc_type='markdown', max_tokens=50)
    assert len(chunks) > 1
    assert chunks[0]['content'].startswith('Start.')
    assert chunks[-1]['content'].endswith('End.')

def test_multiple_atomic_environments():
    """Tests that multiple different atomic environments are all preserved."""
    text = r"""
    \begin{equation}E=mc^2\end{equation}
    Some text in between.
    \begin{verbatim}print('hello')\end{verbatim}
    """
    ch = AdaptiveChunker()
    chunks = ch.chunk(text, doc_type='latex', atomic_blocks=('equation', 'verbatim'), max_tokens=40)
    
    equation_chunk_found = False
    verbatim_chunk_found = False
    for c in chunks:
        content = c['content']
        if '\\begin{equation}' in content:
            assert '\\end{equation}' in content
            equation_chunk_found = True
        if '\\begin{verbatim}' in content:
            assert '\\end{verbatim}' in content
            verbatim_chunk_found = True
            
    assert equation_chunk_found
    assert verbatim_chunk_found

def test_gradual_topic_drift():
    """Tests if the chunker can detect a slow, gradual shift in topic."""
    # Text with more distinct topic shift
    text = "mathematics calculus derivatives integrals. physics quantum mechanics relativity. chemistry molecules atoms bonds. biology cells organisms evolution."
    ch = AdaptiveChunker()
    # We expect the sweeping window to detect the shift between different topics
    chunks = ch.chunk(text, doc_type='markdown', max_tokens=20, prefer_sweeping=True)
    
    assert len(chunks) >= 2
    # Check that different topics are separated
    all_content = " ".join([chunk['content'] for chunk in chunks])
    assert "mathematics" in all_content
    assert "physics" in all_content
    assert "chemistry" in all_content
    assert "biology" in all_content