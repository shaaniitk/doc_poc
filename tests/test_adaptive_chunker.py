import re
import pytest

from modules.chunker import AdaptiveChunker


def test_atomic_blocks_preserved_equation():
    text = r"""
    Intro text before equation.
    \begin{equation}
    E = mc^2
    \end{equation}
    Some text after equation.
    """
    ch = AdaptiveChunker()
    chunks = ch.chunk(text, doc_type='latex', atomic_blocks=('equation',), max_tokens=60)

    # Ensure that no chunk splits the equation environment
    begin_count = sum('\\begin{equation}' in c['content'] for c in chunks)
    end_count = sum('\\end{equation}' in c['content'] for c in chunks)
    assert begin_count == end_count == 1

    # If a chunk contains begin, it must also contain end (same chunk)
    for c in chunks:
        if '\\begin{equation}' in c['content']:
            assert '\\end{equation}' in c['content']


def test_token_limit_creates_multiple_chunks():
    # Create a long text with many short sentences
    sentence = "This is a sentence about alpha. "
    text = sentence * 200
    ch = AdaptiveChunker()
    chunks = ch.chunk(text, doc_type='markdown', max_tokens=80, prefer_sweeping=False)

    # With a small token limit, we should get more than one chunk
    assert len(chunks) > 1
    # Chunks should be non-empty and ordered
    assert all(c['content'].strip() for c in chunks)


def test_topic_shift_helps_split_sections():
    text = r"""
    \section{Alpha}
    Alpha is discussed here. alpha alpha alpha. More alpha discussion.
    \section{Beta}
    Beta is discussed here. beta beta beta. More beta discussion.
    """
    ch = AdaptiveChunker()
    chunks = ch.chunk(text, doc_type='latex', max_tokens=120, prefer_sweeping=False)

    # Expect at least two chunks, and that early chunks focus on Alpha while later on Beta
    assert len(chunks) >= 2
    contents = [c['content'].lower() for c in chunks]

    # First chunk(s) should be dominated by 'alpha', last by 'beta'
    assert contents[0].count('alpha') >= contents[0].count('beta')
    assert contents[-1].count('beta') >= contents[-1].count('alpha')