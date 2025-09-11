from modules.chunker import _tokens, _jaccard, _topic_boundaries

# Test sentences from the failing test
sents = ['alpha alpha alpha.', 'alpha beta alpha.', 'alpha beta beta.', 'beta beta beta.']

print("Testing Jaccard similarity:")
for i in range(len(sents) - 1):
    s1_tokens = _tokens(sents[i])
    s2_tokens = _tokens(sents[i + 1])
    sim = _jaccard(s1_tokens, s2_tokens)
    print(f"Sentence {i+1}: {repr(sents[i])} -> tokens: {s1_tokens}")
    print(f"Sentence {i+2}: {repr(sents[i+1])} -> tokens: {s2_tokens}")
    print(f"Jaccard similarity: {sim}")
    print(f"Below threshold 0.25: {sim < 0.25}")
    print(f"Below threshold 0.4: {sim < 0.4}")
    print()

# Test topic boundaries
boundaries_25 = _topic_boundaries(sents, 0.25)
boundaries_40 = _topic_boundaries(sents, 0.4)
print(f"Boundaries with threshold 0.25: {boundaries_25}")
print(f"Boundaries with threshold 0.4: {boundaries_40}")