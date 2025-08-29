# Document Refactoring, Orphan Handling, and Similarity Distance — Enhancement Plan

This plan upgrades the pipeline to state-of-the-art behavior for mapping, rescuing, and presenting orphaned chunks, while exposing similarity “distance” signals for analysis and UI/LaTeX rendering.

## Objectives
- Compute and persist per-chunk similarity profiles: top-k candidate sections with scores and distance metrics.
- Improve orphan handling with richer signals for downstream remediation (LLM or deterministic passes).
- Expose this data in outputs (JSON and LaTeX) to guide manual review or further automation.
- Keep the system modular, configurable, and compatible with existing pipeline stages.

## Phased Plan

### Phase 1 — Core Similarity Signals (Implemented Now)
1. Compute top-k candidate sections for every chunk during mapping, including:
   - path, path_str, score (cosine), distance (1 − cosine), nearest_section_suggestion, distance_to_nearest.
2. Persist these fields in chunk.metadata for both mapped and orphaned chunks.
3. Add configuration knob SEMANTIC_MAPPING_CONFIG.top_k_candidates (default: 3).
4. Ensure these signals appear in the saved mapped JSON (2_mapped_tree.json) for quick inspection.

Acceptance criteria:
- JSON shows candidate_sections for each chunk with the fields above.
- No regressions in mapping behavior or pipeline stability.

### Phase 2 — Orphan Presentation & Reviewer UX (Next)
5. Enhance LaTeX template to render orphaned chunks with a compact “Nearest Sections” table:
   - Show top-3 candidates (Section path, Similarity, Distance).
   - Keep the orphan’s original content unchanged below the table.
6. Optionally flag low-confidence mapped chunks (near threshold) with a footnote and their nearest alternative.

Acceptance criteria:
- final_document.tex includes a clearly formatted Orphaned Content section with suggestions.
- Compiles cleanly; no LaTeX conflicts.

### Phase 3 — Advanced Rescue Strategies (Optional, SOTA)
7. Neighborhood-boosted assignment refinement:
   - Use sliding window of adjacent chunks to re-score borderline orphans (already partially available via cohesion pass).
8. Graph-aware suggestion ranking:
   - Boost candidates using citation/ref reference edges to sections where their neighbors cluster.
9. Cluster-driven dynamic subsections:
   - For large sections with heterogeneous content, auto-cluster orphans + low-confidence chunks into coherent dynamic subsections; re-run mapping within that localized skeleton (leveraging existing dynamic_subsections capability).

Acceptance criteria:
- Reduced orphan rate with measurable improvement (e.g., >20% fewer orphans on benchmark docs).
- Deterministic behavior when LLM is disabled.

### Phase 4 — LLM-Aided Remediation (Refinement)
10. Retrieval-augmented LLM pass:
    - Provide candidate_sections and local neighbors as RAG context to improve assignment precision and reduce hallucination.
11. Self-critique pass for borderline decisions:
    - Ask LLM to reconsider when distance_to_nearest is close to threshold or when candidates are semantically close.

Acceptance criteria:
- Higher precision on tough mappings with bounded token usage and timeouts.

### Phase 5 — Analytics & Quality Metrics
12. Post-run analysis report (JSON/Markdown) summarizing:
    - Orphan count, average assignment score, distribution, top ambiguous sections, rescued vs. final orphans.
13. Optional export to CSV for offline analysis.

Acceptance criteria:
- run_analysis outputs include new similarity and orphan diagnostics.

## Rollout Strategy
- Start with Phase 1 (done) to avoid UI changes. Validate outputs in 2_mapped_tree.json.
- Proceed to Phase 2 to improve LaTeX review UX with minimal risk.
- Gate later phases behind config flags for performance and reproducibility.

## What’s Implemented in This Commit
- Per-chunk top-k candidate sections with scores and distances are computed and persisted during mapping (and in the internal semantic pass), controlled by SEMANTIC_MAPPING_CONFIG.top_k_candidates.
- These signals are available for both mapped chunks and orphans and will appear in saved JSON artifacts for immediate inspection.

## Next Actions
- Update the LaTeX template to render orphan suggestions (Phase 2, step 5).
- Optionally flag low-confidence assignments in LaTeX with a footnote (Phase 2, step 6).
- Add basic analytics to run_analysis (Phase 5).

## Validation Checklist
- Re-run the pipeline through the “map” stage and open 2_mapped_tree.json:
  - Verify candidate_sections exists for each chunk with scores and distances.
  - Check nearest_section_suggestion and distance_to_nearest are sensible.
- Run full pipeline and confirm final_document.tex compiles and the output is unchanged in structure (until Phase 2 is applied).