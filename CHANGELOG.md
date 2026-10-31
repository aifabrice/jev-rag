# Changelog

All notable changes are documented here. The project follows Semantic Versioning after the initial alpha series.

## [0.4.0] - 2026-10-31

This is the first public release after `v0.3.0`. It consolidates the internal
0.4–0.7 development milestones, including the additional retrieval pipelines,
reliability, security, benchmark, and documentation work that landed on `main`.
The default UI remains the smaller BM25/embedding candidate pool; high-recall
and deep-search variants are explicit experiments.

The public release sequence is therefore `v0.2.0` → `v0.3.0` → `v0.4.0`.
The intermediate milestone numbers were never published as Git tags or GitHub
Releases.

### Highlights

- Seven selectable retrieval paths: BM25 + Jev, agentic lexical search,
  embedding hybrid, multi-round Agentic Hybrid, taxonomy routing, unified Jev
  Passage Gate, and hierarchical Jev Line Search.
- A reproducible no-key BM25 quickstart, a local web UI with streaming cited
  answers, and English/Chinese tutorials and project pages.
- Full BEIR NFCorpus reporting on 3,633 documents and 323 queries, including
  quality, latency, cost, cache state, and negative-result disclosures.

### Added

- A seventh `agentic-hybrid` retrieval mode: two rounds of model-planned
  lexical searches feed local multi-query BM25 while the original query's
  embedding lookup runs in parallel; weighted RRF keeps 50 candidates for the
  existing Jev reranker.
- Web UI, CLI, latency reporting, source-rank inspection, benchmark flags, and
  regression tests for the new pipeline.
- Complete BEIR NFCorpus disclosure for both the pre-Jev retrieval order and
  the final Jev ranking, plus a machine-readable summary.
- Optional post-rerank RRF between the Jev order and the original retrieval
  order. Agentic Hybrid uses the dev-selected `1.0:0.25` ratio by default and
  can disable it with `--jev-retrieval-prior-weight 0`.
- Experimental Jev batch-size and candidate-text controls for benchmark
  robustness checks; the production default remains ten candidates per call.

### Changed

- Agentic planner responses with empty, truncated, or invalid JSON now retry
  with progressively larger output limits and a less restrictive fallback.
- The default Agentic-Hybrid fusion weight is `0.65:1.0` for Agentic versus
  dense retrieval, selected on the NFCorpus dev split before one fixed test
  evaluation.
- Persist exact query embeddings by query and model so repeated Hybrid runs
  skip a redundant provider request without changing vectors or rankings.
- Generalize Agentic retrieval to any positive configured round count while
  keeping two rounds as the default.
- Allow benchmark and Agentic branches to evaluate candidate pools up to 500,
  enabling the published top-200 high-recall operating point.

### Benchmark

- Agentic-Hybrid retrieval reached `0.424145` nDCG@10 and `0.333047`
  Recall@50 on the full 323-query NFCorpus test split.
- Agentic-Hybrid + Jev reached `0.445761` nDCG@10 before post-rerank prior
  fusion, only `+0.001434` over Hybrid + Jev. A paired
  20,000-sample bootstrap interval (`[-0.006883, 0.009834]`) crosses zero, so
  the result is reported as statistically tied rather than a clear win.
- The two-round planner added substantial latency: the cold retrieval-only run
  measured a `6.99 s` median and `24.39 s` p95.
- Dev-selected Jev/retrieval rank fusion raised the fixed test result to
  `0.450750` nDCG@10 without another provider call. Its paired 20,000-sample
  bootstrap delta interval versus Jev-only ordering was
  `[0.000397, 0.009757]`.
- The high-recall Agentic Hybrid variant (BM25 + embedding top 200, Jev batches
  of 100) reached `0.466171` nDCG@10, `0.669516` MRR@10, and `0.228912`
  Recall@10. The quality run recorded 3.05 s median, 4.24 s p95, and
  `$1.421625` provider cost; a unified cold audit measured 6.52 s median and
  7.90 s p95 because the quality run had cache hits.
- Added `benchmarks/nfcorpus-pipeline-latency-audit10-summary.json` with a
  unified 10-query cold latency audit, cache state, and stage breakdown.

### Reliability and security

- Rebuild the index when its configured document root changes and reject
  stale-root searches when automatic indexing is disabled.
- Treat an answer stream that closes without a provider completion signal as an
  error instead of recording a partial answer as successful.
- Ignore document symlinks whose resolved target is outside the selected
  knowledge root.
- Pin every GitHub Action to a full commit SHA, add weekly dependency checks
  and repository-wide CODEOWNERS coverage, and keep secret-scanner output
  aggregate-only.
- Install the embedding extra in CI and compile `taxonomy.py` explicitly
  during bytecode validation.

### Documentation and reproducibility

- Publish architecture, seven-pipeline, FAQ, benchmark, latency, and
  contribution guides with machine-readable benchmark artifacts.
- Keep the benchmark honest about provider-dependent latency, cold versus
  cached runs, experimental modes, and the boundary between local retrieval
  and remote Jev/answer-provider calls.

## [0.3.0] - 2026-09-26

### Added

- Zero-configuration local-folder discovery: `serve` now indexes `~/Documents`
  by default, keeps BM25 + Jev as the vector-free default, and excludes the
  source checkout when it is inside the discovered folder.
- Selectable `bm25`, `agentic`, and `hybrid` retrieval modes in the CLI, HTTP API, and web UI.
- Two-round MiniMax lexical query planning, multi-query RRF, local plan caching,
  Agentic latency/cost records, and defensive planner-output validation.
- Optional OpenRouter embeddings, local NumPy vector cache, and reciprocal-rank fusion before Jev.
- BM25, Agentic, embedding/RRF, Jev, first-token, generation, and total latency reporting.
- Complete NFCorpus Agentic and hybrid benchmark results with reproducible configurations.
- Interactive public benchmark explorer deployed with GitHub Pages.

## [0.2.0] - 2026-09-26

### Added

- Local web UI with streaming MiniMax answers and numbered citations.
- End-to-end BM25, Jev, first-token, generation, and total latency reporting.
- Persistent answer-run records with usage and cost metadata.
- Configurable folder exclusion patterns.
- BM25 heading weighting and Chinese bigram tokenization.
- Jev network retries and concurrent batches of ten candidates.
- Defaults of 30 BM25 candidates and 10 final evidence passages.
- Optional Jev threshold for CLI and web-server workflows.
- Repository screenshots and a social preview asset.
- Reproducible BM25/Jev retrieval benchmark runner and evaluation guide.
- English and Chinese project pages with architecture comparison and community guidance.

### Fixed

- Recover from transient TLS and retryable provider failures.
- Avoid Jev context-limit failures caused by sending all 30 candidates in one request.

## [0.1.0] - 2026-09-24

- Initial local SQLite FTS5/BM25 index, Jev reranking, CLI search, and provider smoke test.
