# Changelog

All notable changes are documented here. The project follows Semantic Versioning after the initial alpha series.

## [Unreleased]

### Security

- Pin every GitHub Action to a full commit SHA while retaining the upstream
  major version in a comment for Dependabot updates.
- Add weekly dependency update checks for Python packages and GitHub Actions,
  plus repository-wide CODEOWNERS coverage.

### Fixed

- Install the embedding extra in CI so the taxonomy test suite has NumPy on
  every supported Python version.
- Compile `taxonomy.py` explicitly during the CI bytecode validation step.

## [0.7.0] - 2026-09-27

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

## [0.6.0] - 2026-09-27

### Added

- An experimental sixth `taxonomy` mode that builds a deterministic two-level
  corpus taxonomy from cached embeddings, routes each query to relevant leaf
  nodes, and appends up to 20 unique node-local candidates to the unchanged
  Hybrid top-50 pool before ordinary Jev reranking.
- Persistent JSON/NPZ taxonomy caches, readable node labels, primary and
  near-boundary secondary document assignments, UI/CLI timing, and benchmark
  controls. Taxonomy construction uses corpus documents only, never benchmark
  queries or relevance judgments.
- Complete BEIR NFCorpus disclosure: taxonomy expansion increased candidate
  recall from 0.318075 at 50 to 0.342964 at 70, but final nDCG@10 was 0.441851,
  slightly below Hybrid + Jev at 0.444327. The mode remains experimental and
  is not the default.

### Changed

- Jev reranking can execute up to eight independent ten-candidate batches in
  parallel, reducing the extra wait introduced by 70-candidate taxonomy pools.

## [0.5.0] - 2026-09-27

### Added

- A fifth `hybrid-gate` mode: BM25 top 50 and embedding top 50 are fused with
  RRF, then one unified Jev stage evaluates relevance, usable answer evidence,
  contradiction of a query premise, and prompt injection for every candidate.
- Official-cookbook-inspired routing with separate normal and conflicting
  evidence blocks for answer generation, plus prompt-injection exclusion.
- CLI, HTTP, web UI, cache, benchmark, latency, route-count, and cost support
  for the unified Passage Gate.
- Complete 323-query BEIR NFCorpus disclosure. The fixed gate scored 0.376298
  nDCG@10 versus 0.396712 for bare Hybrid and 0.444327 for Hybrid + Jev, while
  excluding 84.4% of candidates; it is therefore experimental and not default.

## [0.4.0] - 2026-09-27

### Added

- A fourth `line-search` retrieval mode based on TypeSafe's Line-by-line Search
  cookbook: indexed passages are partitioned into windows of at most 255,
  every window is searched concurrently with Jev Choice + Noul, and a second
  Choice request globally ranks the window finalists.
- Configurable window size, finalists per window, local caching, per-stage
  latency/cost metadata, CLI support, HTTP support, and a web UI selector for
  two-level Line Search.
- Capacity checks for the two-level `255 × 255 = 65,025` passage hierarchy and
  explicit documentation of its cost and remote-data boundary.
- Complete 323-query BEIR NFCorpus Line Search results, including first-hit,
  ranking, recall, provider usage, cost, latency caveats, and reproduction data.
- Per-window cache persistence so successful parallel stages survive a sibling
  timeout or connection failure and a resumed benchmark only retries misses.

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
