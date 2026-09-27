# Changelog

All notable changes are documented here. The project follows Semantic Versioning after the initial alpha series.

## [Unreleased]

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
