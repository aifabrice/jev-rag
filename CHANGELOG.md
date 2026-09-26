# Changelog

All notable changes are documented here. The project follows Semantic Versioning after the initial alpha series.

## [Unreleased]

### Added

- Zero-configuration local-folder discovery: `serve` now indexes `~/Documents`
  by default, keeps BM25 + Jev as the vector-free default, and excludes the
  source checkout when it is inside the discovered folder.
- Selectable `bm25` and `hybrid` retrieval modes in the CLI, HTTP API, and web UI.
- Optional OpenRouter embeddings, local NumPy vector cache, and reciprocal-rank fusion before Jev.
- BM25, embedding/RRF, Jev, first-token, generation, and total latency reporting.
- Complete NFCorpus hybrid benchmark result and clearly qualified MTEB numerical context.

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
