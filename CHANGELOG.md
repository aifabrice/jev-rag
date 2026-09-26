# Changelog

All notable changes are documented here. The project follows Semantic Versioning after the initial alpha series.

## [Unreleased]

- Prepare the repository for a public open-source release.

## [0.2.0] - 2026-09-25

### Added

- Local web UI with streaming MiniMax answers and numbered citations.
- End-to-end BM25, Jev, first-token, generation, and total latency reporting.
- Persistent answer-run records with usage and cost metadata.
- Configurable folder exclusion patterns.
- BM25 heading weighting and Chinese bigram tokenization.
- Jev network retries and concurrent batches of ten candidates.
- Defaults of 30 BM25 candidates and 10 final evidence passages.
- Optional Jev threshold for CLI and web-server workflows.

### Fixed

- Recover from transient TLS and retryable provider failures.
- Avoid Jev context-limit failures caused by sending all 30 candidates in one request.

## [0.1.0] - 2026-09-24

- Initial local SQLite FTS5/BM25 index, Jev reranking, CLI search, and provider smoke test.
