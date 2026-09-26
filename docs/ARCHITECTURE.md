# Architecture

Jev RAG keeps a vector-free default while exposing an optional hybrid retrieval
path. Both paths use the same remote decision-model reranker and optional answer
generator.

```text
documents
   |
   v
parsers -> heading-aware passages -> SQLite documents/passages tables
                                      |
query -> CJK/Latin lexical tokens -> FTS5 MATCH -> BM25 top 30 --------+
                                                                     |
optional hybrid: BM25 top 50 + embedding top 50 -> RRF top 50 -------+
                                                                     |
                                                                     v
                                  Jev Noul relevance judgments
                                  (batches of 10, up to 4 workers)
                                                    |
                                      threshold -> top 10 evidence passages
                                                    |
                                                    v
                                  OpenRouter chat completion with citations
```

## Components

### Ingestion

`KnowledgeBase.index()` scans supported files, skips configured glob patterns, extracts text, and creates passages according to the selected chunking mode. File hashes allow unchanged files to avoid unnecessary work.

### Lexical retrieval

SQLite FTS5 supplies BM25 ranking. Latin text is normalized with NFKC and lowercased. Continuous Chinese text is represented with overlapping CJK bigrams; short Chinese spans are also retained whole. Query terms are deduplicated and the first 80 unique terms are joined with `OR`.

This design favors recall. Conversational filler may therefore retrieve weak candidates; Jev is the second-stage relevance judge.

### Optional hybrid retrieval

With `retrieval_mode=hybrid`, the application requests passage and query
embeddings from OpenRouter, normalizes the vectors locally, retrieves by cosine
similarity, and combines the vector and BM25 rankings using reciprocal rank
fusion. The default configuration uses BM25 top 50, vector top 50, RRF
`k=60`, and keeps 50 fused candidates for Jev.

The corpus matrix is cached under `.knowledge/embeddings/` and invalidated by
the embedding model or passage-content fingerprint. There is no vector database.
The default `bm25` mode never loads NumPy or calls the embedding endpoint.

### Jev reranking

Each candidate receives a `Noul` question asking whether it contains concrete evidence useful for the query. Candidates are divided into groups of 10. Groups run concurrently and their absolute scores are merged, then ties fall back to the first-stage retrieval order.

The full reranking result is cached by provider, query, passage identity, and batching-policy version. A threshold may remove weak evidence after scoring.

### Answer generation

The answer model receives only the selected evidence, file paths, headings, and line ranges. The system prompt requires numbered citations and an explicit insufficient-evidence response. Generation uses OpenRouter's streaming chat-completions endpoint.

### Persistence

The SQLite database contains:

- source document metadata;
- searchable passages and the FTS5 index;
- Jev reranking cache entries;
- answer-run metrics, usage, cost, sources, and generated text.

The schema is internal until version 1.0. Rebuild the index if an incompatible development version changes it.

## Trust boundaries

| Data | Location |
|---|---|
| Original files | Local filesystem |
| FTS index and run history | Local SQLite database |
| Normalized embedding matrix (hybrid only) | Local `.knowledge/embeddings/` cache |
| Query and BM25 candidate excerpts | Sent to the configured Jev provider |
| Passage text and query (hybrid only) | Sent to the configured OpenRouter embedding model |
| Query and selected evidence | Sent to OpenRouter for answer generation |
| API keys | Process environment or local `.env` |

## Design tradeoffs

- BM25 is fast and inspectable but cannot directly capture semantic paraphrases.
- Hybrid retrieval improves semantic recall but adds an embedding cost, a first-run indexing delay, and another remote-data boundary.
- Jev improves evidence selection but adds network latency, cost, and a remote-data boundary.
- Batching reduces context-limit failures; scores from separate batches may not be perfectly comparable.
- Passing more evidence can improve recall while increasing answer latency, cost, and distraction.
- The built-in HTTP server is intentionally local and single-process. It is not a production web server.

## Extension points

- Add a query-cleaning or query-expansion stage before `lexical_search()`.
- Replace the answer model while preserving `answer_messages()` and the stream event contract.
- Add new extractors in `extract_text()`.
- Add a calibrated filtering profile or evaluation harness around `jev_score`.
- Split the current modules into a `src/jev_rag/` package if the public Python API grows.
