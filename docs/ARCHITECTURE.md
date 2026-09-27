# Architecture

Jev RAG keeps a vector-free default while exposing optional agentic lexical,
hybrid embedding, and hierarchical Jev Line Search paths. BM25, Agentic, and
hybrid retrieval share the same decision-model reranker. Line Search uses Jev
Choice + Noul as retrieval itself.

```text
documents
   |
   v
parsers -> heading-aware passages -> SQLite documents/passages tables
                                      |
query -> CJK/Latin lexical tokens -> FTS5 MATCH -> BM25 top 30 --------+
                                                                     |
optional agentic: planner -> multi-query BM25 -> RRF top 50 ---------+
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

alternative line-search path:

all indexed passages -> windows of <=255 -> parallel Jev Choice + Noul
                                             |
                                  up to 4 finalists per window
                                             |
                                             v
                              global Jev Choice + Noul (<=255)
                                             |
                                             v
                                  OpenRouter answer with citations
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

### Optional agentic lexical retrieval

With `retrieval_mode=agentic`, MiniMax first generates five compact lexical
queries using abbreviations, synonyms, and likely document terminology. The
application executes the original and generated queries against the same local
FTS5 index, fuses their rankings with RRF, then shows up to eight untrusted
result snippets to a second planning round. A final RRF top 50 enters Jev.

No embeddings or vector index are created. The planner receives the user query
and second-round snippets through OpenRouter, so this mode is not offline. Plans
are cached by model, query, round, prior queries, and observations. Planner JSON
is schema-checked, bounded, and defensively repaired for common truncation
errors. Retrieved snippets are explicitly treated as untrusted data.

### Jev reranking

Each candidate receives a `Noul` question asking whether it contains concrete evidence useful for the query. Candidates are divided into groups of 10. Groups run concurrently and their absolute scores are merged, then ties fall back to the first-stage retrieval order.

The full reranking result is cached by provider, query, passage identity, and batching-policy version. A threshold may remove weak evidence after scoring.

### Optional two-level Line Search

With `retrieval_mode=line-search`, every indexed passage participates in Jev
retrieval through a bounded passage representation. Passages are partitioned
into at most 255 stable windows containing
at most 255 passages each. Every window runs a `Choice` question to rank its
passages and a `Noul` question to decide whether it contains an answer. Windows
run concurrently. By default, the best four passages from every window advance;
that count is reduced automatically when needed so the second-level Choice
never exceeds 255 options. A final Choice + Noul request globally ranks the
finalists.

The hierarchy has a structural capacity of 65,025 passages; this is not a
practical latency or cost guarantee. Cost grows with the whole corpus because a
bounded representation of every passage is sent to the configured Jev provider.
Stage responses are cached by provider, query, passage identity, and hierarchy
version.

The implementation follows the TypeSafe
[Semantic Find / line-by-line search cookbook](https://docs.typesafe.ai/cookbooks/semantic_find),
then adds the parallel fan-out and global-reduce level needed for larger local
corpora.

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
| Agentic search plans | Local SQLite cache |
| Normalized embedding matrix (hybrid only) | Local `.knowledge/embeddings/` cache |
| Query and BM25 candidate excerpts | Sent to the configured Jev provider |
| Passage text and query (hybrid only) | Sent to the configured OpenRouter embedding model |
| Query and first-round snippets (agentic only) | Sent to the configured OpenRouter planner |
| Query and a bounded representation of every indexed passage (line-search only) | Sent to the configured Jev provider in windows; finalists are sent again |
| Query and selected evidence | Sent to OpenRouter for answer generation |
| API keys | Process environment or local `.env` |

## Design tradeoffs

- BM25 is fast and inspectable but cannot directly capture semantic paraphrases.
- Hybrid retrieval improves semantic recall but adds an embedding cost, a first-run indexing delay, and another remote-data boundary.
- Agentic retrieval improves lexical recall without embeddings, but adds planner calls, query latency, provider cost, and a remote-data boundary.
- Jev improves evidence selection but adds network latency, cost, and a remote-data boundary.
- Line Search avoids lexical and embedding retrieval, but evaluates the full
  corpus remotely and is substantially more expensive than shortlist reranking.
- Batching reduces context-limit failures; scores from separate batches may not be perfectly comparable.
- Passing more evidence can improve recall while increasing answer latency, cost, and distraction.
- The built-in HTTP server is intentionally local and single-process. It is not a production web server.

## Extension points

- Add a query-cleaning or query-expansion stage before `lexical_search()`.
- Replace the answer model while preserving `answer_messages()` and the stream event contract.
- Add new extractors in `extract_text()`.
- Add a calibrated filtering profile or evaluation harness around `jev_score`.
- Split the current modules into a `src/jev_rag/` package if the public Python API grows.
