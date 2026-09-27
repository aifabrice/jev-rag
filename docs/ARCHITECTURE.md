# Architecture

Jev RAG keeps a vector-free default while exposing optional agentic lexical,
hybrid embedding, multi-round Agentic Hybrid, corpus-taxonomy routing, unified Passage Gate, and
hierarchical Jev Line Search paths.
BM25, Agentic, and standard hybrid retrieval share the same decision-model
reranker. Hybrid Gate replaces that reranker with one four-question decision
stage. Line Search uses Jev Choice + Noul as retrieval itself.

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
optional agentic-hybrid: two-round multi-BM25 + embedding -> RRF 50 -+
                                                                     |
optional taxonomy: preserve Hybrid top 50 + routed extras (<=20) ----+
                                                                     |
                                                                     v
                                  Jev Noul relevance judgments
                                  (batches of 10, up to 8 workers)
                                                    |
                         Agentic Hybrid only: fuse Jev rank + retrieval rank
                                      (weights 1.0:0.25, no provider call)
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

alternative hybrid-gate path:

BM25 top 50 + embedding top 50 -> RRF top 50 -> one Jev Passage Gate
                                                       |
                         relevance + evidence + contradiction + injection
                                                       |
                         include | conflicting evidence | exclude
                                                       |
                                  OpenRouter answer with separated evidence
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

### Optional corpus taxonomy routing

With `retrieval_mode=taxonomy`, the application reuses the normalized corpus
embedding matrix to build a deterministic two-level spherical-k-means tree.
Representative terms make nodes inspectable. Each passage receives a primary
leaf and may receive a secondary assignment when it lies near a leaf boundary.
The JSON tree and NPZ centroids are cached under `.knowledge/taxonomy/` and are
invalidated by the embedding model, corpus fingerprint, or tree parameters.

At query time, the query embedding selects leaf nodes until four leaves and at
least 200 assigned passages are covered. BM25 and dense retrieval run inside
that routed subset. Their fused results do not replace or reorder the global
Hybrid top 50: up to 20 unique routed candidates are appended, then the whole
pool enters the ordinary Jev reranker. This fail-open design protects Hybrid
recall when routing is wrong. It also means more Jev calls and a larger remote
data boundary. The public NFCorpus result improved candidate recall but did
not improve final nDCG@10, so the mode is experimental.

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

### Optional multi-round Agentic Hybrid retrieval

With `retrieval_mode=agentic-hybrid`, the application first loads or builds the
same locally cached corpus embedding matrix used by Hybrid mode. It then starts
the original query's embedding request in a worker while the main thread runs
both Agentic lexical planning rounds and their local BM25 searches. The complete
Agentic RRF ranking is treated as one source and the dense ranking as another;
weighted RRF (`agentic=0.65`, `vector=1.0`, `k=60`) keeps 50 candidates for the
ordinary Jev reranker. After Jev, Agentic Hybrid performs one local RRF between
the Jev rank and original retrieval rank (`1.0:0.25`, `k=60`). This prior-aware
ordering makes no additional provider call and can be disabled with
`--jev-retrieval-prior-weight 0`.

Generated lexical searches are not embedded. This prevents ten additional
remote embedding calls from multiplying cost and avoids counting the original
BM25 ranking twice. Round two remains sequential because it uses first-round
titles and snippets. The pre-prior NFCorpus result was statistically tied with
Hybrid + Jev. The dev-selected prior-aware order reached `0.450750` test
nDCG@10, while planning still substantially increased latency.

### Jev reranking

Each candidate receives a `Noul` question asking whether it contains concrete evidence useful for the query. Candidates are divided into groups of 10. Groups run concurrently and their absolute scores are merged, then ties fall back to the first-stage retrieval order. Benchmark-only controls can vary batch size and candidate text length; the default remains 10 candidates and 1,800 characters per candidate.

The full reranking result is cached by provider, query, passage identity, and batching-policy version. A threshold may remove weak evidence after scoring.

### Optional unified Passage Gate

With `retrieval_mode=hybrid-gate`, the first stage is identical to Hybrid:
BM25 top 50 and embedding top 50 are fused into 50 RRF candidates. Ordinary
Jev reranking is not run. Instead, a single batched Jev stage asks four `Noul`
questions for every passage: relevance, usable answer evidence, contradiction
of a factual premise in the query, and prompt injection.

The fixed routing profile excludes injection above 0.70, routes contradiction
above 0.70 to a separate conflict block, excludes relevance below 0.45,
includes answer evidence above 0.55, and excludes the rest. Normal evidence
and conflicting evidence are separately labeled in the generator prompt.
Responses are cached per batch of ten candidates and up to four batches run
concurrently. This is one Jev decision stage, not reranking followed by a
second gate.

The prompt-injection judgment is probabilistic and is not a security boundary.
The fixed thresholds also proved too aggressive on NFCorpus: 84.4% of
candidates were excluded and nDCG@10 fell below bare Hybrid. The mode is
therefore experimental and deliberately not the default.

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
| Query and fused candidate excerpts (hybrid-gate only) | Sent to Jev for four judgments per passage |
| Corpus passages and queries (taxonomy only) | Sent to the configured embedding provider; the tree and vectors remain local |
| Query and first-round snippets (agentic only) | Sent to the configured OpenRouter planner |
| Query, first-round snippets, and corpus/query text (agentic-hybrid only) | Sent to the configured planner and embedding provider; only the original query is embedded online |
| Query and a bounded representation of every indexed passage (line-search only) | Sent to the configured Jev provider in windows; finalists are sent again |
| Query and selected evidence | Sent to OpenRouter for answer generation |
| API keys | Process environment or local `.env` |

## Design tradeoffs

- BM25 is fast and inspectable but cannot directly capture semantic paraphrases.
- Hybrid retrieval improves semantic recall but adds an embedding cost, a first-run indexing delay, and another remote-data boundary.
- Agentic retrieval improves lexical recall without embeddings, but adds planner calls, query latency, provider cost, and a remote-data boundary.
- Agentic Hybrid improved the measured candidate pool and its local
  post-rerank prior improved the measured final order without more calls, but
  two sequential planning rounds still dominate latency.
- Jev improves evidence selection but adds network latency, cost, and a remote-data boundary.
- Unified Passage Gate adds conflict and injection routing, but four judgments
  per passage increase output tokens and fixed thresholds can reduce recall.
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
