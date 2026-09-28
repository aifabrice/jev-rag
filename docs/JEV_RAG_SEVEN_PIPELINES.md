# Jev RAG Is Not One Pipeline: What 323 Queries Taught Me About BM25, Agentic Search, Embeddings, and Reranking

I started Jev RAG with a narrow question: how far can a local knowledge-search system go before it needs an embedding index or a vector database?

The first version was intentionally simple:

```text
local files -> SQLite FTS5/BM25 -> Jev reranking -> grounded answer
```

That baseline was useful, but it also made the failure modes obvious. A reranker can improve the order of retrieved passages, but it cannot recover evidence that never entered the candidate pool. Over the next two days I expanded the project into seven selectable retrieval paths and evaluated them on the same public test set.

The result is not a story about keywords defeating embeddings, or Jev replacing retrieval. It is a more practical lesson:

> Retrieval, judgment, ranking policy, and generation are separate layers. Treating them separately makes both improvements and failures measurable.

## The evaluation setup

All headline results use the complete BEIR NFCorpus test split:

- 3,633 documents;
- 323 test queries;
- the same corpus, queries, qrels, and metric implementation for every row;
- no relevance labels supplied to query planning or Jev prompts;
- project-run results, not an official MTEB submission.

Provider cost and latency numbers are observations from these runs, not universal prices or service-level guarantees. Network conditions, provider routing, cache state, and model versions can change them materially.

## The seven pipelines

| Pipeline | Candidate retrieval | Jev's role | Vector index? | Intended use |
| --- | --- | --- | --- | --- |
| BM25 + Jev | SQLite FTS5/BM25 | Passage relevance reranking | No | Smallest and most transparent baseline |
| Agentic lexical + Jev | Two planning rounds, multi-query BM25, RRF | Rerank the expanded lexical pool | No | Improve vocabulary coverage without document embeddings |
| Hybrid + Jev | BM25 + embeddings + RRF | Rerank lexical and semantic candidates | Local cached matrix | Strong quality/latency compromise |
| Agentic Hybrid + Jev | Multi-round lexical planning + embeddings + weighted RRF | Rerank, then blend Jev and retrieval ranks | Local cached matrix | Highest measured overall ranking quality |
| Taxonomy + Hybrid + Jev | Two-level corpus taxonomy expands Hybrid candidates | Rerank up to 70 candidates | Local cached matrix and tree | Route through structured corpora |
| Hybrid + Passage Gate | Hybrid candidate pool | Judge relevance, evidence, contradiction, and injection risk | Local cached matrix | Evidence governance experiments |
| Two-level Jev Line Search | Parallel Jev Choice windows over the full index | Search and globally select finalists | No | Expensive deep search for small, high-value corpora |

These modes vary three independent decisions:

1. how candidates enter the pool;
2. what Jev is asked to decide;
3. how the final order combines semantic judgment with retrieval priors.

## Results on the same 323 queries

| Pipeline | nDCG@10 | MRR@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| BM25 top 30 | 0.305654 | 0.512697 | 0.147309 |
| BM25 top 30 + Jev | 0.353235 | 0.585817 | 0.158667 |
| BM25 top 50 + Jev | 0.362468 | 0.593023 | 0.164474 |
| Two-level Jev Line Search | 0.366280 | **0.657660** | 0.169397 |
| Hybrid + Unified Passage Gate | 0.376298 | 0.618043 | 0.166977 |
| Agentic lexical top 50 | 0.380168 | 0.597940 | 0.185464 |
| BM25 + embedding RRF top 50 | 0.396712 | 0.632089 | 0.193977 |
| Multi-round Agentic Hybrid top 50 | 0.424145 | 0.637722 | 0.206303 |
| Agentic lexical top 50 + Jev | 0.430969 | 0.644041 | 0.204138 |
| Hybrid top 50 + Jev | 0.444327 | **0.654583** | 0.214907 |
| **Agentic Hybrid + Jev/retrieval rank fusion** | **0.450750** | 0.652606 | **0.220885** |

The ranking table is useful, but the transitions between rows explain more than the final winner.

## 1. BM25 + Jev: the smallest useful system

The default pipeline uses SQLite only:

```text
local files -> FTS5/BM25 top 30 -> Jev -> answer model
```

Adding Jev to BM25 top 30 raised nDCG@10 from `0.305654` to `0.353235`, a relative improvement of `15.57%`. MRR@10 rose from `0.512697` to `0.585817`.

The candidate-pool recall did not change. That is expected: reranking only changes the order of documents that were already retrieved.

Increasing the pool from 30 to 50 candidates improved nDCG@10 again, from `0.353235` to `0.362468`, but the observed median reranking latency increased from about `1.08 s` to `1.92 s`. The complete-run Jev cost increased from about `$0.164` to `$0.261`.

That is why top 30 remains the default. It is not a magic number; it was the better operating point for this dataset.

Use this path when documents contain stable product names, identifiers, API terms, regulations, or other explicit vocabulary, and when infrastructure simplicity matters.

## 2. Agentic lexical + Jev: improve recall without document embeddings

Lexical retrieval fails when the query and the document describe the same thing with different words. The Agentic mode addresses that mismatch without building a document embedding index.

The planning model has a deliberately narrow job: generate search queries for the local BM25 index.

```text
question
  -> round 1: five lexical queries
  -> local BM25 for every query
  -> inspect up to eight titles and snippets
  -> round 2: five corrective queries
  -> multi-query BM25 + RRF
  -> top 50 candidates
  -> Jev reranking
```

Before Jev, Agentic lexical retrieval reached `0.380168` nDCG@10. Its Recall@50 improved from ordinary BM25's `0.209810` to `0.280765`. After Jev, nDCG@10 reached `0.430969`.

That is `18.90%` above BM25 top 50 + Jev. It still trails Hybrid + Jev by `0.013358`, and a paired bootstrap interval favored Hybrid, so I do not treat the two systems as equivalent.

The cost is online planning. Across the complete run, planning and Jev together cost about `$0.476`. The estimated serial online median was about `7.52 s`.

This path makes sense when the corpus changes often, queries contain synonyms or colloquial language, and avoiding an embedding index matters more than minimizing per-query latency.

## 3. Hybrid + Jev: the strongest practical default for quality

Hybrid retrieval combines lexical precision with semantic coverage:

```text
BM25 top 50 ---------+
                     +-> RRF top 50 -> Jev
embedding top 50 ----+
```

BM25 + embedding RRF reached `0.396712` nDCG@10 before Jev and raised Recall@50 from `0.209810` to `0.318075`. Jev then improved nDCG@10 to `0.444327`.

The important causal order is candidate recall first, reranking second. Embeddings recover relevant documents that lexical search missed; only then can Jev move them toward the top.

For this 3,633-document corpus, the one-time embedding build took about 285 seconds and cost about `$0.162`, after which the matrix was cached locally. The complete Jev stage cost about `$0.368`. A ten-query cold end-to-end trial had a median of about `3.17 s`.

For most deployments that can accept an embedding boundary, this is the safest high-quality choice in the current implementation.

## 4. Agentic Hybrid: the best measured score came from rank fusion

Agentic Hybrid runs multi-round lexical planning alongside the original-query embedding lookup, then uses weighted RRF:

```text
two-round Agentic BM25 -----+
                            +-> weighted RRF top 50 -> Jev
original-query embedding ---+
                                                   |
                                      Jev rank + retrieval rank
```

The Agentic and dense branches use a dev-selected weight of `0.65:1.0`. This raised Recall@50 to `0.333047` and pre-Jev nDCG@10 to `0.424145`.

But Jev alone did not create a decisive improvement over ordinary Hybrid:

- Hybrid + Jev: `0.444327`;
- Agentic Hybrid + Jev: `0.445761`.

The meaningful gain came from a local, zero-provider-call post-processing step:

```text
final score = 1.0 * Jev rank + 0.25 * retrieval rank
```

That reached `0.450750` nDCG@10. The `0.004989` improvement over Jev-only ordering had a 20,000-sample paired bootstrap interval of `[0.000397, 0.009757]`.

The result suggests that semantic judgment and retrieval priors contain complementary information. Replacing one completely with the other throws some of that information away.

This was the highest measured quality point, but it was not cheap or fast. Cold retrieval had a median of about `6.99 s` and p95 of `24.39 s`; planning and Jev together cost about `$0.537`, excluding the cached corpus-embedding build.

## 5. Taxonomy: higher candidate recall did not improve final ranking

The Taxonomy mode builds a deterministic two-level tree from cached corpus embeddings, routes each query to four leaf nodes, and appends up to 20 unique candidates to the unchanged Hybrid top 50.

It did improve candidate coverage:

- Hybrid Recall@50: `0.318075`;
- Taxonomy Recall@70: `0.342964`;
- 121 queries received at least one relevant document missing from the original pool.

Final nDCG@10 was `0.441851`, slightly below Hybrid + Jev at `0.444327`.

More candidates raised the ceiling, but they also made the ranking problem harder. Higher candidate recall is necessary for improvement, but it is not sufficient.

## 6. Unified Passage Gate: a useful negative result

The Passage Gate replaces ordinary relevance reranking with four independent judgments per passage:

1. is it relevant?
2. does it contain answer-supporting evidence?
3. does it conflict with a premise in the query?
4. does it contain prompt-injection risk?

Application code combines those answers into `include`, `conflicting evidence`, and `exclude` routes.

On NFCorpus, the fixed policy excluded 13,631 of 16,150 candidates, or `84.4%`. Final nDCG@10 was `0.376298`, below both bare Hybrid (`0.396712`) and Hybrid + Jev (`0.444327`).

This does not make the four judgments useless. It means a fixed safety-and-evidence policy should not be confused with a universally optimal ranking policy. Thresholds need calibration on the actual deployment objective.

## 7. Two-level Jev Line Search: strong first-hit behavior, high cost

Line Search removes BM25 and embeddings entirely. It divides the full index into windows, asks Jev Choice to select finalists from each window, then runs a global Choice over those finalists.

```text
all passages
  -> 15 parallel windows
  -> four finalists per window
  -> 60 global candidates
  -> final Jev Choice
```

It produced the strongest first-hit behavior: nDCG@1 was `0.554180`, and MRR@10 was `0.657660`. But nDCG@10 was only `0.366280`, with Recall@10 of `0.169397`.

The method behaved more like “find the single best answer” than “build a consistently ranked evidence set.” It was also the most expensive path: 5,168 requests, about 108 million input tokens, and `$4.553` observed cost for the complete cold run. A ten-query cold trial had a median of about `11.74 s` and p95 of `35.75 s`.

It remains an interesting deep-search option for small corpora and high-value questions, not a general default.

## What I would choose in practice

| Priority | Recommended mode | Why |
| --- | --- | --- |
| Minimum infrastructure | BM25 + Jev | SQLite only; easy to inspect and measure |
| No vector index, better vocabulary coverage | Agentic lexical + Jev | Multi-query lexical search recovers wording mismatches |
| Strong quality with manageable latency | Hybrid + Jev | Best current practical compromise |
| Maximum measured ranking quality | Agentic Hybrid | Best nDCG@10, but high planning latency |
| Structured corpus routing | Taxonomy | Improves pool coverage; final ranking still needs work |
| Evidence and safety policy | Passage Gate | Useful judgments, but thresholds require task-specific calibration |
| Small-corpus deep search | Line Search | Strong first hit, highest cost and latency |

My default progression is deliberately boring:

1. establish an inspectable BM25 + Jev baseline;
2. add Agentic lexical search if failures come from vocabulary mismatch;
3. add embeddings if candidate recall remains the ceiling;
4. reserve Agentic Hybrid for questions whose value justifies several seconds of planning;
5. treat Taxonomy, Passage Gate, and Line Search as objective-specific experiments.

## The broader lesson

Jev is most useful here as a bounded decision layer, not as retrieval magic and not as the answer model.

- BM25, embeddings, Agentic search, and Taxonomy decide what can be considered.
- Jev decides which bounded candidates contain useful evidence.
- Application code owns thresholds, fusion, caching, and safety boundaries.
- The answer model organizes the selected evidence into cited prose.

Keeping those layers separate makes it possible to identify the actual failure: weak query terms, insufficient candidate recall, a bad Jev ordering, an over-aggressive gate, or a generator that ignored its evidence.

The seven modes are valuable not because a UI has seven options, but because they expose seven measurable engineering trade-offs inside the same reproducible system.

- Repository: <https://github.com/aifabrice/jev-rag>
- Interactive benchmark: <https://aifabrice.github.io/jev-rag/>
- Full configuration, costs, caveats, and reproduction commands: <https://github.com/aifabrice/jev-rag/blob/main/benchmarks/NFCORPUS_RESULTS.md>

The project is independent and is not affiliated with or endorsed by TypeSafe AI, OpenRouter, or MiniMax.
