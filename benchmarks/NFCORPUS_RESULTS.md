# BEIR NFCorpus result

This is a reproducible Jev RAG project result on the public BEIR NFCorpus test
split. It is not a claim of certification or endorsement by BEIR, TypeSafe, or
OpenRouter.

## Run configuration

- Date: 2026-09-26
- Report version: Jev RAG 0.7.0 (individual experiments below record their
  implementation version)
- Corpus: 3,633 documents
- Test queries: 323
- Dataset archive SHA-256:
  `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`
- Index: SQLite FTS5/BM25, one BEIR corpus record per local text document
- Candidate pool: BM25 top 30
- Reranker: `typesafe/jev-1.13-20260917` through OpenRouter
- Final ranking: all 30 candidates sorted by Jev score, with BM25 rank as the
  tie breaker

## Retrieval quality

| Metric | BM25 | BM25 + Jev | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| nDCG@1 | 0.407637 | 0.489164 | +0.081527 | +20.00% |
| nDCG@10 | 0.305654 | 0.353235 | +0.047581 | +15.57% |
| MRR@10 | 0.512697 | 0.585817 | +0.073120 | +14.26% |
| MAP@10 | 0.218445 | 0.264411 | +0.045966 | +21.04% |
| Recall@10 | 0.147309 | 0.158667 | +0.011358 | +7.71% |
| Recall@30 | 0.189025 | 0.189025 | +0.000000 | +0.00% |

At nDCG@10, Jev improved 129 queries, tied on 143, and degraded 51. The
unchanged Recall@30 is expected: reranking can reorder the BM25 candidate pool,
but it cannot recover a relevant document that BM25 did not retrieve.

## Candidate-pool experiment: top 30 versus top 50

A second complete 323-query run changed only the BM25 candidate pool from 30
to 50. Jev continued to score candidates in concurrent batches of 10.

| Metric | Top 30 + Jev | Top 50 + Jev | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| nDCG@1 | 0.489164 | 0.491228 | +0.002064 | +0.42% |
| nDCG@10 | 0.353235 | 0.362468 | +0.009233 | +2.61% |
| MRR@10 | 0.585817 | 0.593023 | +0.007206 | +1.23% |
| MAP@10 | 0.264411 | 0.270442 | +0.006031 | +2.28% |
| Precision@10 | 0.247059 | 0.250464 | +0.003405 | +1.38% |
| Recall@10 | 0.158667 | 0.164474 | +0.005807 | +3.66% |
| Candidate-pool recall | 0.189025 at 30 | 0.209810 at 50 | +0.020785 | +11.00% |

At nDCG@10, the top-50 run improved 46 queries, tied on 224, and degraded
53 relative to the top-30 run. Jev promoted 477 candidates originally ranked
31--50 by BM25 into the final top 10; 77 of those promoted candidates were
relevant, across 47 queries. The larger pool therefore produced a real but
modest average quality gain.

| Candidate pool | Median latency | p95 latency | Provider cost |
| --- | ---: | ---: | ---: |
| Top 30 + Jev | 1,075.16 ms | 16,952.82 ms | $0.163658 |
| Top 50 + Jev | 1,921.29 ms | 17,839.73 ms | $0.261124 |

Relative to top 30, top 50 increased median latency by 78.70% and observed
provider cost by 59.56%. Top 30 remains the default because it offers the
better quality/cost tradeoff; top 50 is useful when retrieval quality matters
more than latency and provider usage.

## Hybrid BM25 + Text Embedding experiment

An additional run combined BM25 top 50 and `openai/text-embedding-3-large`
top 50 using reciprocal rank fusion (`rrf_k=60`), retaining 50 fused candidates.
The embedding-only retrieval portion completed across all 323 queries:

| Metric | BM25 top 50 | BM25 + embedding RRF top 50 | Change |
| --- | ---: | ---: | ---: |
| nDCG@10 | 0.305654 | 0.396712 | +0.091058 |
| Precision@10 | 0.215170 | 0.283901 | +0.068731 |
| Recall@10 | 0.147309 | 0.193977 | +0.046668 |
| Recall@50 | 0.209810 | 0.318075 | +0.108265 |

The 3,633-document embedding index used 3,072 dimensions, took 285.19 seconds
to create, and reported a one-time provider cost of $0.161792. It is cached
locally for subsequent runs.

The complete 323-query Jev pass produced the following final result:

| Metric | BM25 + Jev top 50 | Hybrid retrieval | Hybrid + Jev | Hybrid + Jev vs BM25 + Jev |
| --- | ---: | ---: | ---: | ---: |
| nDCG@1 | 0.491228 | 0.524252 | 0.552116 | +0.060888 |
| nDCG@10 | 0.362468 | 0.396712 | 0.444327 | +0.081859 |
| MRR@10 | 0.593023 | 0.632089 | 0.654583 | +0.061560 |
| MAP@10 | 0.270442 | 0.295243 | 0.337663 | +0.067221 |
| Precision@10 | 0.250464 | 0.283901 | 0.317337 | +0.066873 |
| Recall@10 | 0.164474 | 0.193977 | 0.214907 | +0.050433 |
| Recall@50 | 0.209810 | 0.318075 | 0.318075 | +0.108265 |

Relative to BM25 top 50 + Jev, the hybrid + Jev pipeline improved nDCG@10
by 22.58%. It improved 138 queries, tied on 116, and degraded 69. Within the
hybrid pipeline, Jev improved 147 queries, tied on 99, and degraded 77 relative
to the fused retrieval order.

The Jev stage reported 8,767,271 input tokens, 369,835 output tokens, and
$0.368225 provider cost. Original cold Jev calls had a 1,763 ms median,
17,582 ms p95, and 61,320 ms maximum latency. The ten-query cold end-to-end
pilot, which also included online query embedding, had a 3,172.77 ms median.
The final resumed run was cache-dominated, so its aggregate latency must not be
used as a cold online latency claim.

### Unofficial leaderboard context

On the [MTEB NFCorpus task page](https://mteb-leaderboard.hf.space/tasks/NFCorpus)
observed on 2026-09-26, 250 model results were
listed. Numerically inserting this pipeline's `0.444327` nDCG@10 would place it
at approximately **#4 of 251 results (top 1.6%)**, behind the displayed
`0.5574`, `0.4699`, and `0.4496` scores.

This is not an official MTEB rank. Jev RAG is a multi-stage retrieval and
reranking pipeline rather than a single embedding model, it has not been
submitted through MTEB, and the top-50 choice was observed on the same test
set. The comparison is disclosed only to give the score numerical context.

## Agentic lexical search experiment

The third application mode uses `minimax/minimax-m3` to plan local lexical
searches without creating embeddings. The benchmark disclosed a `medical`
domain hint because NFCorpus is a medical collection; the product default is
domain-neutral. Round one generates five searches from
the user query. Round two inspects up to eight first-round titles/snippets and
generates five follow-up searches. The original query and generated searches
each retrieve up to 100 local BM25 candidates; RRF (`k=60`) retains 50 for Jev.
Official relevance judgments were never included in planner prompts.

| Pipeline | nDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 top 50 | 0.305654 | 0.512697 | 0.218445 | 0.147309 | 0.209810 |
| BM25 top 50 + Jev | 0.362468 | 0.593023 | 0.270442 | 0.164474 | 0.209810 |
| Agentic lexical top 50 | 0.380168 | 0.597940 | 0.275298 | 0.185464 | 0.280765 |
| **Agentic lexical top 50 + Jev** | **0.430969** | **0.644041** | **0.327298** | **0.204138** | **0.280765** |
| Hybrid top 50 + Jev | 0.444327 | 0.654583 | 0.337663 | 0.214907 | 0.318075 |

Agentic + Jev improved nDCG@10 by 18.90% relative to BM25 top 50 +
Jev. It finished 0.013358 (3.01%) below hybrid + Jev. A paired 20,000-sample
query bootstrap estimated the Agentic-minus-Hybrid+Jev difference at
`[-0.025480, -0.001697]` (95% interval). The result is therefore a strong
no-vector alternative, not evidence that iterative lexical search has surpassed
the best measured embedding pipeline.

An oracle diagnostic sorted only the retrieved candidate pools by official
relevance. Agentic top 50 reached an oracle nDCG@10 of `0.604524`, compared
with `0.488842` for BM25 and `0.649287` for hybrid retrieval. This diagnostic
is not a system score; it shows that both candidate recall and Jev ranking still
have headroom.

The complete planner run reported 317,865 prompt tokens, 17,279 completion
tokens, and $0.105439 provider cost. The Jev pass reported 8,817,321 input
tokens, 369,835 output tokens, and $0.370327 provider cost, for $0.475766
combined planner + reranker cost. Estimated serial online latency was 7,516 ms
median and 26,347 ms p95. Plans and reranks were cached while developing and
resuming the run; latency and price remain provider-dependent observations.

Reproduce the mode with:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus \
  --top-k 50 \
  --agentic \
  --agentic-model minimax/minimax-m3 \
  --agentic-rounds 2 \
  --agentic-queries 5 \
  --agentic-per-query-k 100 \
  --agentic-domain-hint medical \
  --rrf-k 60 \
  --use-jev \
  --provider openrouter \
  --output .knowledge/public-benchmarks/results/nfcorpus-agentic-jev.json
```

Use `--limit-queries 10` for a lower-cost connectivity check before a full run.
The compact machine-readable disclosure is stored in
[`nfcorpus-agentic-summary.json`](nfcorpus-agentic-summary.json).

## Multi-round Agentic Hybrid experiment

Version 0.7.0 combines the two-round Agentic lexical branch with dense
retrieval. For each query, MiniMax plans five lexical searches in round one and
five follow-up searches after inspecting up to eight first-round snippets. The
original query and generated searches run against local BM25; in the
application, the original query's embedding request runs concurrently with the
Agentic branch. Weighted RRF (`agentic=0.65`, `vector=1.0`, `k=60`) retains 50
candidates for the unchanged Jev reranker. Generated searches are not embedded.

The weight was selected on the NFCorpus dev split. The test split was then run
once with the fixed configuration.

| Pipeline | nDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bare Hybrid top 50 | 0.396712 | 0.632089 | 0.295243 | 0.193977 | 0.318075 |
| Multi-round Agentic Hybrid top 50 | 0.424145 | 0.637722 | 0.321039 | 0.206303 | **0.333047** |
| Hybrid top 50 + Jev | 0.444327 | **0.654583** | 0.337663 | 0.214907 | 0.318075 |
| Multi-round Agentic Hybrid top 50 + Jev | 0.445761 | 0.645722 | 0.341929 | 0.217207 | **0.333047** |
| **Agentic Hybrid + Jev/retrieval rank fusion** | **0.450750** | **0.652606** | **0.345403** | **0.220885** | **0.333047** |

The fused retrieval order improved nDCG@10 by `0.027433` (6.92% relative)
and Recall@50 by `0.014972` over bare Hybrid. After Jev, the new pipeline's
`0.445761` nDCG@10 was only
`0.001434` (0.32% relative) above Hybrid + Jev. Per-query comparison found 97
improvements, 142 ties, and 84 degradations. A paired 20,000-sample bootstrap
95% interval for the difference was `[-0.006883, 0.009834]`, so the evidence
does not establish a statistically significant win for the Jev-only order.
The separately dev-selected post-rerank prior then reached `0.450750` without
changing the candidate pool or adding a provider call.

The cold retrieval-only run measured 6,990.67 ms median and 24,391.42 ms p95
latency. It reported 437,433 planner prompt tokens, 50,241 completion tokens,
and `$0.16607392` planner cost. The Jev pass reported 8,833,108 input tokens,
369,835 output tokens, and `$0.37099054` cost. The application overlaps the
query-embedding request with the Agentic branch, but the two planner rounds are
sequential by design. Corpus embeddings remain a separately cached, one-time
indexing cost.

Reproduce the fixed test configuration with:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus --split test --top-k 50 \
  --embedding-model openai/text-embedding-3-large \
  --vector-top-k 50 --rrf-k 60 \
  --agentic-hybrid --agentic-rounds 2 --agentic-queries 5 \
  --agentic-per-query-k 100 --agentic-domain-hint medical \
  --agentic-weight 0.65 --vector-weight 1.0 \
  --use-jev --jev-retrieval-prior-weight 0.25 --provider openrouter \
  --output .knowledge/public-benchmarks/results/nfcorpus-agentic-hybrid-jev-prior-test.json
```

The post-rerank retrieval prior was selected on all 324 `dev` queries. Jev
rank weight `1.0` plus original retrieval rank weight `0.25` reached
`0.412509` dev nDCG@10. The fixed test run reached `0.450750` nDCG@10,
`0.652606` MRR@10, `0.345403` MAP@10, and `0.220885` Recall@10. This is a
local rank fusion over already available ranks, so it adds no provider call,
tokens, or cost.

Against the same test run ordered by Jev alone (`0.445761`), the prior-aware
order improved 100 queries, tied on 158, and degraded 65. A paired 20,000
sample bootstrap interval for the `+0.004989` nDCG@10 delta was
`[0.000397, 0.009757]`. This comparison isolates only the final ordering;
candidate recall remains unchanged at `0.333047` at 50.

A separate ten-query `dev` pilot tested all 50 compact 600-character
candidates in one Jev request. On the exact same candidate lists it scored
`0.373795` nDCG@10 versus `0.367176` for five batches of ten, with median Jev
latency of 1,536 ms versus 2,315 ms and observed cost of `$0.00656758` versus
`$0.00709191`. This is a small stability pilot, not a full benchmark result;
the production default therefore remains ten candidates per Jev request.

The compact disclosure is stored in
[`nfcorpus-agentic-hybrid-summary.json`](nfcorpus-agentic-hybrid-summary.json).

## Two-level Jev Line Search experiment

On 2026-09-27, version 0.4.0 evaluated the TypeSafe Semantic Find pattern as a
two-level full-corpus search. The 3,633 documents were partitioned into 15
windows of at most 255 documents. Every window ran Jev Choice + Noul in
parallel, retained four finalists, and a final Choice + Noul ranked the 60
finalists. BM25 and embeddings were not used.

| Pipeline | nDCG@1 | nDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 top 50 + Jev | 0.491228 | 0.362468 | 0.593023 | 0.270442 | 0.164474 | 0.209810 |
| Two-level Line Search | **0.554180** | 0.366280 | **0.657660** | 0.247978 | 0.169397 | 0.234411 |
| Agentic lexical top 50 + Jev | 0.546956 | 0.430969 | 0.644041 | 0.327298 | 0.204138 | 0.280765 |
| Hybrid top 50 + Jev | 0.552116 | **0.444327** | 0.654583 | **0.337663** | **0.214907** | **0.318075** |

Line Search improved nDCG@10 by only 1.05% over BM25 top 50 + Jev: it
improved 120 queries, tied on 85, and degraded 118. Relative to Hybrid + Jev,
it improved 67, tied on 88, and degraded 168, with 17.57% lower average
nDCG@10. The very strong nDCG@1 and MRR together with weaker MAP and recall
show that hierarchical Choice is effective at selecting one likely answer,
but does not produce a broad graded ranking as reliably as the Agentic and
Hybrid candidate pools.

The exact cold run used 5,168 Jev requests, 108,411,611 input tokens,
12,119,606 output tokens, and $4.553288 provider cost. That was 17.44 times
the observed BM25 top 50 + Jev cost and 12.37 times the Hybrid + Jev reranking
cost. A ten-query cold pilot had 11,742.93 ms median and 35,751.55 ms p95
latency. The full run was resumed twice after upstream connection failures, so
its cache-dominated aggregate latency is not a valid cold-latency measurement.

On the MTEB NFCorpus page observed on 2026-09-27, numerically inserting
`0.366280` would place the pipeline at approximately #88 of 251 results (the
displayed scores around that point are rounded). This is not an official MTEB
rank: the result is a multi-request pipeline evaluated locally, not a submitted
embedding model.

The compact disclosure is stored in
[`nfcorpus-line-search-summary.json`](nfcorpus-line-search-summary.json).

## Hybrid + Unified Jev Passage Gate experiment

On 2026-09-27, version 0.5.0 kept the strongest Hybrid retrieval configuration
(BM25 top 50 + `openai/text-embedding-3-large` top 50 + RRF top 50) and replaced
ordinary Jev reranking with one unified Passage Gate. For every candidate, the
same decision request produced four `Noul` probabilities: relevance, usable
answer evidence, contradiction of a query premise, and prompt injection.

| Pipeline | nDCG@1 | nDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bare Hybrid retrieval | 0.524252 | 0.396712 | 0.632089 | 0.295243 | 0.193977 | 0.318075 |
| Hybrid + Unified Passage Gate | 0.530444 | 0.376298 | 0.618043 | 0.276441 | 0.166977 | 0.180129 |
| Hybrid + ordinary Jev reranking | **0.552116** | **0.444327** | **0.654583** | **0.337663** | **0.214907** | **0.318075** |

The gate slightly improved nDCG@1 over bare Hybrid by 0.006192, but reduced
nDCG@10 by 0.020414 and Recall@10 by 0.027000. Against ordinary Jev reranking,
nDCG@10 was 0.068029 lower (15.31% relative). The fixed routing profile kept
2,518 candidates as normal evidence, marked one as conflicting evidence, and
excluded 13,631: 84.40% of all 16,150 candidates. Average retained evidence was
7.80 passages per query. The result suggests these fixed thresholds are too
aggressive for NFCorpus graded retrieval, even though the extra judgments may
still be useful for an application-level security or premise-checking policy.

The exact cold run used 1,615 batched Jev requests, 9,815,406 input tokens,
1,350,140 output tokens, and $0.412247052 provider cost. Median query latency
was 1,555.05 ms and p95 was 17,469.24 ms. Embedding vectors were reused from
the local cache. The compact disclosure is stored in
[`nfcorpus-passage-gate-summary.json`](nfcorpus-passage-gate-summary.json).

Reproduce it with:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus --split test --top-k 50 \
  --embedding-model openai/text-embedding-3-large \
  --vector-top-k 50 --rrf-k 60 \
  --passage-gate --provider openrouter \
  --output .knowledge/public-benchmarks/results/nfcorpus-hybrid-passage-gate.json
```

## Corpus taxonomy routing experiment

On 2026-09-27, version 0.6.0 built a deterministic two-level taxonomy from the
3,633 cached corpus embeddings: 12 top-level branches and 62 leaf nodes, with a
target leaf size of 64. Each document has one primary leaf and can receive a
near-boundary secondary assignment. Queries route to four leaves covering at
least 200 assigned documents. The original Hybrid top 50 remains unchanged;
up to 20 unique node-local BM25/embedding/RRF candidates are appended before
ordinary Jev reranking. Taxonomy construction used neither queries nor qrels.

| Pipeline | nDCG@10 | MRR@10 | MAP@10 | Recall@10 | Recall@50 | Recall@70 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bare Hybrid top 50 | 0.396712 | 0.632089 | 0.295243 | 0.193977 | 0.318075 | - |
| Hybrid top 50 + Jev | **0.444327** | **0.654583** | **0.337663** | **0.214907** | 0.318075 | - |
| Taxonomy-expanded Hybrid top 70 + Jev | 0.441851 | 0.654006 | 0.334660 | 0.214344 | **0.330883** | **0.342964** |

The expanded retrieval pool improved recall from `0.318075` at 50 to
`0.342964` at 70; 121 of 323 queries gained at least one relevant document in
the appended portion. Final nDCG@10 was nevertheless 0.002476 lower than
Hybrid + Jev. Per-query comparison found 58 improvements, 194 ties, and 71
degradations. The evidence supports taxonomy routing as a recall-expansion
mechanism, but not as a new best final-ranking configuration. Better score
calibration or a global third-stage reranker is the next experiment.

The candidate pool contained 50--70 passages (mean 67.76). The complete Jev
run reported 11,919,795 input tokens, 503,311 output tokens, and `$0.50063139`
provider cost. Observed median latency was 1,097.18 ms and p95 was 17,231.70 ms;
cached responses from interrupted/resumed runs mean this is not a clean cold
latency claim. Corpus and query embeddings reused the existing local cache.

Reproduce the experiment with:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus --split test --top-k 50 \
  --embedding-model openai/text-embedding-3-large \
  --vector-top-k 50 --rrf-k 60 \
  --taxonomy --taxonomy-extra-candidates 20 \
  --use-jev --provider openrouter \
  --output .knowledge/public-benchmarks/results/nfcorpus-taxonomy-hybrid-jev-top70.json
```

The compact disclosure is stored in
[`nfcorpus-taxonomy-summary.json`](nfcorpus-taxonomy-summary.json).

## Latency and provider usage

| Mode | Median query latency | p95 query latency |
| --- | ---: | ---: |
| BM25 top 30 | 1.92 ms | 6.60 ms |
| BM25 top 30 + Jev | 1,075.16 ms | 16,952.82 ms |
| Multi-round Agentic Hybrid retrieval (cold) | 6,990.67 ms | 24,391.42 ms |

The Jev quality run reported 3,896,612 input tokens, 164,103 output tokens, and
$0.163657704 total provider cost. Of the 323 queries, 25 produced no BM25
candidate and therefore made no Jev request. Eight query-level Jev results were
already cached by the pilot run; cold remote requests had a 1,141.83 ms median,
17,060.52 ms p95, and 61,691.96 ms maximum latency. Provider latency and pricing
can change, so these timing and cost numbers are observations from this run,
not stable product guarantees.

## Reproduce

BM25 does not require an API key:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus \
  --top-k 30 \
  --output .knowledge/public-benchmarks/results/nfcorpus-bm25-top30.json
```

The complete Jev run may incur provider charges:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus \
  --top-k 30 \
  --use-jev \
  --provider openrouter \
  --output .knowledge/public-benchmarks/results/nfcorpus-jev-openrouter-top30.json
```

The hybrid run requires the optional NumPy dependency and may incur embedding
and Jev charges:

```bash
python -m pip install -e '.[benchmark]'
python scripts/benchmark_beir.py \
  --dataset nfcorpus \
  --top-k 50 \
  --embedding-model openai/text-embedding-3-large \
  --vector-top-k 50 \
  --rrf-k 60 \
  --use-jev \
  --provider openrouter \
  --output .knowledge/public-benchmarks/results/nfcorpus-hybrid.json
```

Start with `--limit-queries 10` when validating a new provider key. Limited
runs are deterministic plumbing checks and must not be compared with the full
323-query result above.

The complete two-level Line Search run uses Jev implicitly and is substantially
more expensive than shortlist reranking:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus \
  --top-k 100 \
  --line-search \
  --line-search-window-size 255 \
  --line-search-beam 4 \
  --provider openrouter \
  --output .knowledge/public-benchmarks/results/nfcorpus-line-search-full.json
```
