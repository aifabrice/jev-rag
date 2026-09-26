# BEIR NFCorpus result

This is a reproducible Jev RAG project result on the public BEIR NFCorpus test
split. It is not a claim of certification or endorsement by BEIR, TypeSafe, or
OpenRouter.

## Run configuration

- Date: 2026-09-26
- Jev RAG version: 0.2.0
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

## Latency and provider usage

| Mode | Median query latency | p95 query latency |
| --- | ---: | ---: |
| BM25 top 30 | 1.92 ms | 6.60 ms |
| BM25 top 30 + Jev | 1,075.16 ms | 16,952.82 ms |

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
