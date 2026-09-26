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

Start with `--limit-queries 10` when validating a new provider key. Limited
runs are deterministic plumbing checks and must not be compared with the full
323-query result above.
