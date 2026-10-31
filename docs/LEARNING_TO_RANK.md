# Development-set learning-to-rank experiment

Jev RAG now includes a dependency-free, pairwise learning-to-rank experiment
for the cached NFCorpus runs. It is a second-stage ranker over existing
pipeline signals, not a new embedding model and not another Jev request.

## How it works

1. The candidate pool is the current best Agentic Hybrid retrieval pool (50
   documents per query).
2. For each candidate, the script creates transparent features from its
   reciprocal rank, normalized rank, and presence in four cached runs:
   Agentic Hybrid retrieval, the current Jev/retrieval-prior order, Agentic
   Hybrid + Jev, and ordinary Hybrid.
3. A small pairwise logistic ranker is fitted using **dev qrels only**. Positive
   documents are compared with non-relevant documents in the same query.
4. The learned order is evaluated once on the untouched test qrels. The test
   qrels are never used for fitting or feature construction.

This keeps the experiment reproducible and avoids additional provider charges:

```bash
python scripts/benchmark_ltr.py \
  --candidate-pool base \
  --epochs 5 --learning-rate 0.03 --l2 0.01 \
  --max-negatives-per-positive 8 \
  --output .knowledge/experiments/nfcorpus-ltr-base-devselected-test.json
```

## NFCorpus result

The dev-selected ranker was compared with the existing Agentic Hybrid + Jev
retrieval-prior order on the 323-query test split:

| Order | nDCG@10 | Recall@50 |
| --- | ---: | ---: |
| Existing Jev + retrieval-prior baseline | 0.450750 | 0.333047 |
| Dev-trained LTR (same 50-document pool) | 0.449055 | 0.333047 |

The experiment therefore does **not** replace the current baseline. It is a
useful extension point for future work, but this first linear ranker learned no
generalizing gain on NFCorpus. The negative result is important: it shows that
the current gap is not solved merely by learning a new linear weight over the
already available pipeline ranks. A stronger next step would require richer
per-document features (for example raw BM25/embedding scores, Jev confidence,
query-document lexical statistics) and cross-validation rather than training
on one development split alone.
