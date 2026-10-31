# Pseudo-relevance feedback experiment

This experiment tests a second retrieval pass without adding another provider
call. It takes the first five Agentic Hybrid retrieval results, extracts
high-signal terms using corpus IDF, runs a second BM25 query, and fuses the
expanded lexical results with the original candidate list.

The configuration is selected on the development split only:

```bash
python scripts/benchmark_prf.py \
  --output .knowledge/experiments/nfcorpus-prf-retrieval-test.json
```

## Result

On the 323-query NFCorpus test split, retrieval-only nDCG@10 improved from
`0.424145` to `0.428961` (`+0.004816`). Recall@50 stayed at `0.333047`, so
the gain came from ordering rather than discovering more relevant documents.

A 10-query Jev pilot was then run on the PRF candidates with batch size 50:

| Full-chain order | nDCG@10 |
| --- | ---: |
| Existing Agentic Hybrid + Jev baseline | 0.476183 |
| PRF candidates + Jev | 0.463357 |

The pilot was negative (`-0.012826`). PRF changes the candidate order in a way
that helps the lexical metric but makes Jev's final judgment less effective.
Therefore this remains an experiment and is not enabled in the default UI.
The next safer variant is to use PRF only as a small candidate expansion while
preserving the original retrieval order and Jev prior, rather than replacing
the candidate order.
