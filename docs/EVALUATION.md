# Evaluation

Jev RAG includes a small retrieval benchmark runner so changes can be compared without inventing or hand-picking performance claims.

The bundled cases are synthetic smoke tests for the four public example documents. They verify the evaluation path; they are not a representative quality benchmark.

## Run the bundled BM25 smoke benchmark

```bash
python scripts/benchmark.py
python scripts/benchmark.py --json
python scripts/benchmark.py --markdown
```

The runner reports hit rate, mean reciprocal rank (MRR), median latency, p95 latency, and the first relevant rank for each query.

## Compare BM25 with BM25 + Jev

The Jev run sends the query and retrieved passages to the selected provider and may incur API charges:

```bash
python scripts/benchmark.py --use-jev --provider openrouter
```

Use the same cases, documents, chunking mode, `top-k`, and `top-n` for both runs. Jev results are cached by the application unless the underlying query or passage identity changes.

To save shareable Markdown reports, run the same benchmark in each mode:

```bash
# Offline baseline: no credentials or provider calls.
python scripts/benchmark.py --markdown > bm25.md
# Optional: requires provider credentials and may incur API charges.
python scripts/benchmark.py --markdown --use-jev --provider openrouter > bm25-jev.md
```

Compare the summary and per-query tables in the two files. `--markdown` changes
only the output format; Jev is enabled only by `--use-jev`. It cannot be combined
with `--json`; omitting both output options keeps the existing terminal output.
Hit rates use one decimal percentage place, MRR uses four decimal places, and
latencies use two decimal places in milliseconds. Formatting is deterministic,
but measured latency can vary between runs; note cache state when comparing it.
A `miss` means no expected path appeared in the returned results and contributes
zero to hit rate and reciprocal rank. Query text is escaped for Markdown tables,
with line breaks rendered as `<br>`. Review reports for private queries before sharing.

## Bring your own evaluation set

Create a JSONL file with one object per line:

```json
{"query":"How long does a duplicate-charge refund take?","expected_paths":["billing/refunds.md"]}
```

Then run:

```bash
python scripts/benchmark.py \
  --documents /absolute/path/to/documents \
  --cases /absolute/path/to/eval.jsonl \
  --db .knowledge/my-eval.db
```

Keep private evaluation sets outside the repository when they contain internal filenames, questions, or document content.

## Run a public BEIR benchmark

The public benchmark adapter downloads the pinned NFCorpus archive into the
Git-ignored `.knowledge/public-benchmarks` directory, verifies its SHA-256,
materializes its corpus as local text files, and evaluates the official test
queries and relevance judgments:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus \
  --top-k 100 \
  --output .knowledge/public-benchmarks/results/nfcorpus-bm25.json
```

This reports nDCG, recall, precision, MRR, and MAP at standard cutoffs. The
corpus is already passage-oriented, so the adapter indexes each BEIR record as
one unit without additional chunking.

Jev calls may incur provider charges. Start with a deterministic pilot before
running the complete 323-query test split:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus \
  --top-k 30 \
  --use-jev \
  --provider openrouter \
  --limit-queries 10 \
  --output .knowledge/public-benchmarks/results/nfcorpus-jev-pilot.json
```

A limited pilot is a plumbing and cost check, not an official full-dataset
result. Do not compare a limited run with a complete benchmark score.

The project's first complete public run is documented in
[`benchmarks/NFCORPUS_RESULTS.md`](../benchmarks/NFCORPUS_RESULTS.md), including
the exact configuration, quality deltas, provider usage, latency caveats, and
reproduction commands.

## Evaluate hybrid BM25 + embedding retrieval

Install the optional benchmark dependency, then use an OpenRouter embedding
model. Corpus and query vectors are cached under the Git-ignored `.knowledge/`
directory. Reciprocal rank fusion (RRF) combines the BM25 and dense rankings
before the optional Jev stage:

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

The first run pays the one-time corpus embedding cost. Later runs with the same
model and pinned dataset reuse the local vector cache. The runner records the
pre-Jev `retrieval_summary` separately from the final reranked `summary`.

The same retrieval path is available in the application with
`jev-rag search 'query' --retrieval-mode hybrid` or from the web UI. The
application keeps `bm25` as its default, so installing the project does not
silently create embeddings or send corpus text to an embedding provider.

## Interpreting results

- **Hit rate** answers whether at least one expected file appeared in the returned evidence.
- **MRR** rewards placing the first expected file nearer the top.
- **Median and p95 latency** show typical and tail retrieval time.
- Retrieval scores do not measure final-answer correctness. Evaluate grounded answer quality separately with a documented rubric or human review.
- Lexical retrieval is strong for exact identifiers and terminology. Synonyms and paraphrases are expected failure modes; query expansion or a semantic retrieval stage may be more appropriate for those collections.
