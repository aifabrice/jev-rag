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

## Evaluate corpus taxonomy routing

Taxonomy mode builds a two-level tree from corpus embeddings only, preserves
the global Hybrid top 50, and appends up to 20 unique candidates from routed
leaf nodes before ordinary Jev reranking:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus --split test --top-k 50 \
  --embedding-model openai/text-embedding-3-large \
  --vector-top-k 50 --rrf-k 60 \
  --taxonomy --taxonomy-extra-candidates 20 \
  --use-jev --provider openrouter \
  --output .knowledge/public-benchmarks/results/nfcorpus-taxonomy-hybrid-jev-top70.json
```

The tree is cached locally and records that neither benchmark queries nor qrels
were used during construction. The full run reached 0.342964 candidate recall
at 70 versus 0.318075 at the preserved Hybrid top 50, but final nDCG@10 was
0.441851 versus 0.444327 for Hybrid + Jev. Treat the mode as an inspectable
candidate-expansion experiment, not a benchmark improvement claim.

## Evaluate Hybrid + Unified Passage Gate

The Passage Gate uses the same Hybrid top-50 candidate pool, but replaces
ordinary Jev reranking with one four-question stage for relevance, usable
answer evidence, contradiction, and prompt injection:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus \
  --top-k 50 \
  --embedding-model openai/text-embedding-3-large \
  --vector-top-k 50 \
  --rrf-k 60 \
  --passage-gate \
  --provider openrouter \
  --output .knowledge/public-benchmarks/results/nfcorpus-hybrid-passage-gate.json
```

Do not add `--use-jev`: the gate replaces the ordinary reranker. The runner
reports route counts and exact cold-cache stage usage. On the complete 323-query
NFCorpus test split, the fixed threshold profile scored 0.376298 nDCG@10,
below bare Hybrid (0.396712) and Hybrid + Jev (0.444327), while excluding 84.4%
of candidates. Treat it as a safety/routing experiment rather than a quality
improvement claim.

## Evaluate Agentic Search + Jev

Agentic mode asks MiniMax for five lexical queries, runs them against the local
FTS5 index, performs a second planning round over up to eight snippets, and
fuses all lexical rankings before Jev. It does not create embeddings, but both
the planner and Jev make paid provider calls. Start with a limited run:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus \
  --top-k 50 \
  --agentic \
  --agentic-rounds 2 \
  --agentic-queries 5 \
  --agentic-per-query-k 100 \
  --agentic-domain-hint medical \
  --use-jev \
  --provider openrouter \
  --limit-queries 10 \
  --output .knowledge/public-benchmarks/results/nfcorpus-agentic-pilot.json
```

Remove `--limit-queries 10` for the complete 323-query run. Query plans and Jev
judgments are cached locally. Official relevance judgments are used only after
ranking to calculate metrics; they are never included in planner prompts. The
`medical` domain hint is disclosed because NFCorpus is a medical collection;
the application default is domain-neutral.

## Evaluate multi-round Agentic Hybrid + Jev

This mode combines the complete two-round Agentic lexical ranking with the
original query's dense ranking, then applies the ordinary Jev reranker. Select
fusion weights on a development split; do not tune them on the final test set.
The published run selected `0.65:1.0` on `dev` and evaluated `test` once:

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

The dev-selected post-rerank fusion gives Jev rank weight `1.0` and original
retrieval rank weight `0.25`. It reached `0.412509` nDCG@10 on `dev`, then
`0.450750` on the fixed `test` run versus `0.445761` for the Jev-only order.
It adds no provider call. The paired 20,000-sample bootstrap interval for the
test delta was `[0.000397, 0.009757]`. The underlying two-round retrieval is
still expensive: its cold retrieval-only median was `6.99 s`.

For robustness experiments, ordinary Jev reranking also accepts
`--jev-batch-size` and `--jev-candidate-max-chars`. A ten-query pilot found
that one compact batch of 50 was promising, but it is not a full benchmark and
the default remains batches of ten.

## Evaluate two-level Line Search

Line Search evaluates the complete corpus with Jev rather than retrieving a
BM25 or embedding shortlist. It is much more expensive, so run a deterministic
pilot first:

```bash
python scripts/benchmark_beir.py \
  --dataset nfcorpus \
  --top-k 100 \
  --line-search \
  --line-search-window-size 255 \
  --line-search-beam 4 \
  --provider openrouter \
  --limit-queries 10 \
  --output .knowledge/public-benchmarks/results/nfcorpus-line-search-pilot.json
```

Remove `--limit-queries 10` for the full run. Stage responses are cached, and
the runner records exact cold-cache request usage even when a run is resumed.
Do not add `--use-jev`: Line Search already uses Jev Choice + Noul as retrieval.

## Interpreting results

- **Hit rate** answers whether at least one expected file appeared in the returned evidence.
- **MRR** rewards placing the first expected file nearer the top.
- **Median and p95 latency** show typical and tail retrieval time.
- Retrieval scores do not measure final-answer correctness. Evaluate grounded answer quality separately with a documented rubric or human review.
- Lexical retrieval is strong for exact identifiers and terminology. Synonyms and paraphrases are expected failure modes; query expansion or a semantic retrieval stage may be more appropriate for those collections.
