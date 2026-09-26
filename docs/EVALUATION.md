# Evaluation

Jev RAG includes a small retrieval benchmark runner so changes can be compared without inventing or hand-picking performance claims.

The bundled cases are synthetic smoke tests for the four public example documents. They verify the evaluation path; they are not a representative quality benchmark.

## Run the bundled BM25 smoke benchmark

```bash
python scripts/benchmark.py
python scripts/benchmark.py --json
```

The runner reports hit rate, mean reciprocal rank (MRR), median latency, p95 latency, and the first relevant rank for each query.

## Compare BM25 with BM25 + Jev

The Jev run sends the query and retrieved passages to the selected provider and may incur API charges:

```bash
python scripts/benchmark.py --use-jev --provider openrouter
```

Use the same cases, documents, chunking mode, `top-k`, and `top-n` for both runs. Jev results are cached by the application unless the underlying query or passage identity changes.

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

## Interpreting results

- **Hit rate** answers whether at least one expected file appeared in the returned evidence.
- **MRR** rewards placing the first expected file nearer the top.
- **Median and p95 latency** show typical and tail retrieval time.
- Retrieval scores do not measure final-answer correctness. Evaluate grounded answer quality separately with a documented rubric or human review.
- Lexical retrieval is strong for exact identifiers and terminology. Synonyms and paraphrases are expected failure modes; query expansion or a semantic retrieval stage may be more appropriate for those collections.
