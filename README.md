# Jev RAG

**Local knowledge search with vector-free BM25 + Jev by default and optional embedding hybrid retrieval.**

[![CI](https://github.com/aifabrice/jev-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/aifabrice/jev-rag/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/aifabrice/jev-rag?include_prereleases)](https://github.com/aifabrice/jev-rag/releases)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776ab)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-17624f)](LICENSE)

[简体中文](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Security](SECURITY.md) · [Contributing](CONTRIBUTING.md)

## Public benchmark

Complete BEIR NFCorpus test split: 3,633 documents and 323 queries, with Jev
reranking the top 30 SQLite BM25 candidates.

| Pipeline | nDCG@10 | MRR@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| BM25 top 30 | 0.305654 | 0.512697 | 0.147309 |
| BM25 top 30 + Jev | 0.353235 | 0.585817 | 0.158667 |
| BM25 top 50 + Jev | 0.362468 | 0.593023 | 0.164474 |
| BM25 top 50 + Embedding top 50 + RRF | 0.396712 | 0.632089 | 0.193977 |
| **Hybrid top 50 + Jev** | **0.444327** | **0.654583** | **0.214907** |

On the [MTEB NFCorpus page](https://mteb-leaderboard.hf.space/tasks/NFCorpus)
observed on 2026-09-26, inserting `0.444327`
numerically would place this run at approximately **#4 of 251 results (top
1.6%)**. This is an **unofficial comparison**, not an MTEB leaderboard rank:
the multi-stage pipeline has not been submitted to MTEB, and its top-50
configuration was evaluated on the same test set.

[Full results, exact configuration, cost, caveats, and reproduction commands](benchmarks/NFCORPUS_RESULTS.md)

![Jev RAG local web interface](docs/assets/demo-ui.png)

```text
default: local files -> SQLite BM25 ----------------------> Jev -> MiniMax
hybrid:  local files -> BM25 + OpenRouter embeddings/RRF -> Jev -> MiniMax
```

Jev RAG indexes a local folder and defaults to vector-free SQLite FTS5/BM25.
The web UI and CLI can switch to an optional hybrid mode that fuses BM25 and
embedding rankings before Jev. No vector database is required: the normalized
embedding matrix is cached locally. The UI streams cited answers and reports
each pipeline stage's latency.

> Status: alpha. The software is usable locally, but APIs and storage schemas may change before 1.0.

## Why this project

- Vector-free BM25 + Jev remains the default; no embedding setup is required.
- Optional BM25 + Embedding reciprocal-rank fusion before the same Jev stage.
- No vector database or GPU; hybrid vectors are cached in `.knowledge/`.
- Local incremental indexing for Markdown, text, HTML, JSON, CSV, YAML, DOCX, and PDF.
- Chinese-aware lexical tokenization using CJK bigrams.
- BM25 candidate retrieval followed by Jev evidence scoring.
- Automatic Jev batching for long candidate lists.
- Optional relevance threshold and explicit no-evidence behavior.
- Streaming grounded answers with numbered file citations.
- SQLite caches and per-run latency, usage, and cost records.
- Standard-library core; `pypdf` is optional for PDF extraction.

## How it differs from vector RAG

| | Default mode | Optional hybrid mode | Typical vector RAG |
|---|---|---|---|
| First-stage retrieval | SQLite FTS5/BM25 | BM25 + embedding, fused with RRF | Embedding similarity |
| Extra infrastructure | None beyond SQLite | OpenRouter embedding API; local NumPy cache | Embedding model and vector store |
| Strongest queries | Exact terms, IDs, names | Exact terms plus semantic paraphrases | Semantic similarity and paraphrases |
| Second stage | Jev evidence reranking | Jev evidence reranking | Optional reranker |
| Main trade-off | Can miss synonyms | Better recall with added cost and latency | Indexing and infrastructure |

This is a deliberate retrieval architecture, not a claim that lexical search always beats embeddings. Measure it on your own documents and questions.

## Non-goals

Jev RAG is not a hosted multi-user service or a guarantee of factual correctness. Jev and the answer model are remote services. Hybrid mode additionally sends document text and queries to the configured OpenRouter embedding model.

## Requirements

- Python 3.9+
- SQLite with FTS5 enabled
- An OpenRouter API key for the default end-to-end path
- Optionally, a TypeSafe API key for direct Jev access
- Optionally, `pypdf` or the system `pdftotext` command for PDFs

## Install

From a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[documents]'
# Include the optional hybrid mode:
python -m pip install -e '.[documents,embeddings]'
cp .env.example .env
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Set at least `OPENROUTER_API_KEY` in `.env` for the default pipeline:

```dotenv
OPENROUTER_API_KEY=
TYPESAFE_API_KEY=
```

Never commit `.env`. If a key is exposed, revoke it immediately; deleting it from the latest commit is not sufficient.

## Quick start

Put documents in `knowledge/`, then run:

```bash
jev-rag index
jev-rag search 'How long does a duplicate-charge refund take?'
jev-rag serve
```

Open <http://127.0.0.1:8765>. The server binds to localhost by default.
Choose either **BM25 + Jev** (default) or **BM25 + Embedding + Jev** in the UI.

Without installing the command, the equivalent source commands are:

```bash
python3 local_kb.py index
python3 local_kb.py search '退款需要多久？'
python3 local_kb.py serve
```

## Index another folder

```bash
jev-rag \
  --documents /absolute/path/to/documents \
  --db .knowledge/documents.db \
  --exclude 'private/**' \
  index --rebuild

jev-rag \
  --documents /absolute/path/to/documents \
  --db .knowledge/documents.db \
  --exclude 'private/**' \
  serve
```

Exclusions are relative glob patterns and may be repeated. Exclude the project directory when indexing one of its parent folders.

## Retrieval and answer defaults

The web application currently uses:

- Retrieval mode: `bm25` by default
- BM25 candidates: up to 30 passages
- Hybrid mode: BM25 top 50 + embedding top 50, RRF top 50 (`rrf_k=60`)
- Embedding model: `openai/text-embedding-3-large` through OpenRouter
- Jev batch size: 10 candidates per request, with batches executed concurrently
- Evidence passed to the answer model: up to 10 passages
- Answer model: `minimax/minimax-m3` through OpenRouter
- Relevance threshold: `0.0` by default, preserving all scored candidates

To require a minimum Jev score and return a local no-evidence response when nothing passes:

```bash
jev-rag serve --threshold 0.20
```

Thresholds are application policy, not proof of relevance. Calibrate them on representative questions and documents.

## Chunking

```bash
# Default: keep short files whole; split long files by headings and paragraphs.
jev-rag index --chunking auto --rebuild

# One searchable record per file.
jev-rag index --chunking none --rebuild

# Always group content into heading-aware passages.
jev-rag index --chunking paragraph --rebuild
```

Very large files should usually be chunked. With `none`, citations identify the file but cannot point precisely to a small passage.

## CLI reference

```bash
# BM25 only; makes no Jev request.
jev-rag search 'query' --no-jev

# Optional hybrid retrieval followed by Jev.
jev-rag search 'query' --retrieval-mode hybrid

# Make hybrid the initial selection in the web UI.
jev-rag serve --retrieval-mode hybrid

# Keep only passages at or above a Jev score.
jev-rag search 'query' --threshold 0.20

# Machine-readable output.
jev-rag search 'query' --json

# Ignore a cached Jev judgment.
jev-rag search 'query' --no-cache

# Inspect the local index.
jev-rag status

# Test provider connectivity without starting the knowledge-base app.
jev-rag-smoke-test --provider openrouter
jev-rag-smoke-test --provider typesafe
jev-rag-smoke-test --dry-run
```

Run `jev-rag --help` and `jev-rag <command> --help` for all options.

## Supported files

`.txt`, `.md`, `.markdown`, `.rst`, `.log`, `.csv`, `.tsv`, `.json`, `.jsonl`, `.yaml`, `.yml`, `.html`, `.htm`, `.docx`, and `.pdf`.

Scanned or image-only PDFs require OCR before indexing. PDF extraction prefers `pypdf`, then falls back to `pdftotext` when available.

## Privacy and security

- Indexes, caches, and answer histories are stored under `.knowledge/` by default.
- BM25 indexing and retrieval stay local.
- Hybrid mode sends passage text once for corpus embeddings and sends each query for query embedding; vectors are cached locally.
- Jev receives the query and candidate passage text.
- OpenRouter receives the query and final evidence passages for answer generation.
- The local HTTP server has no authentication. Do not expose it directly to the public internet.
- Retrieved documents are untrusted input. The answer prompt asks the model to treat them as evidence, but this is not a complete prompt-injection security boundary.

Read [SECURITY.md](SECURITY.md) before using sensitive documents.

## Development

```bash
python -m pip install -e '.[dev,documents]'
python scripts/check_release.py
python -m unittest discover -s tests -v
python -m compileall -q local_kb.py jev_test.py tests
python -m build
python -m twine check dist/*
```

Normal tests do not call paid APIs. Live calls are always explicit.

## Reproducible evaluation

Run the bundled BM25 smoke benchmark without paid API calls:

```bash
python scripts/benchmark.py
```

To compare the same questions after Jev reranking, explicitly opt in to provider calls:

```bash
python scripts/benchmark.py --use-jev --provider openrouter
```

See the [complete NFCorpus result](benchmarks/NFCORPUS_RESULTS.md) and
[Evaluation](docs/EVALUATION.md) for public benchmark reproduction, the JSONL
format, limitations, and instructions for testing a private document collection.

## Community and roadmap

- Use [Discussions](https://github.com/aifabrice/jev-rag/discussions) for questions, use cases, and design ideas.
- Use [Issues](https://github.com/aifabrice/jev-rag/issues) for reproducible bugs and scoped feature requests.
- Good first contributions include OCR adapters, more document loaders, evaluation datasets, provider adapters, and packaging improvements.
- Planned work is tracked in the [issue tracker](https://github.com/aifabrice/jev-rag/issues).

## Project maturity and naming

Other public repositories use similar `jev-rag` names. This project is distinguished by its SQLite FTS5/BM25 retrieval, no-vector design, local-folder indexing, and MiniMax streaming answer path. It is an independent community project and is not affiliated with or endorsed by TypeSafe AI, OpenRouter, or MiniMax.

## License

[MIT](LICENSE)
