# Jev RAG

**Vector-free local knowledge search with SQLite BM25, Jev reranking, and grounded streaming answers.**

[简体中文](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Security](SECURITY.md) · [Contributing](CONTRIBUTING.md)

```text
local files -> SQLite FTS5 / BM25 -> Jev relevance reranking -> MiniMax answer
```

Jev RAG indexes a local folder without embeddings or a vector database. It retrieves passages with SQLite FTS5/BM25, asks Jev to judge their usefulness as evidence, and optionally sends the selected evidence to an OpenRouter chat model for a cited answer. The built-in web UI streams the answer and reports retrieval, reranking, first-token, generation, and total latency.

> Status: alpha. The software is usable locally, but APIs and storage schemas may change before 1.0.

## Why this project

- No embedding model, vector database, GPU, or external indexing service.
- Local incremental indexing for Markdown, text, HTML, JSON, CSV, YAML, DOCX, and PDF.
- Chinese-aware lexical tokenization using CJK bigrams.
- BM25 candidate retrieval followed by Jev evidence scoring.
- Automatic Jev batching for long candidate lists.
- Optional relevance threshold and explicit no-evidence behavior.
- Streaming grounded answers with numbered file citations.
- SQLite caches and per-run latency, usage, and cost records.
- Standard-library core; `pypdf` is optional for PDF extraction.

## Non-goals

Jev RAG is not a vector-search framework, a hosted multi-user service, or a guarantee of factual correctness. Jev and the answer model are remote services: the query and selected document text leave your machine when those features are enabled.

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

- BM25 candidates: up to 30 passages
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
jev-rag search 'query' --lexical-only

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

## Project maturity and naming

Other public repositories use similar `jev-rag` names. This project is distinguished by its SQLite FTS5/BM25 retrieval, no-vector design, local-folder indexing, and MiniMax streaming answer path. It is an independent community project and is not affiliated with or endorsed by TypeSafe AI, OpenRouter, or MiniMax.

## License

[MIT](LICENSE)
