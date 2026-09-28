# Jev RAG

**Jev RAG is an open-source local knowledge search engine with seven selectable pipelines: BM25 + Jev by default, agentic lexical search, embedding hybrid retrieval, multi-round Agentic Hybrid, taxonomy-routed hybrid retrieval, a unified Jev Passage Gate, and hierarchical Jev Line Search.**

Use Jev RAG to search a local document folder, rerank candidate passages with
Jev, and stream grounded answers with file citations. The default path is a
vector-free RAG architecture: it requires neither embeddings nor a vector
database. Jev RAG is local-first rather than fully offline because selected
passages are sent to the configured Jev and answer-model providers.

[![CI](https://github.com/aifabrice/jev-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/aifabrice/jev-rag/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/aifabrice/jev-rag?include_prereleases)](https://github.com/aifabrice/jev-rag/releases)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776ab)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-17624f)](LICENSE)

[简体中文](README.zh-CN.md) · [Seven-pipeline field report](docs/JEV_RAG_SEVEN_PIPELINES.md) · [Architecture](docs/ARCHITECTURE.md) · [AI search discoverability](docs/AI_DISCOVERABILITY.md) · [Security](SECURITY.md) · [Contributing](CONTRIBUTING.md)

**[Project website and interactive benchmark →](https://aifabrice.github.io/jev-rag/)**
· [Evidence-backed FAQ](https://aifabrice.github.io/jev-rag/faq.html)
· [Machine-readable project facts](https://aifabrice.github.io/jev-rag/llms.txt)

## Public benchmark

Complete BEIR NFCorpus test split: 3,633 documents and 323 queries. All rows
use the same corpus, queries, qrels, and metric implementation; candidate-pool
sizes and remote stages are shown explicitly.

| Pipeline | nDCG@10 | MRR@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| BM25 top 30 | 0.305654 | 0.512697 | 0.147309 |
| BM25 top 30 + Jev | 0.353235 | 0.585817 | 0.158667 |
| BM25 top 50 + Jev | 0.362468 | 0.593023 | 0.164474 |
| Two-level Jev Line Search (60 finalists) | 0.366280 | **0.657660** | 0.169397 |
| Hybrid top 50 + Unified Passage Gate | 0.376298 | 0.618043 | 0.166977 |
| Agentic lexical top 50 | 0.380168 | 0.597940 | 0.185464 |
| BM25 top 50 + Embedding top 50 + RRF | 0.396712 | 0.632089 | 0.193977 |
| Multi-round Agentic Hybrid top 50 | 0.424145 | 0.637722 | 0.206303 |
| **Agentic lexical top 50 + Jev** | **0.430969** | **0.644041** | **0.204138** |
| Hybrid top 50 + Jev | 0.444327 | **0.654583** | 0.214907 |
| **Agentic Hybrid + Jev/retrieval rank fusion** | **0.450750** | **0.652606** | **0.220885** |

Line Search achieved the strongest first-hit behavior (`nDCG@1=0.554180`),
but lower multi-document ranking quality and recall than Agentic or Hybrid.
Its full-corpus cold provider cost was $4.553288, so it is an experimental
deep-search path rather than the default.

The unified Passage Gate is a deliberately disclosed negative result: with the
cookbook-inspired fixed thresholds it excluded 13,631 of 16,150 candidates
(84.4%) and scored below both bare Hybrid (`0.396712`) and Hybrid + Jev
(`0.444327`) at nDCG@10. It remains available for experiments that value
prompt-injection screening and query-premise checks, but it is not the default
quality path.

Taxonomy and the pre-fusion Agentic Hybrid order remain documented in the full
report, but are omitted from this headline table because they sit in the same
`0.44-0.45` band without a useful new operating point. The table keeps ordinary
Hybrid as the representative reference and the best measured pipeline.

The best pipeline uses a dev-selected local RRF after Jev (Jev rank weight
`1.0`, retrieval rank weight `0.25`). It adds no provider call and reached
`0.450750` test nDCG@10. Its cold retrieval-only median was `6.99 s` and p95
was `24.39 s` because two planning rounds run per query.

On the [MTEB NFCorpus page](https://mteb-leaderboard.hf.space/tasks/NFCorpus)
observed on 2026-09-26, inserting `0.450750`
numerically would place this run at approximately **#4 of 251 results (top
1.6%)**. This is an **unofficial comparison**, not an MTEB leaderboard rank:
the multi-stage pipeline has not been submitted to MTEB, and its top-50
configuration was evaluated on the same test set.

[Full results, exact configuration, cost, caveats, and reproduction commands](benchmarks/NFCORPUS_RESULTS.md)
· [Machine-readable Agentic summary](benchmarks/nfcorpus-agentic-summary.json)
· [Machine-readable Line Search summary](benchmarks/nfcorpus-line-search-summary.json)
· [Machine-readable Passage Gate summary](benchmarks/nfcorpus-passage-gate-summary.json)
· [Machine-readable Taxonomy summary](benchmarks/nfcorpus-taxonomy-summary.json)
· [Machine-readable Agentic Hybrid summary](benchmarks/nfcorpus-agentic-hybrid-summary.json)

![Jev RAG local web interface](docs/assets/demo-ui.png)

```text
default: local files -> SQLite BM25 ----------------------> Jev -> MiniMax
agentic: local files -> MiniMax plans -> multi-BM25/RRF --> Jev -> MiniMax
hybrid:  local files -> BM25 + OpenRouter embeddings/RRF -> Jev -> MiniMax
agentic-hybrid: two-round plans -> multi-BM25 + embedding/RRF -> Jev + retrieval prior -> MiniMax
taxonomy: local files -> corpus taxonomy -> Hybrid top 50 + routed extras -> Jev -> MiniMax
gate:    local files -> BM25 + embeddings/RRF -> unified Jev Gate -> MiniMax
line:    local files -> parallel Jev Choice windows -> global Choice -> MiniMax
```

Jev RAG indexes a local folder and defaults to vector-free SQLite FTS5/BM25.
The web UI and CLI expose seven modes. Agentic mode runs two rounds of
model-planned local lexical searches and fuses them before Jev, without an
embedding index. Hybrid mode fuses BM25 and embedding rankings before Jev. No
vector database is required: the optional normalized embedding matrix is cached
locally. Passage Gate replaces ordinary reranking with four simultaneous Jev
judgments per candidate and routes evidence into include, conflicting, or
exclude groups. Line Search partitions the full index into windows of at most 255,
searches every window with Jev Choice + Noul, and globally re-ranks the window
finalists with a second Choice request. The UI streams cited answers and reports
each pipeline stage's latency.

Agentic Hybrid keeps those two planning rounds, runs their multi-query BM25
branch alongside an original-query embedding lookup, then fuses the Agentic and
dense rankings with weighted RRF (`0.65:1.0`) before the same Jev reranker.
The Agentic Hybrid path then blends the Jev rank with the original retrieval
rank at `1.0:0.25`; this post-rerank step is local and makes no provider call.

Taxonomy mode deterministically clusters the cached corpus embeddings into two
levels, routes each query to four leaf nodes, and appends up to 20 unique routed
candidates to the unchanged Hybrid top 50 before the ordinary Jev stage. The
tree is built from corpus documents only; benchmark queries and qrels are not
used.

> Status: alpha. The software is usable locally, but APIs and storage schemas may change before 1.0.

## Why this project

- Vector-free BM25 + Jev remains the default; no embedding setup is required.
- Optional two-round Agentic Search + Jev improves lexical recall without building embeddings.
- Optional BM25 + Embedding reciprocal-rank fusion before the same Jev stage.
- Optional two-round Agentic BM25 + original-query embedding fusion before Jev.
  A dev-selected local Jev/retrieval rank fusion improves the measured final
  order without another provider call, but planning still adds substantial
  latency.
- Optional two-level corpus taxonomy for query routing and Hybrid candidate
  expansion. The public benchmark improved pool recall but not nDCG@10.
- Optional Hybrid + Unified Jev Passage Gate for relevance, answer-evidence,
  contradiction, and prompt-injection routing. The fixed profile is
  experimental and did not improve NFCorpus nDCG@10.
- Optional two-level Jev Line Search with a structural capacity of 65,025
  indexed passages, without BM25 or embeddings. Practical cost and latency grow
  with corpus size.
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

| | Default | Agentic | Hybrid | Agentic Hybrid | Taxonomy | Hybrid Gate | Line Search |
|---|---|---|---|---|---|---|---|
| First stage | SQLite FTS5/BM25 | Planned multi-BM25 + RRF | BM25 + embedding RRF | Planned multi-BM25 + embedding RRF | Hybrid top 50 + taxonomy extras | BM25 + embedding RRF | Parallel Jev Choice windows |
| Second stage | Jev reranking | Jev reranking | Jev reranking | Jev reranking | Jev reranking | Unified four-question Jev gate | Global Jev Choice |
| Vector index | No | No | Local cached matrix | Local cached matrix | Local cached matrix + tree | Local cached matrix | No |
| Main trade-off | Can miss synonyms | Planner latency/cost | Embedding boundary | Highest measured point estimate; high planner latency | More recall, candidates, and Jev cost | Aggressive filtering; four judgments per passage | Full corpus remotely; highest cost |

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

By default, Jev RAG discovers and indexes the current user's `~/Documents`
folder. Start the local search application with:

```bash
jev-rag serve
```

Open <http://127.0.0.1:8765>. The server binds to localhost by default.
Choose **BM25 + Jev** (default), **Agentic Search + Jev**,
**BM25 + Embedding + Jev**, **Multi-round Agentic + BM25 + Embedding + Jev**,
or one of the experimental routing/search modes in the UI.
Startup incrementally updates the local BM25 index. The default mode does not
create embeddings and does not require a vector database.

Without installing the command, the equivalent source commands are:

```bash
python3 local_kb.py index
python3 local_kb.py serve
```

You can also search directly from the CLI:

```bash
jev-rag search 'Which AI products are mentioned in my documents?'
```

To persist a different default folder, set this in `.env`:

```dotenv
JEV_RAG_DOCUMENTS=/absolute/path/to/documents
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

- Document root: auto-discovered `~/Documents`, falling back to `knowledge/`
- Incremental local indexing whenever `serve` starts
- Retrieval mode: `bm25` by default
- BM25 candidates: up to 30 passages
- Hybrid mode: BM25 top 50 + embedding top 50, RRF top 50 (`rrf_k=60`)
- Agentic Hybrid mode: two planning rounds with five lexical queries per
  round; multi-query BM25 runs alongside the original-query embedding lookup;
  weighted RRF (`0.65:1.0`) keeps 50 candidates for Jev, then a zero-call
  Jev/retrieval rank fusion (`1.0:0.25`) produces the final order
- Taxonomy mode: preserve Hybrid top 50, route over a cached two-level corpus
  tree, then append up to 20 unique node-local candidates before Jev
- Hybrid Gate mode: the same fused top 50, then one Jev stage asks four `Noul`
  questions per passage and applies fixed relevance/evidence/conflict/injection thresholds
- Agentic mode: two planning rounds, five lexical queries per round, BM25 top 100 per query, RRF top 50
- Line Search mode: up to 255 passages per window, four finalists per window,
  at most 255 windows, and a final global Choice; no additional Jev reranker
- The Line Search path extends TypeSafe's
  [Semantic Find cookbook](https://docs.typesafe.ai/cookbooks/semantic_find)
  with parallel window fan-out and a global reduce stage.
- Agentic planner: `minimax/minimax-m3` through OpenRouter; plans are cached locally
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

# Experimental corpus-taxonomy routing plus Hybrid candidate expansion.
jev-rag search 'query' --retrieval-mode taxonomy

# Hybrid retrieval followed by one unified Jev Passage Gate (experimental).
jev-rag search 'query' --retrieval-mode hybrid-gate

# Agent-planned local lexical searches followed by Jev; no vector index.
jev-rag search 'query' --retrieval-mode agentic

# Two-round Agentic BM25 plus parallel original-query embedding, then Jev.
jev-rag search 'query' --retrieval-mode agentic-hybrid

# Search every indexed passage with parallel Jev windows, then globally rank finalists.
jev-rag search 'query' --retrieval-mode line-search

# Tune the two hierarchy levels (maximum window size is 255).
jev-rag search 'query' --retrieval-mode line-search \
  --line-search-window-size 255 --line-search-beam 4

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
- Default discovery indexes supported text documents; it does not upload the folder itself.
- Hybrid mode sends passage text once for corpus embeddings and sends each query for query embedding; vectors are cached locally.
- Hybrid Gate additionally sends the fused top 50 excerpts to Jev for four
  judgments per passage. Injection filtering is probabilistic, not a complete security boundary.
- Agentic mode sends the query and up to eight first-round snippets to the OpenRouter planner; generated search plans are cached locally.
- Line Search sends a bounded representation of every indexed passage to the
  configured Jev provider in windows, then sends the window finalists once more
  for global ranking. Narrow `--documents` and `--exclude` before using it on
  private files.
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

Other public repositories use similar `jev-rag` names. This project is distinguished by its vector-free BM25 default, optional embedding hybrid retrieval, Jev reranking, local-folder indexing, and MiniMax streaming answer path. It is an independent community project and is not affiliated with or endorsed by TypeSafe AI, OpenRouter, or MiniMax.

## License

[MIT](LICENSE)
