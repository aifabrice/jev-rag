# Security Policy

## Supported versions

Until 1.0, security fixes are applied to the latest release only.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose credentials or private documents. Use GitHub's private vulnerability reporting after the repository is published. Include affected versions, reproduction steps, impact, and any suggested mitigation.

Do not include real API keys or sensitive source documents in a report.

## Deployment boundary

The built-in web server is intended for local use. It has no authentication, user isolation, TLS termination, request-rate limiting, or production hardening.

- Keep the default `127.0.0.1` bind address.
- Do not expose the server directly to the internet.
- If remote access is necessary, place it behind an authenticated TLS reverse proxy and apply network access controls.
- Treat the SQLite database as sensitive: it may contain queries, answers, source excerpts, usage, and cost records.

## Remote data processing

BM25 indexing and retrieval are local. Live Jev scoring sends the query and candidate excerpts to OpenRouter or TypeSafe. Answer generation sends the query and selected evidence to OpenRouter. Optional hybrid mode additionally sends passage text and each query to the configured OpenRouter embedding model; the resulting vector matrix is stored locally under `.knowledge/`. Optional Agentic mode sends the query and up to eight first-round snippets to the configured OpenRouter planner; generated plans are cached in SQLite. Review provider retention and privacy terms before indexing confidential material.

The optional `line-search` mode has a broader remote-data boundary than BM25
reranking: it sends a bounded representation of every indexed passage to Jev
in windows of at most 255 so that each passage participates in the first
hierarchy level. Do not use this mode on a broad personal folder or confidential
corpus unless that transfer is acceptable. Use `--documents` and `--exclude` to
narrow the corpus first.

The default `serve` command discovers `~/Documents` and builds only a local
BM25 index. Before asking a question, confirm the displayed folder is appropriate:
Jev and answer generation still send the selected candidate excerpts to the
configured remote providers. Override discovery with `--documents` or
`JEV_RAG_DOCUMENTS` when the default folder is too broad.

## Credentials

- Store credentials in environment variables or a local ignored `.env` file.
- Never pass a key on the command line, where it may enter shell history.
- Never commit `.env`, databases, logs, or request captures containing private content.
- Revoke an exposed credential immediately. Git history rewriting alone does not revoke access.

## Untrusted documents

Documents are untrusted input. HTML scripts and styles are removed during extraction, but document text can still contain prompt injection instructions. The grounding and Agentic planner prompts explicitly treat excerpts as untrusted data, but prompts do not provide a security boundary. Do not let generated answers or search plans directly execute commands or authorize consequential actions.

## Dependency scope

The default BM25 runtime uses the Python standard library. PDF support optionally uses `pypdf`; hybrid retrieval optionally uses NumPy; `pdftotext` may be invoked when installed. Pin and review dependencies in security-sensitive deployments.
