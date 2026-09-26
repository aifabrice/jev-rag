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

BM25 indexing and retrieval are local. Live Jev scoring sends the query and candidate excerpts to OpenRouter or TypeSafe. Answer generation sends the query and selected evidence to OpenRouter. Review provider retention and privacy terms before indexing confidential material.

## Credentials

- Store credentials in environment variables or a local ignored `.env` file.
- Never pass a key on the command line, where it may enter shell history.
- Never commit `.env`, databases, logs, or request captures containing private content.
- Revoke an exposed credential immediately. Git history rewriting alone does not revoke access.

## Untrusted documents

Documents are untrusted input. HTML scripts and styles are removed during extraction, but document text can still contain prompt injection instructions. The grounding prompt reduces risk but does not provide a security boundary. Do not let generated answers directly execute commands or authorize consequential actions.

## Dependency scope

The core runtime uses the Python standard library. PDF support optionally uses `pypdf`; `pdftotext` may be invoked when installed. Pin and review dependencies in security-sensitive deployments.
