# Contributing

Thanks for helping improve Jev RAG.

## Before opening a change

- Search existing issues before creating a duplicate.
- Keep changes focused and avoid unrelated formatting rewrites.
- Never include API keys, private documents, `.env`, or `.knowledge/` data.
- For behavior changes, describe the user-visible effect and the retrieval-quality tradeoff.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,documents]'
cp .env.example .env
```

Normal development and tests do not require credentials. Only add keys when explicitly performing a live smoke test.

## Ten-minute first contribution

You can verify the complete local indexing and retrieval path without an API
key or paid provider call:

```bash
python3 local_kb.py --documents ./docs --db .knowledge/contributor.db index
python3 local_kb.py --documents ./docs --db .knowledge/contributor.db search 'retrieval' --no-jev
python -m unittest discover -s tests -v
```

Then pick one focused improvement from the issue tracker. Good starting areas
are document loaders, offline regression tests, provider adapters, packaging,
and clearer benchmark reproduction. For a larger design change, start a
Discussion first so the scope and compatibility expectations are clear.

When reporting a retrieval change, include a small before/after example or a
repeatable benchmark result. Do not make a quality claim from a single query.

## Checks

Run before submitting:

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
python -m compileall -q local_kb.py jev_test.py tests
ruff check .
python -m build
python -m twine check dist/*
```

## Testing expectations

- Add an offline regression test for new parsing, tokenization, ranking, caching, or persistence behavior.
- Mock provider requests. Paid network calls must never run in the normal test suite.
- Use temporary directories and databases in tests.
- Preserve original source paths and citation metadata.

## Pull requests

Include:

1. the problem being solved;
2. the implementation approach;
3. commands used to verify the change;
4. privacy, compatibility, latency, or cost impact;
5. screenshots for visible UI changes.

By contributing, you agree that your contribution is licensed under the MIT License.
