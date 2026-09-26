## Summary

Describe the problem and the change.

## Verification

- [ ] `python -m unittest discover -s tests -v`
- [ ] `python -m compileall -q local_kb.py jev_test.py tests`
- [ ] `ruff check .`
- [ ] Relevant manual or live checks (describe below)

## Impact

- Retrieval quality:
- Latency and cost:
- Privacy and security:
- Compatibility:

## Checklist

- [ ] No credentials, private documents, local databases, or generated artifacts are included.
- [ ] User-facing behavior and configuration are documented.
- [ ] New provider calls are bounded, retry-aware, and absent from normal tests.
