#!/usr/bin/env python3
"""Fail when files intended for Git contain common release-time secrets."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 5 * 1024 * 1024

FORBIDDEN_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "token.json",
    "tokens.json",
}
FORBIDDEN_PARTS = {
    ".knowledge",
    ".secrets",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "build",
    "dist",
    "secrets",
}
FORBIDDEN_SUFFIXES = {
    ".db",
    ".db-shm",
    ".db-wal",
    ".jks",
    ".key",
    ".log",
    ".p12",
    ".pem",
    ".pfx",
    ".sqlite",
    ".sqlite3",
}

SECRET_PATTERNS = {
    "OpenRouter API key": re.compile(r"sk-" r"or-v1-[A-Za-z0-9_-]{16,}"),
    "GitHub token": re.compile(r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{16,}"),
    "Slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{16,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    "private key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "hard-coded credential": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*"
        r"[\"'](?!<|example|replace|your-)([^\"']{16,})[\"']"
    ),
    "absolute home path": re.compile(
        re.escape("/" + "Users" + "/")
        + "|"
        + re.escape("/" + "home" + "/")
        + r"|[A-Za-z]:\\Users\\"
    ),
}


def publish_candidates() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def main() -> int:
    findings: list[str] = []
    files = publish_candidates()
    for path in files:
        relative = path.relative_to(ROOT)
        if (
            path.name in FORBIDDEN_NAMES
            or any(part in FORBIDDEN_PARTS for part in relative.parts)
            or any(path.name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
        ):
            findings.append(f"{relative}: forbidden release file")
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            findings.append(f"{relative}: cannot read ({exc})")
            continue
        if len(raw) > MAX_TEXT_BYTES or b"\0" in raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{relative}:{line_number}: {label}")

    if findings:
        print("Release safety check failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print(f"Release safety check passed ({len(files)} publishable files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
