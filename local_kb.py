#!/usr/bin/env python3
"""Jev RAG: local lexical retrieval, Jev reranking, and grounded answers.

The application deliberately has no embedding model and no vector database.
It can run as a CLI or as a small local web application.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import time
import unicodedata
import urllib.parse
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from jev_test import load_dotenv, request_decision


APP_VERSION = "0.2.0"
DEFAULT_GENERATOR_MODEL = "minimax/minimax-m3"
DEFAULT_MAX_TOKENS = 1200
ANSWER_SOURCE_MAX_CHARS = 3600
DEFAULT_DOCUMENTS = Path("knowledge")
DEFAULT_DB = Path(".knowledge/knowledge.db")
JEV_BATCH_SIZE = 10
GENERATOR_RETRIES = 2
SUPPORTED_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".docx",
    ".pdf",
}

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_TOKEN_RE = re.compile(
    r"[a-zA-Z0-9][a-zA-Z0-9_./+#:-]*|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
)
_SENTENCE_BREAK_RE = re.compile(r"(?<=[。！？!?;；.])\s*")


@dataclass
class Passage:
    body: str
    heading: str
    start_line: int
    end_line: int


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = unicodedata.normalize("NFKC", text)
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    return re.sub(r"\n{4,}", "\n\n\n", "\n".join(lines)).strip()


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{ns}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{ns}t"))
        if text.strip():
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    except ImportError:
        pass

    command = shutil.which("pdftotext")
    if command:
        result = subprocess.run(
            [command, "-layout", str(path), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout.decode("utf-8", errors="replace")
    raise RuntimeError("读取 PDF 需要安装 pypdf（pip install pypdf）或系统 pdftotext")


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return clean_text(extract_docx(path))
    if suffix == ".pdf":
        return clean_text(extract_pdf(path))
    raw = read_text_file(path)
    if suffix in {".html", ".htm"}:
        parser = _TextHTMLParser()
        parser.feed(raw)
        raw = html.unescape(parser.text())
    elif suffix == ".json":
        try:
            raw = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    return clean_text(raw)


def lexical_tokens(text: str) -> list[str]:
    """Produce explicit tokens for SQLite FTS5, including CJK bigrams.

    macOS SQLite's built-in unicode61 tokenizer treats a long Chinese phrase as
    one token, while trigram cannot match a two-character word. Pre-tokenizing
    CJK text into bigrams gives useful BM25 behaviour for both cases.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(unicodedata.normalize("NFKC", text).lower()):
        value = match.group(0)
        if _CJK_RE.fullmatch(value):
            if len(value) == 1:
                tokens.append(value)
            else:
                tokens.extend(value[i : i + 2] for i in range(len(value) - 1))
                if len(value) <= 8:
                    tokens.append(value)
        else:
            tokens.append(value.strip("./:+-"))
    return [token for token in tokens if token]


def fts_query(text: str) -> str:
    unique = list(dict.fromkeys(lexical_tokens(text)))
    if not unique:
        raise ValueError("查询中没有可检索的文字或数字")
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique[:80])


def _split_oversized(text: str, start_line: int, max_chars: int) -> list[tuple[str, int, int]]:
    if len(text) <= max_chars:
        return [(text, start_line, start_line + text.count("\n"))]
    sentences = [part.strip() for part in _SENTENCE_BREAK_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        sentences = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    output: list[tuple[str, int, int]] = []
    buffer = ""
    line = start_line
    buffer_start = line
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip()
        if buffer and len(candidate) > max_chars:
            output.append((buffer, buffer_start, line + buffer.count("\n")))
            line += buffer.count("\n")
            buffer = sentence
            buffer_start = line
        else:
            buffer = candidate
    if buffer:
        output.append((buffer, buffer_start, buffer_start + buffer.count("\n")))
    return output


def split_passages(
    text: str,
    title: str,
    mode: str = "auto",
    target_chars: int = 1800,
    max_chars: int = 3200,
) -> list[Passage]:
    text = clean_text(text)
    if not text:
        return []
    total_lines = text.count("\n") + 1
    if mode == "none" or (mode == "auto" and len(text) <= 5000):
        return [Passage(text, title, 1, total_lines)]

    # Split at blank lines and Markdown-style headings, then aggregate related
    # paragraphs. This preserves semantic sections without token-sized chopping.
    lines = text.splitlines()
    blocks: list[tuple[str, str, int, int]] = []
    current_heading = title
    buffer: list[str] = []
    block_start = 1

    def flush(end_line: int) -> None:
        nonlocal buffer, block_start
        value = "\n".join(buffer).strip()
        if value:
            blocks.append((value, current_heading, block_start, end_line))
        buffer = []

    for line_no, line in enumerate(lines, start=1):
        heading_match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if heading_match:
            flush(line_no - 1)
            current_heading = heading_match.group(1).strip()
            block_start = line_no + 1
        elif not line.strip():
            flush(line_no - 1)
            block_start = line_no + 1
        else:
            if not buffer:
                block_start = line_no
            buffer.append(line)
    flush(len(lines))

    expanded: list[tuple[str, str, int, int]] = []
    for body, heading, start, end in blocks:
        for part, part_start, part_end in _split_oversized(body, start, max_chars):
            expanded.append((part, heading, part_start, max(part_end, end if len(body) <= max_chars else part_end)))

    passages: list[Passage] = []
    acc: list[str] = []
    acc_heading = title
    acc_start = 1
    acc_end = 1
    for body, heading, start, end in expanded:
        candidate_len = sum(len(item) for item in acc) + len(body) + max(0, len(acc) - 1) * 2
        if acc and (candidate_len > target_chars or heading != acc_heading):
            passages.append(Passage("\n\n".join(acc), acc_heading, acc_start, acc_end))
            acc = []
        if not acc:
            acc_heading, acc_start = heading, start
        acc.append(body)
        acc_end = end
    if acc:
        passages.append(Passage("\n\n".join(acc), acc_heading, acc_start, acc_end))
    return passages


def iter_document_paths(root: Path, exclude_patterns: Iterable[str] = ()) -> Iterable[Path]:
    if not root.exists():
        return []
    patterns = tuple(pattern.strip().strip("/") for pattern in exclude_patterns if pattern.strip())

    def included(path: Path) -> bool:
        relative = path.relative_to(root)
        relative_text = relative.as_posix()
        return not any(
            relative.match(pattern)
            or relative_text == pattern.rstrip("/**")
            or relative_text.startswith(pattern.rstrip("/**") + "/")
            for pattern in patterns
        )

    return (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and not path.name.startswith("~$")
        and included(path)
    )


class KnowledgeBase:
    def __init__(
        self,
        db_path: Path,
        documents_root: Path,
        exclude_patterns: Iterable[str] = (),
    ) -> None:
        self.db_path = db_path.resolve()
        self.documents_root = documents_root.resolve()
        self.exclude_patterns = tuple(exclude_patterns)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                passage_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(
                doc_id UNINDEXED,
                passage_no UNINDEXED,
                path UNINDEXED,
                title UNINDEXED,
                heading UNINDEXED,
                start_line UNINDEXED,
                end_line UNINDEXED,
                body UNINDEXED,
                title_terms,
                body_terms,
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE IF NOT EXISTS rerank_cache (
                cache_key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS answer_runs (
                id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                generator_model TEXT NOT NULL,
                use_jev INTEGER NOT NULL,
                candidate_count INTEGER NOT NULL,
                returned_count INTEGER NOT NULL,
                lexical_ms REAL,
                jev_ms REAL,
                first_token_ms REAL,
                generation_ms REAL,
                total_ms REAL,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                cost REAL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def _setting(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_setting(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def rebuild(self) -> None:
        self.connection.execute("DELETE FROM passages")
        self.connection.execute("DELETE FROM documents")
        self.connection.execute("DELETE FROM rerank_cache")
        self.connection.commit()

    def index(self, chunking: str = "auto", rebuild: bool = False) -> dict[str, Any]:
        if chunking not in {"auto", "none", "paragraph"}:
            raise ValueError(f"未知分段策略: {chunking}")
        previous_mode = self._setting("chunking")
        if rebuild or (previous_mode and previous_mode != chunking):
            self.rebuild()

        self.documents_root.mkdir(parents=True, exist_ok=True)
        current_paths: set[str] = set()
        stats: dict[str, Any] = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "skipped": []}

        for path in iter_document_paths(self.documents_root, self.exclude_patterns):
            relative = path.relative_to(self.documents_root).as_posix()
            current_paths.add(relative)
            file_stat = path.stat()
            existing = self.connection.execute(
                "SELECT * FROM documents WHERE path = ?", (relative,)
            ).fetchone()
            if existing and existing["mtime_ns"] == file_stat.st_mtime_ns and existing["size"] == file_stat.st_size:
                stats["unchanged"] += 1
                continue
            try:
                text = extract_text(path)
                if not text.strip():
                    raise RuntimeError("没有提取到文字")
            except Exception as exc:  # keep indexing other local files
                stats["skipped"].append({"path": relative, "reason": str(exc)})
                continue

            title = path.stem.replace("_", " ").replace("-", " ").strip() or path.name
            passages = split_passages(text, title, mode=chunking)
            content_hash = sha256_text(text)
            if existing:
                doc_id = int(existing["id"])
                self.connection.execute("DELETE FROM passages WHERE doc_id = ?", (str(doc_id),))
                self.connection.execute(
                    "UPDATE documents SET title=?, mtime_ns=?, size=?, content_hash=?, passage_count=?, indexed_at=? WHERE id=?",
                    (title, file_stat.st_mtime_ns, file_stat.st_size, content_hash, len(passages), utc_now(), doc_id),
                )
                stats["updated"] += 1
            else:
                cursor = self.connection.execute(
                    "INSERT INTO documents(path,title,mtime_ns,size,content_hash,passage_count,indexed_at) VALUES(?,?,?,?,?,?,?)",
                    (relative, title, file_stat.st_mtime_ns, file_stat.st_size, content_hash, len(passages), utc_now()),
                )
                doc_id = int(cursor.lastrowid)
                stats["added"] += 1

            for number, passage in enumerate(passages, start=1):
                # Section headings often carry the strongest retrieval signal.
                # Index them in the high-weight title field for every passage.
                title_terms = " ".join(lexical_tokens(f"{title} {passage.heading}"))
                body_terms = " ".join(lexical_tokens(passage.body))
                self.connection.execute(
                    """
                    INSERT INTO passages(
                        doc_id,passage_no,path,title,heading,start_line,end_line,body,title_terms,body_terms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(doc_id),
                        str(number),
                        relative,
                        title,
                        passage.heading,
                        str(passage.start_line),
                        str(passage.end_line),
                        passage.body,
                        title_terms,
                        body_terms,
                    ),
                )

        existing_paths = {row[0] for row in self.connection.execute("SELECT path FROM documents")}
        for removed in existing_paths - current_paths:
            row = self.connection.execute("SELECT id FROM documents WHERE path=?", (removed,)).fetchone()
            if row:
                self.connection.execute("DELETE FROM passages WHERE doc_id=?", (str(row[0]),))
                self.connection.execute("DELETE FROM documents WHERE id=?", (row[0],))
                stats["removed"] += 1

        self._set_setting("chunking", chunking)
        self._set_setting("documents_root", str(self.documents_root))
        self._set_setting("exclude_patterns", json.dumps(self.exclude_patterns, ensure_ascii=False))
        self._set_setting("last_indexed_at", utc_now())
        self.connection.commit()
        stats.update(self.status())
        return stats

    def status(self) -> dict[str, Any]:
        document_count = self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        passage_count = self.connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        answer_count = self.connection.execute("SELECT COUNT(*) FROM answer_runs").fetchone()[0]
        return {
            "version": APP_VERSION,
            "documents_root": str(self.documents_root),
            "database": str(self.db_path),
            "exclude_patterns": list(self.exclude_patterns),
            "chunking": self._setting("chunking") or "not-indexed",
            "documents": document_count,
            "passages": passage_count,
            "answer_runs": answer_count,
            "last_indexed_at": self._setting("last_indexed_at"),
        }

    def lexical_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        match = fts_query(query)
        rows = self.connection.execute(
            """
            SELECT rowid, doc_id, passage_no, path, title, heading, start_line, end_line, body,
                   bm25(passages, 0,0,0,0,0,0,0,0,5.0,1.0) AS rank
            FROM passages
            WHERE passages MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (match, max(1, min(limit, 100))),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for position, row in enumerate(rows, start=1):
            item = dict(row)
            item["bm25_rank"] = position
            item["bm25_score"] = round(-float(item.pop("rank")), 6)
            item["start_line"] = int(item["start_line"])
            item["end_line"] = int(item["end_line"])
            item["passage_no"] = int(item["passage_no"])
            item["snippet"] = make_snippet(item["body"], query)
            results.append(item)
        return results

    def cache_get(self, cache_key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT response_json FROM rerank_cache WHERE cache_key=?", (cache_key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, cache_key: str, response: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO rerank_cache(cache_key,response_json,created_at) VALUES(?,?,?)",
            (cache_key, json.dumps(response, ensure_ascii=False), utc_now()),
        )
        self.connection.commit()

    def record_answer_run(self, run: dict[str, Any]) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO answer_runs(
                query,generator_model,use_jev,candidate_count,returned_count,
                lexical_ms,jev_ms,first_token_ms,generation_ms,total_ms,
                prompt_tokens,completion_tokens,cost,answer,sources_json,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run["query"], run["generator_model"], int(run["use_jev"]),
                run["candidate_count"], run["returned_count"], run.get("lexical_ms"),
                run.get("jev_ms"), run.get("first_token_ms"), run.get("generation_ms"),
                run.get("total_ms"), run.get("prompt_tokens"), run.get("completion_tokens"),
                run.get("cost"), run["answer"],
                json.dumps(run["sources"], ensure_ascii=False), utc_now(),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def recent_answer_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT id,query,generator_model,use_jev,candidate_count,returned_count,
                   lexical_ms,jev_ms,first_token_ms,generation_ms,total_ms,
                   prompt_tokens,completion_tokens,cost,created_at
            FROM answer_runs ORDER BY id DESC LIMIT ?
            """,
            (max(1, min(limit, 100)),),
        ).fetchall()
        return [dict(row) for row in rows]


def make_snippet(text: str, query: str, max_chars: int = 520) -> str:
    if len(text) <= max_chars:
        return text
    lowered = text.lower()
    terms = [term for term in _TOKEN_RE.findall(query.lower()) if len(term) >= 2]
    positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - max_chars // 3)
    end = min(len(text), start + max_chars)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def jev_rerank(
    kb: KnowledgeBase,
    query: str,
    candidates: list[dict[str, Any]],
    provider: str = "openrouter",
    timeout: float = 60.0,
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        return [], {"used": False, "reason": "no_candidates"}

    identity = [f"{c['rowid']}:{sha256_text(c['body'])[:12]}" for c in candidates]
    cache_key = sha256_text(
        json.dumps(["jev-batched-v1", provider, query, identity], ensure_ascii=False)
    )
    if use_cache:
        cached = kb.cache_get(cache_key)
        if cached:
            return cached["results"], {**cached["meta"], "cache_hit": True}

    prepared: list[tuple[int, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        cid = f"c{index}"
        prepared.append(
            (index,
            {
                "id": cid,
                "title": candidate["title"],
                "heading": candidate["heading"],
                "source": candidate["path"],
                "text": make_snippet(candidate["body"], query, max_chars=1800),
            })
        )

    batches = [prepared[i : i + JEV_BATCH_SIZE] for i in range(0, len(prepared), JEV_BATCH_SIZE)]

    def score_batch(batch: list[tuple[int, dict[str, Any]]]) -> tuple[dict[str, Any], dict[int, float]]:
        questions: dict[str, Any] = {}
        compact_candidates = []
        for index, compact in batch:
            compact_candidates.append(compact)
            questions[f"relevance_{index}"] = {
                "type": "noul",
                "instructions": (
                    f"候选 c{index} 是否包含能直接帮助回答用户查询的具体证据？"
                    "仅主题相似但没有可用信息时回答否。"
                ),
                "criteria": {
                    "true": "含有可直接支持回答的事实、规则、步骤、数据或异常条件。",
                    "false": "无关，或仅提到相似词汇而没有回答所需证据。",
                },
            }
        questions["has_answer"] = {
            "type": "noul",
            "instructions": "这批候选中是否至少有一条包含足以帮助回答查询的具体证据？",
        }
        response = request_decision(
            provider,
            {"query": query, "candidates": compact_candidates},
            questions,
            timeout,
        )
        answers = response["answers"]
        scores = {
            index: float(answers.get(f"relevance_{index}", {}).get("noul", 0.0))
            for index, _ in batch
        }
        return response, scores

    rerank_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
        batch_results = list(executor.map(score_batch, batches))
    elapsed_ms = round((time.perf_counter() - rerank_started) * 1000)
    scores = {index: score for _, batch_scores in batch_results for index, score in batch_scores.items()}

    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        item["jev_score"] = round(scores.get(index, 0.0), 4)
        results.append(item)
    results.sort(key=lambda item: (-item["jev_score"], item["bm25_rank"]))
    for position, item in enumerate(results, start=1):
        item["final_rank"] = position

    responses = [response for response, _ in batch_results]
    usage: dict[str, float | int] = {}
    for response in responses:
        for key, value in (response.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    first_response = responses[0]
    has_answer = max(
        float((response.get("answers") or {}).get("has_answer", {}).get("noul", 0.0))
        for response in responses
    )
    meta = {
        "used": True,
        "cache_hit": False,
        "gateway": first_response.get("gateway", provider),
        "upstream_provider": first_response.get("provider"),
        "model": first_response.get("model"),
        "elapsed_ms": elapsed_ms,
        "has_answer": has_answer,
        "batch_count": len(batches),
        "batch_size": JEV_BATCH_SIZE,
        "usage": usage,
    }
    if use_cache:
        kb.cache_put(cache_key, {"results": results, "meta": meta})
    return results, meta


def run_search(
    kb: KnowledgeBase,
    query: str,
    top_k: int = 30,
    top_n: int = 10,
    use_jev: bool = True,
    provider: str = "openrouter",
    timeout: float = 60.0,
    threshold: float = 0.0,
    use_cache: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    lexical_started = time.perf_counter()
    candidates = kb.lexical_search(query, top_k)
    lexical_ms = round((time.perf_counter() - lexical_started) * 1000, 1)
    if use_jev:
        ranked, jev_meta = jev_rerank(kb, query, candidates, provider, timeout, use_cache)
        if threshold > 0:
            ranked = [item for item in ranked if item.get("jev_score", 0) >= threshold]
    else:
        ranked = candidates
        for position, item in enumerate(ranked, start=1):
            item["final_rank"] = position
        jev_meta = {"used": False, "reason": "disabled"}
    return {
        "query": query,
        "results": ranked[:top_n],
        "candidate_count": len(candidates),
        "returned_count": min(len(ranked), top_n),
        "timing": {
            "lexical_ms": lexical_ms,
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
        },
        "jev": jev_meta,
    }


def answer_messages(query: str, sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    evidence: list[str] = []
    for index, source in enumerate(sources, start=1):
        body = make_snippet(source["body"], query, max_chars=ANSWER_SOURCE_MAX_CHARS)
        evidence.append(
            f"[{index}] 文件: {source['path']}\n"
            f"章节: {source['heading']}\n"
            f"行号: {source['start_line']}-{source['end_line']}\n"
            f"内容:\n{body}"
        )
    system = (
        "你是严格依据本地知识库证据回答问题的助手。"
        "只能使用提供的证据，不得用训练记忆补充未出现的事实。"
        "每个关键结论后用 [1]、[2] 这样的编号引用来源。"
        "如果证据不足，明确说‘当前知识库中没有足够证据’，不要猜测。"
        "用与用户问题相同的语言，先给结论，再给必要的依据。"
    )
    user = f"用户问题：\n{query}\n\n本地知识库证据：\n\n" + "\n\n---\n\n".join(evidence)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _delta_text(delta: dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def openrouter_error_message(status: int, detail: str) -> str:
    """Turn provider error payloads into short, actionable UI messages."""
    provider_message = detail
    try:
        payload = json.loads(detail)
        error = payload.get("error", payload)
        if isinstance(error, dict):
            provider_message = str(error.get("message") or detail)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    if status == HTTPStatus.PAYMENT_REQUIRED:
        match = re.search(r"can only afford\s+(\d+)", provider_message, re.IGNORECASE)
        affordable = f"当前最多约可生成 {match.group(1)} 个 token。" if match else ""
        return (
            "MiniMax 生成额度不足。"
            f"{affordable}"
            "请在 OpenRouter 充值或提高该 Key 的额度上限；"
            "也可减少证据数量或继续降低 --max-tokens。"
        )

    compact = re.sub(r"\s+", " ", provider_message).strip()
    if len(compact) > 500:
        compact = compact[:497] + "..."
    return f"MiniMax HTTP {status}: {compact}"


def stream_openrouter_answer(
    query: str,
    sources: list[dict[str, Any]],
    model: str,
    timeout: float,
    max_tokens: int,
) -> Iterable[dict[str, Any]]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少环境变量 OPENROUTER_API_KEY")
    if not sources:
        yield {
            "type": "delta",
            "text": "当前知识库中没有检索到可用证据。",
        }
        yield {
            "type": "generation_done",
            "model": model,
            "first_token_ms": 0.0,
            "generation_ms": 0.0,
            "usage": {},
        }
        return

    body = json.dumps(
        {
            "model": model,
            "messages": answer_messages(query, sources),
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "include_reasoning": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": f"jev-rag/{APP_VERSION}",
            "X-Title": "Jev RAG",
        },
    )
    started = time.perf_counter()
    first_token_ms: float | None = None
    usage: dict[str, Any] = {}
    resolved_model = model
    emitted_text = False
    for attempt in range(GENERATOR_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("error"):
                        raise RuntimeError(json.dumps(payload["error"], ensure_ascii=False))
                    resolved_model = payload.get("model") or resolved_model
                    if isinstance(payload.get("usage"), dict):
                        usage = payload["usage"]
                    choices = payload.get("choices") or []
                    if not choices:
                        continue
                    text = _delta_text(choices[0].get("delta") or {})
                    if text:
                        emitted_text = True
                        if first_token_ms is None:
                            first_token_ms = round((time.perf_counter() - started) * 1000, 1)
                        yield {"type": "delta", "text": text}
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            retryable = exc.code in {408, 425, 429, 500, 502, 503, 504, 529}
            if retryable and not emitted_text and attempt < GENERATOR_RETRIES:
                time.sleep(0.4 * (2**attempt))
                continue
            raise RuntimeError(openrouter_error_message(exc.code, detail)) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if not emitted_text and attempt < GENERATOR_RETRIES:
                time.sleep(0.4 * (2**attempt))
                continue
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"MiniMax 网络请求失败: {reason}") from exc

    yield {
        "type": "generation_done",
        "model": resolved_model,
        "first_token_ms": first_token_ms,
        "generation_ms": round((time.perf_counter() - started) * 1000, 1),
        "usage": usage,
    }


WEB_APP = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev RAG</title><style>
:root{--ink:#18201d;--muted:#68736e;--line:#d9dfdc;--paper:#f6f7f4;--accent:#165c49;--soft:#e5efe9;--blue:#315d9b}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}
main{max-width:960px;margin:0 auto;padding:48px 24px 80px}header{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;margin-bottom:26px}
h1{font:600 34px/1.1 Georgia,"Songti SC",serif;margin:0}header p{margin:7px 0 0;color:var(--muted)}#status{text-align:right;color:var(--muted);font-size:13px}
.search{display:grid;grid-template-columns:1fr auto;gap:10px}input{width:100%;font:inherit;font-size:17px;padding:14px 16px;border:1px solid var(--line);border-radius:8px;background:white;outline:none}input:focus{border-color:var(--accent);box-shadow:0 0 0 3px #165c4918}
button{font:inherit;border:0;border-radius:8px;padding:0 22px;background:var(--accent);color:white;cursor:pointer}button:disabled{opacity:.5}.options{display:flex;gap:18px;align-items:center;margin:12px 2px 24px;color:var(--muted);font-size:13px}.options button{padding:5px 10px;background:transparent;color:var(--accent);border:1px solid var(--line)}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:16px 0}.metric{background:white;border:1px solid var(--line);border-radius:8px;padding:10px 12px}.metric b{display:block;font-size:18px}.metric span{font-size:11px;color:var(--muted)}
.panel{background:white;border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin:14px 0}.panel h2{font-size:15px;margin:0 0 12px;color:var(--muted);font-weight:600}.answer{white-space:pre-wrap;font-size:16px;min-height:42px}.cursor:after{content:'▋';color:var(--accent);animation:blink .8s infinite}@keyframes blink{50%{opacity:0}}
.summary{color:var(--muted);font-size:13px}.source{padding:13px 0;border-top:1px solid var(--line)}.source:first-child{border-top:0}.source h3{font-size:15px;margin:0 0 3px}.meta{display:flex;flex-wrap:wrap;gap:6px 14px;color:var(--muted);font-size:12px}.score{color:var(--accent);font-weight:650}.snippet{white-space:pre-wrap;margin:8px 0 0;color:#36423d;max-height:130px;overflow:hidden}.empty{padding:42px 0;text-align:center;color:var(--muted)}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.error{color:#9d2b24}
@media(max-width:700px){main{padding:28px 16px}.search{grid-template-columns:1fr}.search button{height:48px}header{display:block}#status{text-align:left;margin-top:12px}.metrics{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main>
<header><div><h1>Jev RAG</h1><p>BM25 → Jev → MiniMax M3，全链路无向量问答。</p></div><div id="status">读取索引…</div></header>
<form id="form" class="search"><input id="q" autocomplete="off" placeholder="输入一个需要从本地文档回答的问题"><button id="go">提问</button></form>
<div class="options"><label><input id="jev" type="checkbox" checked style="width:auto"> 使用 Jev 重排</label><button id="reindex" type="button">重新扫描文档</button></div>
<div id="metrics" class="metrics" hidden></div><div id="summary" class="summary"></div>
<section id="answerPanel" class="panel" hidden><h2>MiniMax 回答</h2><div id="answer" class="answer"></div></section>
<section id="sourcesPanel" class="panel" hidden><h2>检索证据</h2><div id="sources"></div></section>
<div id="empty" class="empty">输入问题后，页面会流式显示最终答案和各阶段耗时。</div>
</main><script>
const $=s=>document.querySelector(s);const esc=s=>(s??'').toString().replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>v===null||v===undefined?'—':`${Math.round(v)} ms`;
async function status(){const r=await fetch('/api/status');const d=await r.json();$('#status').innerHTML=`${d.documents} 个文件 · ${d.passages} 条记录<br>${d.answer_runs} 次已记录问答 · <code>${esc(d.chunking)}</code>`;}
function showMetrics(m={}){const items=[['BM25',m.lexical_ms],['Jev',m.jev_ms],['端到端首 token',m.client_first_token_ms??m.first_token_ms],['MiniMax 完成',m.generation_ms],['端到端总时长',m.client_total_ms??m.total_ms]];$('#metrics').hidden=false;$('#metrics').innerHTML=items.map(([k,v])=>`<div class="metric"><b>${fmt(v)}</b><span>${k}</span></div>`).join('')}
function showSources(xs){$('#sourcesPanel').hidden=!xs.length;$('#sources').innerHTML=xs.map((x,i)=>`<div class="source"><h3>[${i+1}] ${esc(x.title)}</h3><div class="meta"><span>${esc(x.path)}:${x.start_line}-${x.end_line}</span><span>BM25 #${x.bm25_rank}</span>${x.jev_score!==undefined?`<span class="score">Jev ${(x.jev_score*100).toFixed(0)}%</span>`:''}</div><div class="snippet">${esc(x.snippet)}</div></div>`).join('')}
$('#form').onsubmit=async e=>{e.preventDefault();const q=$('#q').value.trim();if(!q)return;const clientStart=performance.now();let firstClient=null,answer='',metrics={};$('#go').disabled=true;$('#go').textContent='回答中…';$('#empty').hidden=true;$('#answerPanel').hidden=false;$('#sourcesPanel').hidden=true;$('#answer').textContent='';$('#answer').classList.add('cursor');$('#summary').textContent='正在执行 BM25 召回…';showMetrics(metrics);
try{const r=await fetch('/api/answer',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({query:q,use_jev:$('#jev').checked})});if(!r.ok){const d=await r.json();throw new Error(d.error||'问答失败')}const reader=r.body.getReader(),decoder=new TextDecoder();let buffer='';while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const lines=buffer.split('\n');buffer=lines.pop();for(const line of lines){if(!line.trim())continue;const ev=JSON.parse(line);if(ev.type==='stage'){Object.assign(metrics,ev.metrics);$('#summary').textContent=ev.message;showMetrics(metrics)}else if(ev.type==='sources'){showSources(ev.results)}else if(ev.type==='delta'){if(firstClient===null)firstClient=performance.now()-clientStart;answer+=ev.text;$('#answer').textContent=answer}else if(ev.type==='done'){metrics={...metrics,...ev.metrics,client_first_token_ms:firstClient,client_total_ms:performance.now()-clientStart};showMetrics(metrics);$('#summary').textContent=`记录 #${ev.run_id} · ${ev.model} · 候选 ${ev.candidate_count} 条 · 证据 ${ev.returned_count} 条${ev.cost!=null?` · $${Number(ev.cost).toFixed(6)}`:''}`}else if(ev.type==='error'){throw new Error(ev.error)}}}}catch(e){$('#summary').innerHTML=`<span class="error">${esc(e.message)}</span>`;if(!answer)$('#answer').textContent='未能生成答案。'}finally{$('#answer').classList.remove('cursor');$('#go').disabled=false;$('#go').textContent='提问';status()}};
$('#reindex').onclick=async()=>{const b=$('#reindex');b.disabled=true;b.textContent='扫描中…';try{const r=await fetch('/api/index',{method:'POST'});const d=await r.json();if(!r.ok)throw new Error(d.error);await status();b.textContent=`完成：${d.documents} 个文件`;}catch(e){b.textContent='失败：'+e.message}setTimeout(()=>{b.disabled=false;b.textContent='重新扫描文档'},2500)};status();
</script></body></html>"""


class AppHandler(BaseHTTPRequestHandler):
    kb: KnowledgeBase
    config: dict[str, Any]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[web] " + fmt % args + "\n")

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("无效的 Content-Length") from exc
        if length < 0 or length > 1_048_576:
            raise ValueError("请求体过大（最大 1 MiB）")
        return json.loads(self.rfile.read(length) or b"{}")

    def _stream_event(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()

    def _answer_stream(self, payload: dict[str, Any]) -> None:
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("查询不能为空")
        if len(query) > 8000:
            raise ValueError("查询过长（最大 8000 字符）")
        use_jev = bool(payload.get("use_jev", True))
        top_k = int(payload.get("top_k", self.config["top_k"]))
        top_n = int(payload.get("top_n", self.config["top_n"]))
        threshold = float(payload.get("threshold", self.config["threshold"]))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold 必须在 0 到 1 之间")
        pipeline_started = time.perf_counter()

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        try:
            lexical_started = time.perf_counter()
            candidates = self.kb.lexical_search(query, top_k)
            lexical_ms = round((time.perf_counter() - lexical_started) * 1000, 1)
            self._stream_event(
                {
                    "type": "stage",
                    "message": f"BM25 已召回 {len(candidates)} 条候选，正在进行 Jev 重排…" if use_jev else f"BM25 已召回 {len(candidates)} 条候选…",
                    "metrics": {"lexical_ms": lexical_ms},
                }
            )

            jev_ms = 0.0
            jev_meta: dict[str, Any] = {"used": False, "reason": "disabled"}
            if use_jev and candidates:
                jev_started = time.perf_counter()
                ranked, jev_meta = jev_rerank(
                    self.kb,
                    query,
                    candidates,
                    self.config["provider"],
                    self.config["timeout"],
                    True,
                )
                if threshold > 0:
                    ranked = [item for item in ranked if item.get("jev_score", 0) >= threshold]
                jev_ms = round((time.perf_counter() - jev_started) * 1000, 1)
            else:
                ranked = candidates
                for position, item in enumerate(ranked, start=1):
                    item["final_rank"] = position
            sources = ranked[:top_n]
            public_sources = [
                {
                    key: item[key]
                    for key in (
                        "path", "title", "heading", "start_line", "end_line",
                        "bm25_rank", "bm25_score", "snippet", "final_rank",
                    )
                    if key in item
                }
                | ({"jev_score": item["jev_score"]} if "jev_score" in item else {})
                for item in sources
            ]
            self._stream_event({"type": "sources", "results": public_sources})
            cache_note = "（命中缓存）" if jev_meta.get("cache_hit") else ""
            self._stream_event(
                {
                    "type": "stage",
                    "message": f"Jev 重排完成{cache_note}，MiniMax 正在生成答案…" if use_jev else "检索完成，MiniMax 正在生成答案…",
                    "metrics": {"lexical_ms": lexical_ms, "jev_ms": jev_ms},
                }
            )

            answer_parts: list[str] = []
            generation_meta: dict[str, Any] = {}
            first_pipeline_token_ms: float | None = None
            for event in stream_openrouter_answer(
                query,
                sources,
                self.config["generator_model"],
                self.config["timeout"],
                self.config["max_tokens"],
            ):
                if event["type"] == "delta":
                    if first_pipeline_token_ms is None:
                        first_pipeline_token_ms = round((time.perf_counter() - pipeline_started) * 1000, 1)
                        self._stream_event(
                            {
                                "type": "stage",
                                "message": "MiniMax 已输出首个 token，正在继续生成…",
                                "metrics": {
                                    "lexical_ms": lexical_ms,
                                    "jev_ms": jev_ms,
                                    "first_token_ms": first_pipeline_token_ms,
                                },
                            }
                        )
                    answer_parts.append(event["text"])
                    self._stream_event(event)
                elif event["type"] == "generation_done":
                    generation_meta = event

            total_ms = round((time.perf_counter() - pipeline_started) * 1000, 1)
            answer = "".join(answer_parts)
            usage = generation_meta.get("usage") or {}
            jev_usage = jev_meta.get("usage") or {}
            generator_cost = usage.get("cost")
            jev_cost = jev_usage.get("cost")
            total_cost = None
            if generator_cost is not None or jev_cost is not None:
                total_cost = float(generator_cost or 0) + float(jev_cost or 0)
            run = {
                "query": query,
                "generator_model": generation_meta.get("model", self.config["generator_model"]),
                "use_jev": use_jev,
                "candidate_count": len(candidates),
                "returned_count": len(sources),
                "lexical_ms": lexical_ms,
                "jev_ms": jev_ms,
                "first_token_ms": first_pipeline_token_ms,
                "generation_ms": generation_meta.get("generation_ms"),
                "total_ms": total_ms,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost": total_cost,
                "answer": answer,
                "sources": public_sources,
            }
            run_id = self.kb.record_answer_run(run)
            self._stream_event(
                {
                    "type": "done",
                    "run_id": run_id,
                    "model": run["generator_model"],
                    "candidate_count": len(candidates),
                    "returned_count": len(sources),
                    "cost": total_cost,
                    "usage": usage,
                    "metrics": {
                        "lexical_ms": lexical_ms,
                        "jev_ms": jev_ms,
                        "first_token_ms": first_pipeline_token_ms,
                        "generation_ms": generation_meta.get("generation_ms"),
                        "total_ms": total_ms,
                    },
                }
            )
        except Exception as exc:
            sys.stderr.write(f"[answer-error] {type(exc).__name__}: {exc}\n")
            self._stream_event({"type": "error", "error": str(exc)})

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            body = WEB_APP.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self._json(self.kb.status())
        elif path == "/api/runs":
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            limit = int(query.get("limit", ["20"])[0])
            self._json({"runs": self.kb.recent_answer_runs(limit)})
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == "/api/answer":
                self._answer_stream(self._read_json())
            elif path == "/api/search":
                payload = self._read_json()
                query = str(payload.get("query", "")).strip()
                if not query:
                    raise ValueError("查询不能为空")
                result = run_search(
                    self.kb,
                    query,
                    top_k=int(payload.get("top_k", self.config["top_k"])),
                    top_n=int(payload.get("top_n", self.config["top_n"])),
                    use_jev=bool(payload.get("use_jev", True)),
                    provider=self.config["provider"],
                    timeout=self.config["timeout"],
                    threshold=float(payload.get("threshold", 0.0)),
                )
                self._json(result)
            elif path == "/api/index":
                self._json(self.kb.index(chunking=self.config["chunking"]))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def print_search(result: dict[str, Any]) -> None:
    print(f"\n查询: {result['query']}")
    print(
        f"候选 {result['candidate_count']} 条，返回 {result['returned_count']} 条；"
        f"BM25 {result['timing']['lexical_ms']} ms，总计 {result['timing']['total_ms']} ms"
    )
    if result["jev"].get("used"):
        suffix = "（缓存）" if result["jev"].get("cache_hit") else ""
        print(
            f"Jev: {result['jev'].get('model')} / {result['jev'].get('elapsed_ms')} ms{suffix} / "
            f"has_answer={result['jev'].get('has_answer')}"
        )
    for item in result["results"]:
        jev = f"  Jev={item['jev_score']:.0%}" if "jev_score" in item else ""
        print(f"\n{item['final_rank']}. {item['title']}  [BM25 #{item['bm25_rank']}{jev}]")
        print(f"   {item['path']}:{item['start_line']}-{item['end_line']}  ·  {item['heading']}")
        print(textwrap.indent(item["snippet"].replace("\n", " "), "   "))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jev RAG：SQLite FTS5/BM25 → Jev 的无向量本地知识问答")
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS, help="文档目录")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite 索引路径")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="环境变量文件")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="排除的相对路径模式，可重复使用，例如 'project/tmp/**'",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="建立或更新本地索引")
    index_parser.add_argument("--chunking", choices=["auto", "none", "paragraph"], default="auto")
    index_parser.add_argument("--rebuild", action="store_true", help="删除旧索引后重建")

    search_parser = subparsers.add_parser("search", help="搜索本地知识库")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=30, help="BM25 候选数")
    search_parser.add_argument("--top-n", type=int, default=10, help="Jev 重排后的最终证据数")
    search_parser.add_argument("--threshold", type=float, default=0.0, help="Jev 最低相关度")
    search_parser.add_argument("--provider", choices=["openrouter", "typesafe"], default="openrouter")
    search_parser.add_argument("--lexical-only", action="store_true", help="只运行 BM25，不调用 Jev")
    search_parser.add_argument("--no-cache", action="store_true", help="忽略 Jev 结果缓存")
    search_parser.add_argument("--timeout", type=float, default=60.0)
    search_parser.add_argument("--json", action="store_true", dest="as_json")

    serve_parser = subparsers.add_parser("serve", help="启动本地网页")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--chunking", choices=["auto", "none", "paragraph"], default="auto")
    serve_parser.add_argument("--provider", choices=["openrouter", "typesafe"], default="openrouter")
    serve_parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"MiniMax 最大输出 token 数（默认 {DEFAULT_MAX_TOKENS}）",
    )
    serve_parser.add_argument("--top-k", type=int, default=30)
    serve_parser.add_argument("--top-n", type=int, default=10)
    serve_parser.add_argument("--threshold", type=float, default=0.0, help="Jev 最低相关度")
    serve_parser.add_argument("--timeout", type=float, default=60.0)
    serve_parser.add_argument("--no-index", action="store_true", help="启动时不自动扫描")

    subparsers.add_parser("status", help="显示索引状态")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(args.env_file)
    kb = KnowledgeBase(args.db, args.documents, args.exclude)
    try:
        if args.command == "index":
            print(json.dumps(kb.index(args.chunking, args.rebuild), ensure_ascii=False, indent=2))
        elif args.command == "status":
            print(json.dumps(kb.status(), ensure_ascii=False, indent=2))
        elif args.command == "search":
            result = run_search(
                kb,
                args.query,
                top_k=args.top_k,
                top_n=args.top_n,
                use_jev=not args.lexical_only,
                provider=args.provider,
                timeout=args.timeout,
                threshold=args.threshold,
                use_cache=not args.no_cache,
            )
            if args.as_json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print_search(result)
        elif args.command == "serve":
            if not args.no_index:
                stats = kb.index(args.chunking)
                print(f"已索引 {stats['documents']} 个文件 / {stats['passages']} 条记录")
            AppHandler.kb = kb
            AppHandler.config = {
                "chunking": args.chunking,
                "provider": args.provider,
                "generator_model": args.generator_model,
                "max_tokens": args.max_tokens,
                "top_k": args.top_k,
                "top_n": args.top_n,
                "threshold": args.threshold,
                "timeout": args.timeout,
            }
            # A single-process local app does not need request concurrency, and
            # keeping requests on the owner thread makes SQLite access simple
            # and deterministic. Jev calls still batch all candidate judgments
            # into one network request.
            server = HTTPServer((args.host, args.port), AppHandler)
            print(f"本地知识检索已启动：http://{args.host}:{args.port}")
            print("按 Ctrl+C 停止")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\n已停止")
            finally:
                server.server_close()
        return 0
    except (RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    finally:
        kb.close()


if __name__ == "__main__":
    raise SystemExit(main())
