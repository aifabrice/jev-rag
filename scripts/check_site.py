#!/usr/bin/env python3
"""Validate the static site's discovery files without third-party packages."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
BASE_URL = "https://aifabrice.github.io/jev-rag/"


class MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.canonicals: list[str] = []
        self.descriptions: list[str] = []
        self.json_ld: list[str] = []
        self._in_json_ld = False
        self._json_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "link" and values.get("rel") == "canonical" and values.get("href"):
            self.canonicals.append(values["href"] or "")
        if tag == "meta" and values.get("name") == "description":
            self.descriptions.append(values.get("content") or "")
        if tag == "script" and values.get("type") == "application/ld+json":
            self._in_json_ld = True
            self._json_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self._json_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_json_ld:
            self.json_ld.append("".join(self._json_parts))
            self._in_json_ld = False


def fail(message: str) -> None:
    print(f"site check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_html(relative_path: str, canonical: str) -> None:
    path = SITE / relative_path
    if not path.is_file():
        fail(f"missing {path.relative_to(ROOT)}")
    parser = MetadataParser()
    parser.feed(path.read_text(encoding="utf-8"))
    if parser.canonicals != [canonical]:
        fail(f"{relative_path} canonical is {parser.canonicals!r}, expected {canonical!r}")
    if len(parser.descriptions) != 1 or not parser.descriptions[0].strip():
        fail(f"{relative_path} must contain one non-empty meta description")
    if not parser.json_ld:
        fail(f"{relative_path} has no JSON-LD")
    for block in parser.json_ld:
        try:
            json.loads(block)
        except json.JSONDecodeError as exc:
            fail(f"{relative_path} contains invalid JSON-LD: {exc}")


def check_sitemap() -> None:
    path = SITE / "sitemap.xml"
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        fail(f"invalid sitemap.xml: {exc}")
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locations = {node.text for node in root.findall("s:url/s:loc", namespace)}
    expected = {
        BASE_URL,
        f"{BASE_URL}faq.html",
        f"{BASE_URL}seven-pipelines.html",
        f"{BASE_URL}jev-rag-use-cases.html",
        f"{BASE_URL}how-to-use-jev-for-rag.html",
        f"{BASE_URL}zh/",
        f"{BASE_URL}zh/jev-rag-guide.html",
    }
    if not expected.issubset(locations):
        fail(f"sitemap is missing {sorted(expected - locations)}")


def check_robots() -> None:
    robots = (SITE / "robots.txt").read_text(encoding="utf-8")
    required = ["User-agent: OAI-SearchBot", "Allow: /", f"Sitemap: {BASE_URL}sitemap.xml"]
    missing = [line for line in required if line not in robots]
    if missing:
        fail(f"robots.txt is missing {missing}")


def main() -> int:
    check_html("index.html", BASE_URL)
    check_html("faq.html", f"{BASE_URL}faq.html")
    check_html("seven-pipelines.html", f"{BASE_URL}seven-pipelines.html")
    check_html("jev-rag-use-cases.html", f"{BASE_URL}jev-rag-use-cases.html")
    check_html("how-to-use-jev-for-rag.html", f"{BASE_URL}how-to-use-jev-for-rag.html")
    check_html("zh/index.html", f"{BASE_URL}zh/")
    check_html("zh/jev-rag-guide.html", f"{BASE_URL}zh/jev-rag-guide.html")
    check_sitemap()
    check_robots()
    for name in ("llms.txt", "llms-full.txt"):
        text = (SITE / name).read_text(encoding="utf-8")
        if "Jev RAG" not in text or BASE_URL not in text:
            fail(f"{name} is missing the canonical project name or URL")
    print("Static site discovery check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
