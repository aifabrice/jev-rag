import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import benchmark


class BenchmarkTests(unittest.TestCase):
    def run_fixture(self, *options):
        cases = [
            {"query": "found", "expected_paths": ["answer.md"]},
            {"query": "missing", "expected_paths": ["absent.md"]},
        ]
        results = [
            {"results": [{"path": "answer.md", "final_rank": 2}],
             "returned_count": 1, "timing": {"total_ms": 1.25}},
            {"results": [], "returned_count": 0, "timing": {"total_ms": 2.75}},
        ]
        output = io.StringIO()
        with (
            patch("sys.argv", ["benchmark.py", *options]),
            patch.object(benchmark, "load_cases", return_value=cases),
            patch.object(benchmark, "KnowledgeBase") as kb,
            patch.object(benchmark, "run_search", side_effect=results) as search,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(benchmark.main(), 0)
        kb.return_value.close.assert_called_once_with()
        self.assertEqual(search.call_count, 2)
        for call in search.call_args_list:
            self.assertEqual(call.kwargs["use_jev"], "--use-jev" in options)
        return output.getvalue()

    def test_markdown_summary_and_miss(self):
        self.assertEqual(self.run_fixture("--markdown"), """## Summary

| Mode | Cases | Hit rate | MRR | Median latency (ms) | p95 latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 2 | 50.0% | 0.2500 | 2.00 | 2.75 |

## Per-query results

| Query | First relevant rank | Latency (ms) |
| --- | ---: | ---: |
| found | 2 | 1.25 |
| missing | miss | 2.75 |
""")

    def test_default_text_unchanged(self):
        self.assertEqual(self.run_fixture(), """Mode: bm25 | cases: 2
Hit rate: 50.0% | MRR: 0.2500
Latency: median 2.00 ms | p95 2.75 ms
- rank=2 1.25 ms | found
- rank=miss 2.75 ms | missing
""")

    def test_json_unchanged(self):
        expected = {
            "mode": "bm25", "cases": 2, "hit_rate": 0.5, "mrr": 0.25,
            "latency_ms": {"median": 2.0, "p95": 2.75},
            "results": [
                {"query": "found", "expected_paths": ["answer.md"],
                 "first_relevant_rank": 2, "hit": True, "reciprocal_rank": 0.5,
                 "returned_count": 1, "latency_ms": 1.25},
                {"query": "missing", "expected_paths": ["absent.md"],
                 "first_relevant_rank": None, "hit": False, "reciprocal_rank": 0.0,
                 "returned_count": 0, "latency_ms": 2.75},
            ],
        }
        self.assertEqual(self.run_fixture("--json"),
                         json.dumps(expected, ensure_ascii=False, indent=2) + "\n")

    def test_jev_requires_explicit_flag(self):
        for option in ("--markdown", "--json"):
            with self.subTest(option=option):
                self.assertIn("bm25+jev", self.run_fixture(option, "--use-jev"))

    def test_output_options_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            benchmark.build_parser().parse_args(["--json", "--markdown"])
        self.assertEqual(error.exception.code, 2)

    def test_markdown_escaping(self):
        query = '\\ | `code` *bold* _x_ [link](url) <script> & &#124;\r\n退款\rnext\nlast'
        escaped = benchmark.markdown_cell(query)
        self.assertEqual(
            escaped,
            r'\\ \| \`code\` \*bold\* \_x\_ \[link\]\(url\) '
            r'&lt;script&gt; &amp; &amp;\#124;<br>退款<br>next<br>last',
        )
        report = benchmark.format_markdown({
            "mode": "bm25", "cases": 1, "hit_rate": 0.0, "mrr": 0.0,
            "latency_ms": {"median": 0.0, "p95": 0.0},
            "results": [{"query": query, "first_relevant_rank": None, "latency_ms": 0.0}],
        })
        self.assertEqual(report.splitlines()[-1], f"| {escaped} | miss | 0.00 |")
        self.assertIn("| bm25 | 1 | 0.0% | 0.0000 | 0.00 | 0.00 |", report)

    def test_sample_benchmark_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("local_kb.jev_rerank", side_effect=AssertionError("Jev must stay off")),
                patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")),
                patch("sys.argv", ["benchmark.py", "--markdown", "--db",
                                   str(Path(tmp) / "benchmark.db")]),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(benchmark.main(), 0)
            report = output.getvalue()
            cases = benchmark.load_cases(benchmark.ROOT / "benchmarks" / "sample_queries.jsonl")
            self.assertIn(f"| bm25 | {len(cases)} |", report)
            for case in cases:
                self.assertIn(f"| {benchmark.markdown_cell(case['query'])} |", report)


if __name__ == "__main__":
    unittest.main()
