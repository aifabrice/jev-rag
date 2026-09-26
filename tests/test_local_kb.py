import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from local_kb import (
    ANSWER_SOURCE_MAX_CHARS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRIEVAL_MODE,
    KnowledgeBase,
    answer_messages,
    build_parser,
    checkout_exclude_pattern,
    discover_documents_root,
    fts_query,
    jev_rerank,
    lexical_tokens,
    openrouter_error_message,
    reciprocal_rank_fusion,
    split_passages,
)


class TokenTests(unittest.TestCase):
    def test_chinese_bigrams_and_english_words(self):
        tokens = lexical_tokens("退款需要五天 Access Token")
        self.assertIn("退款", tokens)
        self.assertIn("需要", tokens)
        self.assertIn("access", tokens)
        self.assertIn("token", tokens)

    def test_fts_query_is_safe(self):
        query = fts_query('API "token" + 退款')
        self.assertIn('"api"', query)
        self.assertIn('"退款"', query)


class PassageTests(unittest.TestCase):
    def test_none_keeps_whole_document(self):
        text = "# 标题\n\n" + "内容" * 4000
        passages = split_passages(text, "doc", mode="none")
        self.assertEqual(len(passages), 1)

    def test_auto_keeps_short_document(self):
        passages = split_passages("一段很短的内容。", "doc", mode="auto")
        self.assertEqual(len(passages), 1)

    def test_paragraph_splits_sections(self):
        text = "# A\n\n" + "alpha " * 500 + "\n\n# B\n\n" + "beta " * 500
        passages = split_passages(text, "doc", mode="paragraph")
        self.assertGreaterEqual(len(passages), 2)
        self.assertEqual(passages[0].heading, "A")
        self.assertEqual(passages[-1].heading, "B")

    def test_answer_prompt_numbers_sources(self):
        messages = answer_messages(
            "问题",
            [{"path": "a.md", "heading": "A", "start_line": 1, "end_line": 2, "body": "证据"}],
        )
        self.assertIn("[1]", messages[1]["content"])
        self.assertIn("只能使用", messages[0]["content"])

    def test_answer_prompt_limits_each_source(self):
        messages = answer_messages(
            "needle",
            [{"path": "a.md", "heading": "A", "start_line": 1, "end_line": 2,
              "body": "x" * 5000 + " needle " + "y" * 5000}],
        )
        self.assertLess(len(messages[1]["content"]), ANSWER_SOURCE_MAX_CHARS + 300)

    def test_openrouter_402_is_concise(self):
        detail = json.dumps({"error": {"message": "You can only afford 280 tokens", "previous_errors": ["long"]}})
        message = openrouter_error_message(402, detail)
        self.assertIn("280", message)
        self.assertNotIn("previous_errors", message)

    def test_serve_default_uses_low_cost_output_limit(self):
        args = build_parser().parse_args(["serve"])
        self.assertEqual(args.max_tokens, DEFAULT_MAX_TOKENS)

    def test_default_retrieval_is_vector_free(self):
        args = build_parser().parse_args(["serve"])
        self.assertEqual(args.retrieval_mode, DEFAULT_RETRIEVAL_MODE)
        self.assertEqual(args.retrieval_mode, "bm25")
        self.assertEqual(args.embedding_model, DEFAULT_EMBEDDING_MODEL)

    def test_default_document_root_prefers_documents_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            documents = home / "Documents"
            documents.mkdir(parents=True)
            with patch.dict("local_kb.os.environ", {"JEV_RAG_DOCUMENTS": ""}):
                self.assertEqual(discover_documents_root(home=home), documents)

    def test_default_document_root_falls_back_to_example_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            cwd = Path(tmp) / "checkout"
            home.mkdir()
            with patch.dict("local_kb.os.environ", {"JEV_RAG_DOCUMENTS": ""}):
                self.assertEqual(discover_documents_root(home=home, cwd=cwd), cwd / "knowledge")

    def test_checkout_is_excluded_from_discovered_parent(self):
        root = Path(__file__).resolve().parents[2]
        pattern = checkout_exclude_pattern(root)
        checkout_name = Path(__file__).resolve().parents[1].name
        self.assertEqual(pattern, f"{checkout_name}/**")


class SearchTests(unittest.TestCase):
    def test_rrf_combines_bm25_and_vector_ranks(self):
        base = {
            "title": "t", "heading": "h", "path": "p", "body": "b",
            "start_line": 1, "end_line": 1, "snippet": "b",
        }
        bm25 = [dict(base, rowid=1, bm25_rank=1), dict(base, rowid=2, bm25_rank=2)]
        vector = [dict(base, rowid=2, vector_rank=1), dict(base, rowid=3, vector_rank=2)]
        fused = reciprocal_rank_fusion(bm25, vector, limit=3, rrf_k=60)

        self.assertEqual([item["rowid"] for item in fused], [2, 1, 3])
        self.assertEqual(fused[0]["bm25_rank"], 2)
        self.assertEqual(fused[0]["vector_rank"], 1)
        self.assertEqual(fused[0]["retrieval_rank"], 1)

    def test_bm25_finds_two_character_chinese_word(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docs"
            root.mkdir()
            (root / "billing.md").write_text("重复扣款可以申请退款，五个工作日到账。", encoding="utf-8")
            (root / "weather.md").write_text("今天天气晴朗，适合散步。", encoding="utf-8")
            kb = KnowledgeBase(Path(tmp) / "kb.db", root)
            try:
                stats = kb.index("none")
                self.assertEqual(stats["documents"], 2)
                results = kb.lexical_search("退款", 10)
                self.assertTrue(results)
                self.assertEqual(results[0]["path"], "billing.md")
            finally:
                kb.close()

    def test_jev_rerank_batches_large_candidate_sets(self):
        class MemoryCache:
            def __init__(self):
                self.value = None

            def cache_get(self, _key):
                return self.value

            def cache_put(self, _key, value):
                self.value = value

        candidates = [
            {
                "rowid": index + 1,
                "body": f"candidate {index}",
                "title": f"title {index}",
                "heading": "section",
                "path": f"doc-{index}.md",
                "bm25_rank": index + 1,
            }
            for index in range(23)
        ]
        calls = []
        lock = threading.Lock()

        def fake_decision(provider, state, questions, timeout):
            del timeout
            keys = [key for key in questions if key.startswith("relevance_")]
            with lock:
                calls.append(len(keys))
            answers = {
                key: {"type": "noul", "noul": int(key.rsplit("_", 1)[1]) / 100}
                for key in keys
            }
            answers["has_answer"] = {"type": "noul", "noul": 0.9}
            return {
                "gateway": provider,
                "provider": "test",
                "model": "jev-test",
                "answers": answers,
                "usage": {"input_tokens": len(keys), "output_tokens": 1, "cost": 0.01},
            }

        with patch("local_kb.request_decision", side_effect=fake_decision):
            ranked, meta = jev_rerank(MemoryCache(), "query", candidates)

        self.assertEqual(sorted(calls), [3, 10, 10])
        self.assertEqual(meta["batch_count"], 3)
        self.assertEqual(meta["usage"]["input_tokens"], 23)
        self.assertAlmostEqual(meta["usage"]["cost"], 0.03)
        self.assertEqual(ranked[0]["rowid"], 23)
        self.assertEqual(ranked[-1]["rowid"], 1)

    def test_excluded_directory_is_not_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docs"
            (root / "real").mkdir(parents=True)
            (root / "system").mkdir()
            (root / "real" / "note.md").write_text("真实知识", encoding="utf-8")
            (root / "system" / "fixture.md").write_text("演示预设", encoding="utf-8")
            kb = KnowledgeBase(Path(tmp) / "kb.db", root, ["system/**"])
            try:
                stats = kb.index("none")
                self.assertEqual(stats["documents"], 1)
                self.assertEqual(stats["exclude_patterns"], ["system/**"])
            finally:
                kb.close()

    def test_answer_run_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docs"
            root.mkdir()
            kb = KnowledgeBase(Path(tmp) / "kb.db", root)
            try:
                run_id = kb.record_answer_run(
                    {
                        "query": "q", "generator_model": "m", "use_jev": True,
                        "candidate_count": 2, "returned_count": 1, "lexical_ms": 1.0,
                        "jev_ms": 2.0, "first_token_ms": 3.0, "generation_ms": 4.0,
                        "total_ms": 5.0, "prompt_tokens": 10, "completion_tokens": 2,
                        "cost": 0.001, "answer": "a", "sources": [{"path": "a.md"}],
                    }
                )
                self.assertEqual(run_id, 1)
                self.assertEqual(kb.status()["answer_runs"], 1)
                self.assertEqual(kb.recent_answer_runs(1)[0]["query"], "q")
            finally:
                kb.close()


if __name__ == "__main__":
    unittest.main()
