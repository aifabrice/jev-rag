import unittest

from scripts.benchmark_beir import metrics_for_query, reciprocal_rank_fusion


class BeirMetricTests(unittest.TestCase):
    def test_perfect_ranking_scores_one(self):
        metrics = metrics_for_query(["high", "low"], {"high": 2, "low": 1}, [2])
        self.assertAlmostEqual(metrics["ndcg@2"], 1.0)
        self.assertAlmostEqual(metrics["recall@2"], 1.0)
        self.assertAlmostEqual(metrics["precision@2"], 1.0)
        self.assertAlmostEqual(metrics["mrr@2"], 1.0)
        self.assertAlmostEqual(metrics["map@2"], 1.0)

    def test_late_relevant_document_is_penalized(self):
        metrics = metrics_for_query(["noise", "relevant"], {"relevant": 1}, [2])
        self.assertLess(metrics["ndcg@2"], 1.0)
        self.assertAlmostEqual(metrics["recall@2"], 1.0)
        self.assertAlmostEqual(metrics["precision@2"], 0.5)
        self.assertAlmostEqual(metrics["mrr@2"], 0.5)
        self.assertAlmostEqual(metrics["map@2"], 0.5)

    def test_rrf_rewards_documents_found_by_both_retrievers(self):
        lexical = [
            {"path": "a.txt", "rowid": 1, "body": "a", "bm25_rank": 1},
            {"path": "b.txt", "rowid": 2, "body": "b", "bm25_rank": 2},
        ]
        documents = {
            "a": lexical[0],
            "b": lexical[1],
            "c": {"path": "c.txt", "rowid": 3, "body": "c"},
        }
        fused = reciprocal_rank_fusion(
            lexical,
            ["b", "c"],
            documents,
            {"a.txt": "a", "b.txt": "b", "c.txt": "c"},
            limit=3,
        )
        self.assertEqual([item["path"] for item in fused], ["b.txt", "a.txt", "c.txt"])
        self.assertEqual(fused[0]["original_bm25_rank"], 2)
        self.assertEqual(fused[0]["vector_rank"], 1)


if __name__ == "__main__":
    unittest.main()
