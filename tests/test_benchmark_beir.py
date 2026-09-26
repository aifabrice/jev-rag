import unittest

from scripts.benchmark_beir import metrics_for_query


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


if __name__ == "__main__":
    unittest.main()
