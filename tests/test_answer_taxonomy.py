import tempfile
import unittest
from pathlib import Path

import numpy as np

from answer_taxonomy import (
    answer_taxonomy_candidate_expansion,
    answer_taxonomy_rank_fusion,
    build_answer_supervised_taxonomy,
    load_answer_taxonomy,
    route_answer_taxonomy,
    save_answer_taxonomy,
)


class AnswerTaxonomyTests(unittest.TestCase):
    def setUp(self):
        self.doc_ids = ["d0", "d1", "d2", "d3", "d4", "d5"]
        self.doc_vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, 0.95, 0.0],
                [0.1, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
        self.query_ids = ["q0", "q1", "q2", "q3"]
        self.query_vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, 0.95, 0.0],
            ],
            dtype=np.float32,
        )
        self.qrels = {
            "q0": {"d0": 2, "d1": 1},
            "q1": {"d1": 2, "d2": 1},
            "q2": {"d3": 2, "d4": 1},
            "q3": {"d4": 2, "d5": 1},
        }

    def test_answer_supervision_routes_related_documents(self):
        tree, centroids = build_answer_supervised_taxonomy(
            self.doc_vectors,
            self.doc_ids,
            self.query_vectors,
            self.query_ids,
            self.qrels,
            top_branches=2,
            leaf_size=2,
        )
        allowed, meta = route_answer_taxonomy(
            [1.0, 0.0, 0.0], tree, centroids, route_leaves=1, min_items=2
        )
        self.assertTrue({"d0", "d1"}.intersection(allowed))
        self.assertGreaterEqual(meta["supervised_query_count"], 4)
        self.assertEqual(tree["item_count"], 6)

    def test_cache_round_trip(self):
        tree, centroids = build_answer_supervised_taxonomy(
            self.doc_vectors,
            self.doc_ids,
            self.query_vectors,
            self.query_ids,
            self.qrels,
            top_branches=2,
            leaf_size=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "answer-taxonomy"
            save_answer_taxonomy(prefix, tree, centroids)
            loaded = load_answer_taxonomy(prefix)
            self.assertIsNotNone(loaded)
            loaded_tree, loaded_centroids = loaded
            self.assertEqual(loaded_tree["assignments"], tree["assignments"])
            np.testing.assert_allclose(loaded_centroids, centroids)

    def test_expansion_keeps_global_fallback(self):
        global_results = [{"path": "a"}, {"path": "b"}]
        routed_results = [{"path": "b"}, {"path": "c"}]
        expanded = answer_taxonomy_candidate_expansion(
            routed_results, global_results, extra_candidates=1
        )
        self.assertEqual([item["path"] for item in expanded], ["a", "b", "c"])
        self.assertTrue(expanded[-1]["answer_taxonomy_routed"])

    def test_soft_rank_fusion_keeps_global_limit(self):
        global_results = [{"path": "a"}, {"path": "b"}, {"path": "c"}]
        routed_results = [{"path": "c"}, {"path": "d"}]
        fused = answer_taxonomy_rank_fusion(
            routed_results, global_results, limit=3, route_weight=0.5
        )
        self.assertEqual(len(fused), 3)
        self.assertEqual({item["path"] for item in fused}, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
