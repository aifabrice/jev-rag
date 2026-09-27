import tempfile
import unittest
from pathlib import Path

import numpy as np

from taxonomy import (
    build_taxonomy,
    load_taxonomy,
    route_taxonomy,
    save_taxonomy,
    taxonomy_candidate_expansion,
    taxonomy_rank_fusion,
)


class TaxonomyTests(unittest.TestCase):
    def setUp(self):
        self.vectors = np.asarray(
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
        self.ids = [f"doc-{index}" for index in range(6)]
        self.texts = [
            "cardiology heart treatment",
            "cardiology cardiac diagnosis",
            "heart medicine guideline",
            "nutrition vitamin diet",
            "nutrition food guideline",
            "vitamin diet supplement",
        ]

    def test_tree_assigns_every_item_and_routes_related_cluster(self):
        tree, centroids = build_taxonomy(
            self.vectors, self.ids, self.texts, top_branches=2, leaf_size=2, seed=3
        )
        self.assertEqual(set(tree["assignments"]), set(self.ids))
        self.assertEqual(tree["top_node_count"], 2)
        self.assertGreaterEqual(tree["leaf_node_count"], 2)
        self.assertTrue(all(memberships for memberships in tree["assignments"].values()))

        allowed, meta = route_taxonomy(
            [1.0, 0.0, 0.0], tree, centroids, route_leaves=1, min_items=2
        )
        self.assertTrue({"doc-0", "doc-1"}.intersection(allowed))
        self.assertGreaterEqual(meta["scope_items"], 2)
        self.assertTrue(meta["routes"][0]["label"])

    def test_cache_round_trip(self):
        tree, centroids = build_taxonomy(
            self.vectors, self.ids, self.texts, top_branches=2, leaf_size=2, seed=4
        )
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "taxonomy"
            save_taxonomy(prefix, tree, centroids)
            loaded = load_taxonomy(prefix)
            self.assertIsNotNone(loaded)
            loaded_tree, loaded_centroids = loaded
            self.assertEqual(loaded_tree["assignments"], tree["assignments"])
            np.testing.assert_allclose(loaded_centroids, centroids)

    def test_rank_fusion_keeps_global_fallback(self):
        routed = [{"rowid": 2}, {"rowid": 3}]
        global_results = [{"rowid": 1}, {"rowid": 2}, {"rowid": 4}]
        fused = taxonomy_rank_fusion(routed, global_results, limit=4)
        self.assertEqual(fused[0]["rowid"], 2)
        self.assertEqual({item["rowid"] for item in fused}, {1, 2, 3, 4})
        self.assertEqual(fused[0]["taxonomy_rank"], 1)
        self.assertEqual(fused[0]["global_rank"], 2)

    def test_candidate_expansion_preserves_global_order_and_adds_route_misses(self):
        global_results = [{"rowid": 1}, {"rowid": 2}, {"rowid": 3}]
        routed = [{"rowid": 2}, {"rowid": 4}, {"rowid": 5}]
        expanded = taxonomy_candidate_expansion(
            routed, global_results, extra_candidates=1
        )
        self.assertEqual([item["rowid"] for item in expanded], [1, 2, 3, 4])
        self.assertEqual(expanded[1]["taxonomy_rank"], 1)
        self.assertIsNone(expanded[-1]["global_rank"])


if __name__ == "__main__":
    unittest.main()
