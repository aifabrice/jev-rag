from learning_to_rank import PairwiseLinearRanker, build_feature_vectors, make_feature_names


def test_rank_features_and_pairwise_training_prefer_relevant_document():
    sources = {
        "lexical": {"q1": ["d2", "d1", "d3"]},
        "semantic": {"q1": ["d1", "d3", "d2"]},
    }
    candidates = {"q1": ["d1", "d2", "d3"]}
    features = build_feature_vectors(sources, candidates)
    names = make_feature_names(sources)
    ranker = PairwiseLinearRanker.fit(
        features,
        {"q1": {"d1": 2}},
        names,
        epochs=25,
        max_negatives_per_positive=2,
    )
    assert ranker.rank(features["q1"])[0] == "d1"


def test_missing_source_documents_get_zero_presence_features():
    sources = {"lexical": {"q1": ["d1"]}, "semantic": {"q1": ["d2"]}}
    features = build_feature_vectors(sources, {"q1": ["d1", "d2", "d3"]})["q1"]
    assert features["d3"][2] == 0.0
    assert features["d3"][5] == 0.0
