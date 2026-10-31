from pseudo_relevance import select_expansion_terms, weighted_rrf_merge


def test_expansion_excludes_query_terms_and_prefers_specific_terms():
    terms = select_expansion_terms(
        "hearing loss",
        ["hearing loss sensorineural cochlear", "sensorineural cochlear implant"],
        document_count=100,
        document_frequency={"sensorineural": 2, "cochlear": 5, "implant": 20},
        max_terms=2,
    )
    assert "hearing" not in terms
    assert "loss" not in terms
    assert terms[0] == "sensorineural"


def test_weighted_rrf_merge_is_deterministic():
    assert weighted_rrf_merge([(["a", "b"], 1.0), (["b", "c"], 1.0)], limit=3) == ["b", "a", "c"]
