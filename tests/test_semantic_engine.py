"""test_semantic_engine.py — semantic matching correctness (TF-IDF fallback path,
since no OPENAI_API_KEY is set in the test environment)."""
from src import semantic_engine as sem
from src import recommender as rec


def test_backend_name_is_valid():
    assert sem.backend_name() in ("openai_embeddings", "tfidf_vector_space")


def test_similarity_scores_in_valid_range():
    df = rec.load_catalog()
    scores = sem.semantic_similarity_scores("I want to learn python and machine learning", df)
    assert len(scores) == len(df)
    for cid, s in scores.items():
        assert -1e-6 <= s <= 1 + 1e-6  # cosine similarity of non-negative TF-IDF vectors is in [0,1]


def test_domain_scores_cover_all_domains():
    df = rec.load_catalog()
    scores = sem.semantic_domain_scores("I want to build websites with react", df)
    assert set(scores.keys()) == set(df["domain"].unique())


def test_best_matching_domain_picks_a_real_domain():
    df = rec.load_catalog()
    domain, scores = sem.best_matching_domain("I want to become a penetration tester and ethical hacker", df)
    assert domain in df["domain"].unique()
    assert domain == "Cybersecurity"  # this query should unambiguously match cybersecurity


def test_related_courses_excludes_given_ids():
    df = rec.load_catalog()
    exclude = ["ml101", "py101"]
    related = sem.related_courses("I want to learn machine learning", df, exclude_ids=exclude, top_k=5)
    related_ids = [cid for cid, _ in related]
    assert "ml101" not in related_ids
    assert "py101" not in related_ids


def test_related_courses_returns_nothing_for_irrelevant_query():
    """A goal totally disconnected from any domain shouldn't force weak matches into the results."""
    df = rec.load_catalog()
    related = sem.related_courses("asdkjalksjd qqqzzz nonsense gibberish", df, exclude_ids=[], top_k=5)
    assert related == [] or all(score < 0.3 for _, score in related)

def test_faiss_ann_class_is_available_as_optional_production_backend():
    import src.semantic_engine as sem
    assert hasattr(sem, "FaissANNIndex")
    assert "FAISS" in sem.FaissANNIndex.__doc__
