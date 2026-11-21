from src import assessment
from src import recommender as rec


def test_each_domain_has_diagnostic_questions():
    for domain in rec.load_catalog()["domain"].unique():
        assert len(assessment.questions(domain)) >= 3


def test_diagnostic_grading_is_bounded_and_skill_specific():
    qs=assessment.questions("Data Science")
    r=assessment.grade("Data Science", {str(i): q["answer"] for i,q in enumerate(qs)})
    assert r["pct"] == 100 and all(v == 3 for v in r["skill_scores"].values())


def test_partial_diagnostic_creates_evidence():
    qs=assessment.questions("Web Development")
    r=assessment.grade("Web Development", {"0": qs[0]["answer"], "1": 99})
    assert r["score"] == 1 and r["skill_scores"][qs[0]["skill"]] == 3
