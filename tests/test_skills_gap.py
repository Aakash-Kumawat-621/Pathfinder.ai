"""test_skills_gap.py — skill-gap computation correctness."""
from src import skills_gap as sg
from src import recommender as rec


def _id_to_row():
    df = rec.load_catalog()
    return {row["course_id"]: row for _, row in df.iterrows()}


def test_gap_returns_empty_for_unknown_domain():
    learner = {"target_domain": "Underwater Basket Weaving", "known_skills": [], "experience_level": "beginner"}
    gap = sg.compute_skill_gap(learner, [], _id_to_row())
    assert gap == {}


def test_gap_reflects_known_skills():
    learner = {"target_domain": "Data Science", "known_skills": ["python"], "experience_level": "intermediate"}
    gap = sg.compute_skill_gap(learner, [], _id_to_row())
    assert gap["python"]["current"] >= 2  # intermediate self-report -> score 2
    assert gap["statistics"]["current"] == 0  # not claimed, no completed courses


def test_gap_reflects_completed_courses():
    id_to_row = _id_to_row()
    learner = {"target_domain": "Data Science", "known_skills": [], "experience_level": "beginner"}
    path_items = [{"course_id": "ml101", "status": "completed"}]
    gap = sg.compute_skill_gap(learner, path_items, id_to_row)
    # ml101 -> skill_tag 'machine_learning', level 'intermediate' -> score 2
    assert gap["machine_learning"]["current"] == 2


def test_gap_summary_bounds():
    learner = {"target_domain": "Web Development", "known_skills": [], "experience_level": "beginner"}
    gap = sg.compute_skill_gap(learner, [], _id_to_row())
    summary = sg.gap_summary(gap)
    assert 0 <= summary["readiness_pct"] <= 100
    assert summary["biggest_gap_skill"] in gap


def test_gap_summary_empty_input():
    summary = sg.gap_summary({})
    assert summary["readiness_pct"] == 0
    assert summary["biggest_gap_skill"] is None


def test_gap_reflects_prior_completed_course():
    learner = {"target_domain": "Data Science", "known_skills": [], "experience_level": "beginner", "completed_courses": ["py101"]}
    gap = sg.compute_skill_gap(learner, [], _id_to_row())
    assert gap["python"]["current"] == 1
