"""
test_recommender.py — the centerpiece test: proves the graph-based path
generator's core promise (valid prerequisite ordering) holds across a wide
sweep of synthetic learner profiles, not just one happy-path example.
"""
import itertools
import pytest
from src import recommender as rec
from src import validators as val

DOMAINS = ["Data Science", "Web Development", "Cloud & DevOps",
           "Cybersecurity", "Product Management", "Digital Marketing"]
LEVELS = ["beginner", "intermediate", "advanced"]

KNOWN_SKILLS_VARIANTS = [
    [],
    ["python"],
    ["javascript", "html_basics"],
    ["python", "statistics", "data_wrangling"],
]

COMPLETED_VARIANTS = [
    [],
    ["py101"],
    ["html101", "js101"],
]


@pytest.fixture(scope="module")
def catalog_df():
    return rec.load_catalog()


def test_catalog_loads_and_has_expected_columns(catalog_df):
    expected_cols = {"course_id", "title", "domain", "skill_tag", "level",
                      "prerequisites", "duration_hours", "type", "description", "resource_url"}
    assert expected_cols.issubset(set(catalog_df.columns))
    assert len(catalog_df) > 0
    assert catalog_df["course_id"].is_unique


def test_catalog_prerequisites_reference_real_courses(catalog_df):
    """Every prerequisite listed must actually exist as a course_id — catches typos in the catalog."""
    valid_ids = set(catalog_df["course_id"])
    for _, row in catalog_df.iterrows():
        prereqs = [p.strip() for p in str(row["prerequisites"]).split(";") if p.strip()]
        for p in prereqs:
            assert p in valid_ids, f"'{row['course_id']}' lists unknown prerequisite '{p}'"


@pytest.mark.parametrize(
    "domain,level,known_skills,completed",
    list(itertools.product(DOMAINS, LEVELS, KNOWN_SKILLS_VARIANTS, COMPLETED_VARIANTS))
)
def test_generated_path_is_always_prerequisite_valid(catalog_df, domain, level, known_skills, completed):
    """
    Sweeps 6 domains x 3 levels x 4 skill-variants x 3 completed-variants = 216
    synthetic learner profiles. For every single one, the generated path must
    never place a course before its own prerequisite.
    """
    ordered_ids, id_to_row = rec.generate_path(
        target_domain=domain, experience_level=level,
        known_skills=known_skills, completed_courses=completed,
    )
    # should always produce a non-empty path for a real domain
    assert len(ordered_ids) > 0, f"Empty path for domain={domain}, level={level}"

    # the actual validity guarantee — will raise PathValidationError on violation
    val.validate_prerequisite_order(ordered_ids, catalog_df, completed_courses=completed)


def test_generated_path_has_no_duplicates():
    ordered_ids, _ = rec.generate_path("Data Science", "beginner", [], [])
    assert len(ordered_ids) == len(set(ordered_ids))


def test_beginner_python_course_appears_before_ml(catalog_df):
    ordered_ids, _ = rec.generate_path("Data Science", "beginner", [], [])
    assert "py101" in ordered_ids and "ml101" in ordered_ids
    assert ordered_ids.index("py101") < ordered_ids.index("ml101")


def test_known_skill_trims_matching_beginner_course():
    """An intermediate+ learner who already knows python shouldn't be told to take Python 101 again."""
    ordered_ids, _ = rec.generate_path("Data Science", "intermediate", ["python"], [])
    assert "py101" not in ordered_ids


def test_completed_courses_are_excluded_from_path():
    ordered_ids, _ = rec.generate_path("Data Science", "beginner", [], ["py101"])
    assert "py101" not in ordered_ids


def test_milestones_assigned_for_every_step():
    ordered_ids, id_to_row = rec.generate_path("Web Development", "beginner", [], [])
    milestones = rec.assign_milestones(ordered_ids, id_to_row)
    assert len(milestones) == len(ordered_ids)
    assert all(m in {"Foundations", "Core Skills", "Applied / Advanced", "Capstone"} for m in milestones)


def test_milestone_ordering_has_no_glaring_violations():
    ordered_ids, id_to_row = rec.generate_path("Cybersecurity", "beginner", [], [])
    milestones = rec.assign_milestones(ordered_ids, id_to_row)
    warnings = val.validate_no_duplicate_milestones_out_of_order(milestones)
    assert warnings == [], f"Unexpected milestone ordering issues: {warnings}"


def test_path_progress_stats_math():
    path_items = [
        {"course_id": "a", "status": "completed"},
        {"course_id": "b", "status": "completed"},
        {"course_id": "c", "status": "in_progress"},
        {"course_id": "d", "status": "not_started"},
    ]
    stats = rec.path_progress_stats(path_items)
    assert stats["total"] == 4
    assert stats["completed"] == 2
    assert stats["pct_complete"] == 50.0


def test_unknown_domain_falls_back_gracefully(catalog_df):
    """generate_path should never crash on a domain name it doesn't recognize."""
    ordered_ids, _ = rec.generate_path("Underwater Basket Weaving", "beginner", [], [])
    assert isinstance(ordered_ids, list)  # falls back to full catalog rather than erroring


def test_feedback_reorder_preserves_prerequisite_order(catalog_df):
    learner = {"target_domain": "Data Science", "experience_level": "beginner", "known_skills": [], "completed_courses": [], "interests": []}
    ordered, id_to_row = rec.generate_path("Data Science", "beginner", [], [])
    items = [{"course_id": cid, "status": "not_started", "step_order": i + 1, "milestone": "", "explanation": ""} for i, cid in enumerate(ordered)]
    new_order = rec.adjust_path_on_feedback(learner, items, "py102", "down")
    val.validate_prerequisite_order(new_order, catalog_df)


def test_catalog_validation_passes(catalog_df):
    assert val.validate_catalog(catalog_df) is True
