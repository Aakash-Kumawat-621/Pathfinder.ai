"""
test_agents.py — orchestrator end-to-end tests. Runs in rule-based/TF-IDF
fallback mode since no OPENAI_API_KEY is set in the test environment,
which also validates that the whole agent pipeline works without a live
API key (the resilience property claimed in the README).
"""
from src.agents import Orchestrator
from src import validators as val
from src import recommender as rec


def test_onboarding_produces_valid_profile():
    orch = Orchestrator()
    profile, missing, trace = orch.run_onboarding(
        "I want to become a data scientist, I know basic python", known_skills_hint=[]
    )
    assert profile["target_domain"] == "Data Science"
    assert profile["experience_level"] in ("beginner", "intermediate", "advanced")
    assert isinstance(missing, list)
    assert len(trace) >= 1
    assert "ProfilerAgent" in trace[0]


def test_onboarding_flags_missing_fields():
    orch = Orchestrator()
    _, missing, _ = orch.run_onboarding("I want to learn web development", known_skills_hint=[])
    # no skills/time given -> should be flagged as low-confidence
    assert "known_skills" in missing


def test_path_generation_produces_valid_prerequisite_order():
    orch = Orchestrator()
    learner = {
        "goal_text": "I want to become a data scientist",
        "target_domain": "Data Science", "experience_level": "beginner",
        "known_skills": [], "completed_courses": [], "interests": ["Data Science"],
    }
    path_items, related, trace = orch.run_path_generation(learner)
    assert len(path_items) > 0
    ordered_ids = [p["course_id"] for p in path_items]
    df = rec.load_catalog()
    val.validate_prerequisite_order(ordered_ids, df)  # raises on violation

    # every step should have a non-empty explanation (even in fallback mode)
    assert all(p["explanation"] for p in path_items)
    # trace should mention every agent that ran
    trace_text = " ".join(trace)
    assert "RecommenderAgent" in trace_text
    assert "AdvisorAgent" in trace_text


def test_path_generation_step_orders_are_sequential():
    orch = Orchestrator()
    learner = {
        "goal_text": "I want to learn cloud computing and devops",
        "target_domain": "Cloud & DevOps", "experience_level": "beginner",
        "known_skills": [], "completed_courses": [], "interests": [],
    }
    path_items, _, _ = orch.run_path_generation(learner)
    step_orders = [p["step_order"] for p in path_items]
    assert step_orders == list(range(1, len(path_items) + 1))
