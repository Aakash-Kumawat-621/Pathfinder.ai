from pathlib import Path
import pandas as pd

from src.planner import CostWeightedAStar
from src import recommender

ROOT = Path(__file__).resolve().parents[1]


def test_cost_weighted_astar_respects_prerequisites_and_baseline_costs():
    df = pd.DataFrame([
        {"course_id":"a", "skill_tag":"python", "duration_hours":8, "prerequisites":"", "domain":"Data Science", "level":"beginner", "type":"course"},
        {"course_id":"b", "skill_tag":"stats", "duration_hours":4, "prerequisites":"a", "domain":"Data Science", "level":"beginner", "type":"course"},
        {"course_id":"c", "skill_tag":"ml", "duration_hours":10, "prerequisites":"b", "domain":"Data Science", "level":"intermediate", "type":"course"},
    ])
    p = CostWeightedAStar(df).search(["c"], baseline={"python":3})
    assert p == ["a", "b", "c"]
    assert p.index("a") < p.index("b") < p.index("c")
    assert CostWeightedAStar(df)._cost("a", {"python":3}) < CostWeightedAStar(df)._cost("a", {"python":0})


def test_generate_path_accepts_assessed_skill_scores():
    ids, rows = recommender.generate_path(
        "Data Science", "intermediate", ["python"], [], [], "data science", "moderate", [],
        assessed_skill_scores={"python": 3, "statistics": 2}
    )
    assert ids
    assert len(ids) == len(set(ids))


def test_versioned_prompt_templates_are_externalized():
    prompt_dir = ROOT / "config" / "prompts"
    for name in ("profiling.yaml", "advisor.yaml", "chat.yaml"):
        text = (prompt_dir / name).read_text(encoding="utf-8")
        assert "version:" in text
        assert "system:" in text


def test_app_contains_hierarchical_clustering_and_collapse_control():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "AgglomerativeClustering" in text
    assert "Collapse dense prerequisite clusters" in text
    assert "Expand clusters" in text
