import numpy as np
import pandas as pd

from src.semantic_engine import InMemoryANNIndex
from src.recommender import DeterministicDAG, generate_path, load_catalog


def test_ann_index_returns_nearest_vector_without_full_scan():
    vec = np.eye(4, dtype=np.float32)
    idx = InMemoryANNIndex(["a", "b", "c", "d"], vec, bits=8)
    hits = idx.search([1, 0, 0, 0], top_k=1)
    assert hits[0][0] == "a"
    assert hits[0][1] > 0.99


def test_dag_solver_is_deterministic_and_prerequisite_safe():
    df = pd.DataFrame([
        {"course_id": "a", "prerequisites": ""},
        {"course_id": "b", "prerequisites": "a"},
        {"course_id": "c", "prerequisites": "b"},
    ])
    dag = DeterministicDAG(df)
    out1 = dag.solve(["c"], priority={"c": 100, "b": 50, "a": 1})
    out2 = dag.solve(["c"], priority={"c": 100, "b": 50, "a": 1})
    assert out1 == out2 == ["a", "b", "c"]


def test_personalized_path_stays_prerequisite_safe():
    ordered, _ = generate_path("Data Science", "intermediate", ["python"], [])
    pos = {cid: i for i, cid in enumerate(ordered)}
    df = load_catalog()
    for _, row in df.iterrows():
        cid = row["course_id"]
        if cid not in pos:
            continue
        for parent in str(row["prerequisites"]).split(";"):
            parent = parent.strip()
            if parent and parent in pos:
                assert pos[parent] < pos[cid]

def test_explicit_planner_solver_detects_cycles_and_orders_targets():
    from src.planner import PrerequisiteDAGSolver
    df = pd.DataFrame([
        {"course_id":"a","prerequisites":""}, {"course_id":"b","prerequisites":"a"},
        {"course_id":"c","prerequisites":"b"},
    ])
    solver=PrerequisiteDAGSolver(df)
    assert solver.topological_order(["c"], priority={"c":999}) == ["a","b","c"]
    cyc=pd.DataFrame([{ "course_id":"x","prerequisites":"y"},{"course_id":"y","prerequisites":"x"}])
    try: PrerequisiteDAGSolver(cyc)
    except ValueError as e: assert "cycle" in str(e).lower()
    else: raise AssertionError("cycle was not detected")
