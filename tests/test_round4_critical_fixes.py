import json
import os

import pandas as pd


def test_pre_serialized_catalog_artifacts_exist():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    assert os.path.exists(os.path.join(base, "catalog_embeddings.npz"))
    assert os.path.exists(os.path.join(base, "catalog_index_meta.json"))
    assert os.path.exists(os.path.join(base, "catalog_tfidf_vectorizer.joblib"))
    assert os.path.exists(os.path.join(base, "catalog_ann_lsh.npz")) or os.path.exists(os.path.join(base, "catalog_index.faiss"))
    meta = json.load(open(os.path.join(base, "catalog_index_meta.json"), encoding="utf-8"))
    assert meta["version"] >= 2
    assert len(meta["ids"]) == len(pd.read_csv(os.path.join(base, "courses.csv")))


def test_streaming_pipeline_signature():
    from src.agents import Orchestrator
    import inspect
    assert "progress_callback" in inspect.signature(Orchestrator.run_path_generation).parameters


def test_database_query_helpers_return_pydantic_models(isolated_db):
    from src import database as db
    from pydantic import BaseModel
    learner_id = db.create_learner("A", "goal", "Data Science", "beginner", [], [], [], "moderate")
    db.add_chat_message(learner_id, "user", "hello")
    db.add_feedback(learner_id, "py101", "up", "good")
    db.save_learning_path(learner_id, [{"course_id":"py101","step_order":1,"milestone":"x","explanation":"y"}])
    assert isinstance(db.get_learner(learner_id), BaseModel)
    assert isinstance(db.get_chat_history(learner_id)[0], BaseModel)
    assert isinstance(db.get_feedback(learner_id)[0], BaseModel)
    assert isinstance(db.get_learning_path(learner_id)[0], BaseModel)
