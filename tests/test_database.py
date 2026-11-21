"""test_database.py — CRUD correctness for the SQLite persistence layer."""


def test_create_and_get_learner(isolated_db):
    db = isolated_db
    learner_id = db.create_learner(
        name="Test Learner", goal_text="I want to learn data science",
        target_domain="Data Science", experience_level="beginner",
        interests=["Data Science"], known_skills=["python"],
        completed_courses=[], time_commitment="moderate",
    )
    learner = db.get_learner(learner_id)
    assert learner["name"] == "Test Learner"
    assert learner["target_domain"] == "Data Science"
    assert learner["known_skills"] == ["python"]  # JSON round-trips correctly


def test_get_nonexistent_learner_returns_none(isolated_db):
    assert isolated_db.get_learner(99999) is None


def test_save_and_get_learning_path(isolated_db):
    db = isolated_db
    learner_id = db.create_learner("A", "goal", "Data Science", "beginner", [], [], [], "moderate")
    path_items = [
        {"course_id": "py101", "step_order": 1, "milestone": "Foundations", "explanation": "start here"},
        {"course_id": "ml101", "step_order": 2, "milestone": "Core Skills", "explanation": "next step"},
    ]
    db.save_learning_path(learner_id, path_items)
    saved = db.get_learning_path(learner_id)
    assert len(saved) == 2
    assert saved[0]["course_id"] == "py101"
    assert saved[0]["status"] == "not_started"


def test_update_progress(isolated_db):
    db = isolated_db
    learner_id = db.create_learner("A", "goal", "Data Science", "beginner", [], [], [], "moderate")
    db.save_learning_path(learner_id, [
        {"course_id": "py101", "step_order": 1, "milestone": "Foundations", "explanation": ""}
    ])
    db.update_progress(learner_id, "py101", "completed")
    saved = db.get_learning_path(learner_id)
    assert saved[0]["status"] == "completed"


def test_feedback_roundtrip(isolated_db):
    db = isolated_db
    learner_id = db.create_learner("A", "goal", "Data Science", "beginner", [], [], [], "moderate")
    db.add_feedback(learner_id, "py101", "up", "loved it")
    fb = db.get_feedback(learner_id)
    assert len(fb) == 1
    assert fb[0]["rating"] == "up"
    assert fb[0]["comment"] == "loved it"


def test_chat_history_roundtrip(isolated_db):
    db = isolated_db
    learner_id = db.create_learner("A", "goal", "Data Science", "beginner", [], [], [], "moderate")
    db.add_chat_message(learner_id, "user", "hello")
    db.add_chat_message(learner_id, "assistant", "hi there")
    history = db.get_chat_history(learner_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_save_learning_path_overwrites_previous(isolated_db):
    """Re-saving a path (e.g. after feedback-driven reorder) should replace, not append."""
    db = isolated_db
    learner_id = db.create_learner("A", "goal", "Data Science", "beginner", [], [], [], "moderate")
    db.save_learning_path(learner_id, [
        {"course_id": "py101", "step_order": 1, "milestone": "Foundations", "explanation": ""}
    ])
    db.save_learning_path(learner_id, [
        {"course_id": "ml101", "step_order": 1, "milestone": "Core Skills", "explanation": ""}
    ])
    saved = db.get_learning_path(learner_id)
    assert len(saved) == 1
    assert saved[0]["course_id"] == "ml101"
