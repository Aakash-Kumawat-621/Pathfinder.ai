from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def sample_path(status1="complete", status2="not_started"):
    return [
        {
            "step": 2,
            "course": "Advanced Cloud Project",
            "similarity_score": 0.8,
            "rationale": "why",
            "milestone": "Advanced Projects",
            "status": status2,
        },
        {
            "step": 1,
            "course": "Cloud Foundations",
            "similarity_score": 0.9,
            "rationale": "why",
            "milestone": "Foundations",
            "status": status1,
        },
    ]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_extracts_structure():
    response = client.post("/chat", json={"message": "I want to become a cloud engineer as a beginner"})
    data = response.json()
    assert response.status_code == 200
    assert data["skill_level"] == "beginner"
    assert "cloud" in data["interests"]


def test_recommend_returns_path():
    response = client.post("/recommend", json={"goal_statement": "I want to become a cloud engineer"})
    data = response.json()
    assert response.status_code == 200
    assert data["predicted_track"]
    assert len(data["recommended_path"]) > 0
    first = data["recommended_path"][0]
    assert set(first.keys()) == {"step", "course", "similarity_score", "rationale", "milestone", "status"}


def test_recommend_rejects_empty_goal():
    response = client.post("/recommend", json={"goal_statement": "  "})
    assert response.status_code == 400


def test_path_orders_and_resteps():
    response = client.post("/path", json={"recommended_path": sample_path()})
    data = response.json()
    assert response.status_code == 200
    assert data["recommended_path"][0]["step"] == 1
    assert data["recommended_path"][0]["course"] == "Cloud Foundations"


def test_explain_returns_rationale():
    response = client.post(
        "/explain",
        json={"goal_statement": "cloud engineer", "recommended_path": sample_path()},
    )
    data = response.json()
    assert response.status_code == 200
    assert "aligns with your goal" in data["recommended_path"][0]["rationale"].lower()


def test_progress_percentage():
    response = client.post("/progress", json={"recommended_path": sample_path()})
    assert response.status_code == 200
    assert response.json()["progress_pct"] == 50.0


def test_profile_contract_and_progress():
    payload = {
        "goal_statement": "cloud engineer",
        "skill_level": "beginner",
        "interests": ["cloud"],
        "completed_courses": [],
        "predicted_track": "cloud",
        "recommended_path": sample_path(),
    }
    response = client.post("/profile", json=payload)
    data = response.json()
    assert response.status_code == 200
    assert data["learner_id"]
    assert data["progress_pct"] == 50.0
