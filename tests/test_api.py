from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


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


def test_progress_percentage():
    payload = {
        "recommended_path": [
            {
                "step": 1,
                "course": "Course A",
                "similarity_score": 0.9,
                "rationale": "why",
                "milestone": "Foundations",
                "status": "complete",
            },
            {
                "step": 2,
                "course": "Course B",
                "similarity_score": 0.8,
                "rationale": "why",
                "milestone": "Core Skills",
                "status": "not_started",
            },
        ]
    }
    response = client.post("/progress", json=payload)
    assert response.status_code == 200
    assert response.json()["progress_pct"] == 50.0
