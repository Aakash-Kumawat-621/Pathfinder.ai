# Pathify.ai — PathFinder Backend

FastAPI backend for the PathFinder personalized learning-path recommender.

## What is implemented

- `POST /chat`: parses free-text learner goal into structured profile fields.
- `POST /profile`: maintains learner profile contract.
- `POST /recommend`: wraps a two-stage recommender (NB track prediction + cosine ranking within track).
- `POST /path`: returns ordered roadmap items.
- `POST /explain`: generates short recommendation rationales.
- `POST /progress`: computes completion progress and next actions.
- `GET /health`: healthcheck endpoint.

## Learner profile contract

```json
{
  "learner_id": "uuid",
  "goal_statement": "I want to become a cloud engineer",
  "skill_level": "beginner",
  "interests": ["cloud"],
  "completed_courses": ["AWS Cloud Practitioner"],
  "predicted_track": "cloud",
  "recommended_path": [
    {
      "step": 1,
      "course": "Cloud Foundations",
      "similarity_score": 0.83,
      "rationale": "Matches your goal using cloud concepts found in your query.",
      "milestone": "Foundations",
      "status": "not_started"
    }
  ],
  "progress_pct": 0
}
```

## Model artifact loading

At startup, the API tries loading frozen artifacts from `PATHIFY_ARTIFACTS_DIR`:

- `vectorizer.joblib`
- `classifier.joblib`
- `courses.joblib`
- `course_matrix.joblib`

If not present, it boots with a small built-in fallback dataset while preserving the same two-stage logic.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Run tests

```bash
pytest -q
```
