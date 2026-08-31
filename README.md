# Pathify.ai — PathFinder

PathFinder is an AI-powered personalized learning-path recommender built around a two-stage ML core (TF-IDF + NB classification + cosine ranking) exposed via FastAPI.

## Implemented architecture

- Backend: FastAPI service
- ML core: Reused two-stage recommendation flow (`/recommend`)
- Conversational/profile parsing: `/chat` (LLM-ready with deterministic fallback)
- Learning-path sequencing: `/path` (milestone ordering)
- Explainability layer: `/explain` (LLM-ready with deterministic fallback)
- Dashboard data: `/progress`
- Frontend: lightweight live demo UI served at `/`

## API endpoints

- `GET /health`
- `POST /chat`
- `POST /profile`
- `POST /recommend`
- `POST /path`
- `POST /explain`
- `POST /progress`

## Learner profile contract (frozen)

```json
{
  "learner_id": "uuid",
  "goal_statement": "free text from chat",
  "skill_level": "beginner | intermediate | advanced",
  "interests": ["cloud", "machine learning"],
  "completed_courses": ["AWS Cloud Practitioner"],
  "predicted_track": "from NB classifier",
  "recommended_path": [
    {
      "step": 1,
      "course": "string",
      "similarity_score": 0.83,
      "rationale": "LLM-generated explanation",
      "milestone": "Foundations",
      "status": "not_started | in_progress | complete"
    }
  ],
  "progress_pct": 0
}
```

## Model artifact loading

Set `PATHIFY_ARTIFACTS_DIR` with the exported artifacts:

- `vectorizer.joblib`
- `classifier.joblib`
- `courses.joblib`
- `course_matrix.joblib`

If artifacts are absent, the service uses a built-in fallback dataset while preserving the same two-stage logic.

## Optional LLM wiring

Set these env vars to plug `/chat`, `/path`, `/explain` into your LLM gateway:

- `PATHIFY_LLM_URL`
- `PATHIFY_LLM_API_KEY` (optional)
- `PATHIFY_LLM_MODEL` (optional)
- `PATHIFY_LLM_TIMEOUT` (optional, seconds)

Without them, deterministic fallback behavior is used.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/` for the demo UI.

## Run tests

```bash
pytest -q
```
