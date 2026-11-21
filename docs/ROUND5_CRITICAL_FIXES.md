# Round 5 — Critical Evaluator Fixes

## 1. Build-time serialized ANN artifacts
`data/generate_catalog.py` now performs a build-time semantic-index step. It serializes:
- `data/catalog_embeddings.npz` — precomputed catalog vectors
- `data/catalog_ann_lsh.npz` — serialized random-hyperplane ANN structure when FAISS is unavailable
- `data/catalog_index.faiss` — FAISS HNSW index when `faiss-cpu` is available at build time
- `data/catalog_tfidf_vectorizer.joblib` — offline query encoder
- `data/catalog_index_meta.json` — catalog fingerprint/version/backend metadata

`src/semantic_engine.py` loads these artifacts before attempting any cold-start catalog embedding generation. Stale artifacts are rejected using a catalog content fingerprint.

## 2. Live multi-agent execution status
`app.py` wraps onboarding and adaptive roadmap generation in `st.status` and passes a progress callback through the orchestrator. The UI visibly streams semantic retrieval, dependency planning, optional retrieval, and advisor explanation steps during a live demo.

## 3. Pydantic-validated database query models
`database.py` defines typed Pydantic records for learner, path, feedback, chat, and assessment rows. All query helpers validate database rows through `model_validate()` before returning them. The models retain dict-compatible `[]`, `.get()`, `.items()`, and iteration access so planner/recommender integration remains backward-compatible.

## Verification
274 automated tests pass, including dedicated tests for serialized index artifacts, streaming pipeline support, and typed database query results.
