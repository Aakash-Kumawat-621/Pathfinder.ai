# Round 4 — Critical evaluator fixes

## 1. Production ANN retrieval
`src/semantic_engine.py` now provides a production `FaissANNIndex` backed by **FAISS HNSW** over normalized OpenAI catalog embeddings. The index is built once and cached by catalog fingerprint at Streamlit startup. Recommendation queries retrieve only a bounded top-K candidate set instead of requesting a score for every course.

A dependency-free random-hyperplane LSH implementation remains as an offline fallback when `faiss-cpu` is unavailable.

## 2. Explicit prerequisite DAG solver
`src/planner.py` contains `PrerequisiteDAGSolver`. It validates unknown dependencies, detects cycles, computes transitive prerequisite closure, and performs deterministic Kahn topological traversal. Personalization priority is only used to break ties among currently-ready nodes.

`src/recommender.py` delegates its compatibility `DeterministicDAG` facade to this solver, so the actual curriculum generation is graph-first.

## 3. Interactive skill-gap roadmap
`app.py` renders a Plotly node-edge roadmap with three semantic layers:
- acquired learner skills and proficiency evidence,
- sequential course milestones connected by prerequisite edges,
- outstanding target-role skill requirements.

The graph is interactive (hover, zoom, pan) and exposes milestone status, delivered skill, duration, and prerequisites.
