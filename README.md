# 🧭 PathFinder — AI-Powered Personalized Learning Path Recommender

**HCLTech AI Amplified Hackathon · Round 2**

PathFinder is a conversational learning assistant that turns a learner’s natural-language goal into a **personalized, prerequisite-safe, milestone-based roadmap**. It combines AI/semantic retrieval with a deterministic prerequisite graph so recommendations are both adaptive and explainable.

## Why PathFinder is different

A generic chatbot can produce a list of courses, but it can hallucinate resources and ignore prerequisite order. PathFinder separates responsibilities:

**Natural language → Learner profile → Semantic relevance → Personalization → Dependency-safe path → AI explanation → Progress/feedback → Adaptation**

The LLM is used where language understanding helps. The curriculum structure is controlled by deterministic algorithms and validated before display.

## Core features

- 💬 **Conversational onboarding** — learner describes their goal naturally.
- 👤 **Learner profiling** — target domain, experience level, known skills, prior completed courses, interests and weekly study pace.
- 🧠 **AI/semantic understanding** — OpenAI structured extraction + embeddings when an API key is available; TF-IDF semantic retrieval when offline.
- 🗺️ **Personalized path generation** — semantic relevance + level fit + interests + time/pace fit + feedback signals.
- 🔗 **Prerequisite-safe curriculum** — graph closure + topological ordering; runtime validator blocks invalid paths.
- 📍 **Milestones** — Foundations → Core Skills → Applied/Advanced → Capstone.
- 💡 **Grounded explanations** — every path step has a “Why this is here” explanation.
- 🔄 **Adaptive feedback loop** — thumbs-up/down changes future ranking while preserving dependencies.
- 📊 **Skill intelligence dashboard** — progress, skill coverage, role-readiness radar, biggest gap and next-best action.
- 🏅 **Gamification** — completion, consistency and feedback badges.
- ✨ **Semantic electives** — optional related resources outside the core path.
- 🧪 **Adaptive Skill Check** — difficulty-adaptive diagnostic questions select the next unseen item from current ability evidence and persist proficiency signals.
- 🔄 **One-click re-personalization** — regenerate the roadmap from diagnostic evidence, progress and feedback while preserving dependency safety.
- ⬇️ **Roadmap export** — learner can export the current plan as CSV.
- 🛡️ **Offline-safe demo** — no API key is required for the core recommendation workflow.

## Architecture

```text
                         ┌──────────────────────────────┐
                         │        Streamlit UI           │
                         │ Onboarding · Path · Dashboard │
                         │          · Ask AI             │
                         └──────────────┬───────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   Orchestrator     │
                              └──────┬──────┬──────┘
                                     │      │
                   ┌─────────────────┘      └──────────────────┐
                   ▼                                            ▼
          ┌────────────────┐                         ┌──────────────────┐
          │ ProfilerAgent  │                         │  AdvisorAgent    │
          │ LLM + fallback │                         │ grounded LLM/Q&A │
          └───────┬────────┘                         └────────┬─────────┘
                  │                                           │
                  ▼                                           │
          ┌──────────────────┐                                 │
          │ RecommenderAgent │◄────────────────────────────────┘
          └───────┬──────────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
┌───────────────┐    ┌────────────────┐
│ Semantic      │    │ Prerequisite   │
│ retrieval     │    │ DAG + ranking  │
│ embeddings/   │    │ + validator    │
│ TF-IDF        │    └───────┬────────┘
└───────────────┘            │
                             ▼
                    ┌─────────────────┐
                    │ SQLite progress │
                    │ + feedback      │
                    └─────────────────┘
```

## Project structure

```text
pathfinder/
├── app.py
├── requirements.txt
├── Dockerfile
├── .env.example
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── data/
│   ├── courses.csv
│   └── generate_catalog.py
├── src/
│   ├── agents.py
│   ├── badges.py
│   ├── database.py
│   ├── llm_engine.py
│   ├── recommender.py
│   ├── semantic_engine.py
│   ├── skills_gap.py
│   └── validators.py
├── tests/                 # 258 automated tests
├── scripts/
│   └── validate_solution.py
└── docs/
    ├── SCORING_MATRIX.md
    ├── DEMO_SCRIPT.md
    └── ULTIMATE_FEATURES.md
    └── offline_evaluation.json
```

## AI/ML implementation

PathFinder uses a hybrid cold-start strategy: semantic retrieval handles sparse learner history, while an online logistic-feedback ranker activates after at least four mixed explicit interactions. This gives the prototype a genuine supervised personalization layer without pretending to have a large proprietary training corpus.

### 1. Structured learner understanding
`llm_engine.py` asks the LLM for a constrained JSON profile. The result is validated against allowed domains/levels/pace values. If the API is unavailable, deterministic extraction takes over.

### 2. Semantic retrieval
`semantic_engine.py` uses `text-embedding-3-small` when configured. The catalog embeddings are fingerprinted and cached. Without an API key, a TF-IDF + cosine-similarity vector-space model remains available, enriched with domain aliases.

### 3. Personalized ranking
`recommender.py` combines:

- semantic goal/course relevance;
- experience-level fit;
- known-skill avoidance for redundant foundations;
- interest match;
- time/pace fit;
- historical thumbs-up/down signals.

The ranking only decides **which ready node is preferred**. It cannot violate the prerequisite graph.

### 4. Safe curriculum generation
The engine computes the transitive prerequisite closure for the domain destination and performs a Kahn topological sort. `validators.py` then independently checks IDs, duplicates, prerequisite order and catalog cycles.

### 5. Grounded AI advisor
The advisor receives the actual learner profile and current path/progress. Prompts explicitly forbid inventing courses, prerequisites, progress or achievements.

## Running locally

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

### Optional OpenAI configuration

Copy `.env.example` to `.env` and/or set environment variables. The application also supports Streamlit secrets.

```text
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
```

**Do not commit a real API key.** The repository intentionally contains only placeholders.

### Docker

```bash
docker build -t pathfinder .
docker run --rm -p 8501:8501 pathfinder
```

## Verification

Run the complete automated suite:

```bash
pytest -q
```

The included suite covers database persistence, semantic retrieval, skill-gap calculations, multi-agent orchestration, badges and a broad prerequisite-order sweep. The regression suite covers the original recommender/database/semantic/agent behavior plus ultimate-feature tests. Run `pytest -q` to verify the exact count on the submission machine.

Run the pre-submission evaluator:

```bash
python scripts/validate_solution.py
```

It validates the catalog and checks generated paths across all six domains and three experience levels.

## Suggested demo

See `docs/DEMO_SCRIPT.md` for a 3–5 minute walkthrough designed around the judging criteria.

Recommended example:

> I want to become a data scientist. I know basic Python and I’m new to machine learning.

Then use:

> Python; completed py101

and:

> 5–10 hours a week

This demonstrates that prior knowledge affects the generated plan and that the estimated timeline adapts to study pace.

## Submission alignment

See `docs/SCORING_MATRIX.md` for a direct mapping from the implementation to the six judging dimensions: Problem Understanding & Solution Design, Functionality & Feature Completeness, AI/ML Implementation, Innovation & Creativity, User Experience & Interface, and Performance & Code Quality.

## Data and resource quality

The checked-in catalog contains 41 learning items across six domains and uses curated educational/first-party resource URLs instead of placeholder `example.com` links.

## Limitations and production roadmap

- Current catalog is a curated prototype rather than a live provider integration.
- Authentication and multi-user cloud persistence are outside the hackathon prototype scope.
- Feedback currently uses interpretable ranking nudges; a production system could learn ranking weights from larger interaction datasets.
- A production deployment could add provider APIs, richer assessments, spaced repetition and a learned knowledge-tracing model.

## Round 3 — Performance, Deterministic DAG & Interactive UX

- **In-memory ANN retrieval:** catalog embeddings are normalized and indexed at process startup with a FAISS HNSW index (with a dependency-free random-hyperplane LSH fallback). The index is warmed once at startup and queries retrieve only a bounded top-K candidate set rather than performing a full catalog scan.
- **Deterministic DAG solver:** `src/planner.py` owns the explicit `PrerequisiteDAGSolver`; `src/recommender.py` exposes a compatibility `DeterministicDAG` facade. The solver validates unknown dependencies/cycles, expands transitive dependencies, and performs deterministic Kahn-style traversal. Relevance scores only break ties among currently-ready nodes.
- **Interactive prerequisite map:** the Streamlit learning-path page renders the current path as an interactive Plotly node-edge roadmap with acquired learner skills, outstanding target requirements, sequential course milestones, hover details, zoom/pan, status, skills, duration, and prerequisite information.

These three layers are deliberately separated: ANN handles retrieval, the DAG guarantees curriculum correctness, and the graph visualization makes that correctness observable during a live judging demo.

## Round 6 Architecture Highlights
- **CostWeightedAStar** in `src/planner.py` minimizes effective study cost and prerequisite friction using assessed baseline proficiency while preserving DAG legality.
- **Versioned prompts** are externalized to `config/prompts/*.yaml` and loaded at runtime.
- **Hierarchical roadmap clustering** in `app.py` keeps dense prerequisite graphs presentation-ready with collapse/expand controls.
