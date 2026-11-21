# Round 6 Critical Fixes

## 1. Cost-weighted A* planner
`src/planner.py` now contains `CostWeightedAStar`. It expands only currently-ready prerequisite nodes and scores each candidate with:

`f(n) = g(n) + h(n) - preference_nudge`

`g(n)` combines course study hours, prerequisite friction, and a baseline-proficiency discount. `h(n)` is a lower-bound remaining workload estimate. The DAG remains the hard safety boundary, so personalization cannot violate prerequisites.

## 2. External versioned prompts
LLM prompt templates live under `config/prompts/*.yaml` and are loaded through `src/llm_engine.py::load_prompt()`. Prompt versions can be changed without editing Python orchestration code.

## 3. Hierarchical DAG collapsing
`app.py` uses `AgglomerativeClustering` to group dense milestone nodes. Judges can toggle clustering and expand selected clusters while acquired-skill and target-requirement nodes remain visible.
