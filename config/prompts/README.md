# Versioned Prompt Templates

PathFinder keeps LLM system prompts outside Python code so prompt changes do not require application edits.

- `profiling.yaml` — learner profile extraction
- `advisor.yaml` — course-placement explanations
- `chat.yaml` — grounded advisor conversation

Each file carries a `version` field. `src/llm_engine.py::load_prompt()` validates and loads these templates at runtime with a safe offline fallback.
