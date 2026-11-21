# PathFinder Ultimate — judging evidence

## What makes this version stronger

1. **Hybrid recommendation score**: goal semantic relevance, skill-gap priority, domain fit, difficulty fit, interest match, time fit and feedback fit.
2. **Prerequisite-constrained ranking**: personalization only chooses among dependency-safe nodes; Kahn topological ordering remains the structural guarantee.
3. **Explainability**: each course exposes a score breakdown and grounded natural-language rationale.
4. **Time-aware planning**: the learner's weekly capacity is converted into a week-by-week schedule; completed resources are omitted automatically.
5. **Adaptive preference memory**: thumbs-up/down and free-text feedback produce interpretable preference signals for projects, duration and course/skill affinity.
6. **AI + offline resilience**: OpenAI embeddings/LLM when configured, TF-IDF and deterministic fallbacks otherwise.
7. **Progress-safe adaptation**: completed/in-progress resources remain fixed while only the unfinished suffix is re-ranked.
8. **Quality gates**: catalog validation, prerequisite validation, regression tests and offline scenario evaluation.

## Adaptive diagnostic evidence

The Skill Check asks domain-specific questions tied directly to catalog skills. Results are persisted per learner and used as additional evidence in role-readiness calculations and optional roadmap re-personalization. This makes personalization evidence-based rather than purely self-reported.

## Judge demo proof

- Enter the same goal with two different time budgets and show different weekly schedules.
- Add an interest such as hands-on projects and show the project preference signal.
- Mark a prerequisite complete and show it disappear from remaining work.
- Down-vote a resource and regenerate; verify completed/in-progress status is preserved and prerequisites remain valid.
- Open “Recommendation score breakdown” to show transparent ranking factors.
