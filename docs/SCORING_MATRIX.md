# Round 2 judging alignment

PathFinder is intentionally designed around the six published judging dimensions.

| Criterion | Weight | Evidence in the prototype |
|---|---:|---|
| Problem Understanding & Solution Design | 20% | Conversational profile, goal/domain detection, skill-gap model, prerequisite DAG, milestone roadmap, pacing estimate, explainable agent trace |
| Functionality & Feature Completeness | 25% | Onboarding, learner profile, prior-course handling, recommendation engine, prerequisite path, progress tracking, feedback adaptation, adaptive skill diagnostic, one-click re-personalization, dashboard, AI advisor, CSV export |
| AI/ML Implementation | 20% | LLM structured profile extraction, OpenAI embeddings when configured, offline TF-IDF retrieval, semantic domain matching, grounded explanations, difficulty-adaptive diagnostic evidence, and an online logistic feedback ranker with deterministic cold-start fallback |
| Innovation & Creativity | 15% | Hybrid graph + semantic ranking, adaptive diagnostic evidence loop, role-readiness score, badges, optional semantic electives, explainable multi-agent workflow |
| User Experience & Interface | 10% | Conversational onboarding, visual progress metrics, milestone sections, expandable course cards, next-best-action card, clear AI status and export |
| Performance & Code Quality | 10% | Modular services, SQLite indexes/constraints, atomic path writes, dependency validation, cached TF-IDF/embedding data, 282 automated tests, offline-safe execution |

## Demo proof points

1. Start with a natural-language goal rather than a form-heavy questionnaire.
2. Show that known skills and completed courses remove redundant learning.
3. Show the prerequisite-safe path and the reason for each recommendation.
4. Mark a step complete and show progress/readiness change.
5. Give a thumbs-down and demonstrate adaptive reordering **without breaking prerequisites**.
6. Ask the advisor why a prerequisite exists.
7. Show the dashboard and export the roadmap.

## Final verification

- `pytest -q`: **282/282 passed**
- Offline validator: **18/18 scenarios valid**
- Prerequisite validity: **100%**
- Diagnostic coverage: all 6 supported domains


## Round 7 AI-depth upgrades

- Online supervised feedback ranker (`LogisticRegression`) activates after 4+ mixed learner ratings.
- Recommendation score breakdown now uses the correct skill-level feedback signal.
- Adaptive diagnostic selects the next unseen question near estimated learner ability and records bounded proficiency evidence.
- Offline validator reports domain-ranking NDCG and adaptive-controller smoke-test coverage.
- New regression tests cover learned feedback ranking, feedback explainability consistency, and adaptive question selection.
