# PathFinder Ultimate — Release Notes

## What changed in this release

### Evidence-based personalization
- Added a deterministic domain-aware Skill Check for all six supported domains.
- Diagnostic results are persisted in SQLite and surfaced as per-skill proficiency evidence.
- Dashboard role-readiness now combines self-report, prior learning, completed path work and diagnostic evidence.

### Closed-loop adaptation
- Added one-click roadmap re-personalization using current progress, feedback and diagnostic evidence.
- Replanning is validated before persistence and remains prerequisite-safe.
- Feedback comments now infer positive/negative signals instead of always recording a thumbs-up.
- Fixed skill-level feedback propagation so feedback on one resource can influence related resources sharing the same skill.

### Quality
- 266 automated tests pass.
- 18/18 offline evaluation scenarios pass.
- Prerequisite validation remains 100%.

## Recommended 3–5 minute demo
1. Create a learner with a natural-language goal.
2. Show the personalized prerequisite-safe roadmap.
3. Open **Skill Check**, answer a few questions and show per-skill evidence.
4. Click **Rebuild my roadmap from this evidence**.
5. Mark a course complete and show progress/readiness.
6. Give a recommendation a thumbs-down or note and click **Re-personalize path**.
7. Open score breakdown and AI Advisor to demonstrate explainability.


## Round 7 — AI depth and adaptive assessment
- Added an online logistic feedback ranker with cold-start fallback.
- Fixed feedback explainability to use skill-level preference keys consistently.
- Added difficulty-aware adaptive question selection and evidence estimation.
- Added offline NDCG and adaptive-controller validation.
- Test suite: 282 passed.
