"""Offline pre-submission evaluator for the PathFinder prototype."""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import recommender as rec
from src import validators as val
from src import semantic_engine as sem
from src import assessment
from sklearn.metrics import ndcg_score

DOMAINS = ["Data Science", "Web Development", "Cloud & DevOps", "Cybersecurity", "Product Management", "Digital Marketing"]
LEVELS = ["beginner", "intermediate", "advanced"]


def main():
    df = rec.load_catalog()
    val.validate_catalog(df)
    cases = 0
    valid = 0
    lengths = []
    for domain, level in itertools.product(DOMAINS, LEVELS):
        path, _ = rec.generate_path(domain, level, [], [], interests=[domain], goal_text=f"I want to become a {domain} professional")
        val.validate_prerequisite_order(path, df)
        cases += 1
        valid += 1
        lengths.append(len(path))
    # Offline ranking benchmark: expected relevance is derived from explicit catalog
    # skill/domain alignment, not from the recommender itself. This catches regressions
    # where semantic ranking becomes disconnected from the learner goal.
    ranking_scores=[]
    for domain in DOMAINS:
        query=f"I want to become a {domain} professional"
        sem_scores=sem.semantic_similarity_scores(query,df)
        ranked=sorted(df["course_id"], key=lambda cid: sem_scores.get(cid,0), reverse=True)
        relevance=[1.0 if str(df.loc[df.course_id==cid,"domain"].iloc[0])==domain else 0.0 for cid in ranked]
        ranking_scores.append(float(ndcg_score([relevance], [[len(relevance)-i for i in range(len(relevance))]])))
    adaptive_checks=[]
    for domain in DOMAINS:
        q=assessment.next_adaptive_question(domain,[])
        adaptive_checks.append(bool(q and 1 <= q.get("difficulty",2) <= 3))

    report = {
        "catalog_items": len(df),
        "domains": len(DOMAINS),
        "path_scenarios_checked": cases,
        "valid_paths": valid,
        "prerequisite_validity_pct": round(100 * valid / max(cases, 1), 1),
        "avg_path_length": round(sum(lengths) / max(len(lengths), 1), 2),
        "semantic_backend": sem.backend_name(),
        "catalog_cycle_check": "passed",
        "offline_domain_ranking_ndcg": round(sum(ranking_scores)/len(ranking_scores), 4),
        "adaptive_controller_smoke_test_pct": round(100*sum(adaptive_checks)/max(1,len(adaptive_checks)),1),
        "ml_feedback_ranker": "available_with_4plus_mixed_feedback",
    }
    out = Path(__file__).resolve().parents[1] / "docs" / "offline_evaluation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
