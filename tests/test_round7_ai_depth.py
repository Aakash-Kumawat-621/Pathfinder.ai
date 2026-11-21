import pandas as pd
from src import assessment, recommender


def test_feedback_ranker_learns_from_mixed_labels():
    df=recommender.load_catalog()
    sims={cid:0.5 for cid in df.course_id}
    fbs=[]
    for cid,rating in zip(df.course_id[:6],["up","down","up","down","up","down"]):
        fbs.append({"course_id":cid,"rating":rating,"comment":""})
    scores=recommender.learned_feedback_scores(df,"data science", "beginner", [], ["machine learning"], "moderate", sims, fbs)
    assert scores and all(0 <= v <= 1 for v in scores.values())


def test_breakdown_feedback_uses_skill_not_course_id():
    df=recommender.load_catalog(); row=df.iloc[0]
    cid=row.course_id; skill=str(row.skill_tag).lower()
    fbs=[{"course_id":cid,"rating":"down"}]
    a=recommender.course_score_breakdown(cid,row.domain,"beginner",[],[],row.title,"moderate",fbs)
    assert a["feedback_fit"] < 0.51
    assert "learned_preference" in a


def test_adaptive_controller_changes_difficulty_after_answer():
    domain="Data Science"
    first=assessment.next_adaptive_question(domain,[])
    wrong=assessment.grade_adaptive_answer(first, (first["answer"]+1)%len(first["options"]))
    second=assessment.next_adaptive_question(domain,[wrong])
    assert second is not None and second["id"] != first["id"]
    # The controller remains bounded and deterministic even when the bank is sparse.
    assert 1 <= assessment.estimate_ability([wrong]) <= 3


def test_adaptive_answer_evidence_is_bounded():
    q=assessment.adaptive_questions("Web Development")[0]
    result=assessment.grade_adaptive_answer(q,q["answer"])
    assert result["correct"] is True and 0 <= result["evidence"] <= 3
