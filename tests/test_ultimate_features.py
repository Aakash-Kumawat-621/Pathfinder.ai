import pandas as pd
from src import recommender as rec
from src import planner

def test_score_breakdown_has_all_dimensions():
    b=rec.course_score_breakdown("py101","Data Science","beginner",[],["python"],"become a data scientist","moderate",[])
    assert set(["semantic_relevance","skill_gap","domain_fit","difficulty_fit","interest_match","time_fit","feedback_fit","overall"]).issubset(b)
    assert 0<=b["overall"]<=1

def test_weekly_plan_respects_capacity_for_normal_courses():
    df=rec.load_catalog(); rows={r.course_id:r for r in df.itertuples()}
    ids=list(df.course_id[:4]); items=[{"course_id":x,"step_order":i+1,"status":"not_started","milestone":"Core"} for i,x in enumerate(ids)]
    weeks=planner.weekly_plan(items,rows,10)
    assert weeks and all(w["hours"]>0 for w in weeks)
    assert round(sum(w["hours"] for w in weeks),1) == round(sum(float(rows[x].duration_hours) for x in ids),1)

def test_feedback_preferences_is_bounded():
    p=rec.feedback_preferences([{"course_id":"py101","rating":"up","comment":"I prefer hands-on projects"},{"course_id":"py102","rating":"down","comment":"too long"}])
    assert -1<=p["project_preference"]<=1 and -1<=p["short_preference"]<=1

def test_domain_fit_is_semantically_alias_aware():
    assert rec.domain_fit("Data Science","I want to build machine learning systems")>0.5

def test_weekly_plan_never_exceeds_capacity():
    df=rec.load_catalog(); rows={r.course_id:r for r in df.itertuples()}
    items=[{"course_id":"py101","step_order":1,"status":"not_started","milestone":"Foundations"}]
    weeks=planner.weekly_plan(items,rows,7)
    assert len(weeks)==3 and all(w["hours"]<=7.0 for w in weeks)
