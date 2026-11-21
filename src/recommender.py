"""
recommender.py — personalized, prerequisite-safe learning-path engine.

The engine is deliberately hybrid:
  • the course catalog is a dependency graph, so prerequisite order is deterministic;
  • semantic retrieval scores goal/course relevance;
  • learner level, known skills, interests, time commitment and feedback affect ranking;
  • every final path is validated before it can be shown to the learner.

This keeps the system explainable and robust instead of asking an LLM to invent a
curriculum from scratch.
"""
import os
from collections import defaultdict

import pandas as pd

from src import semantic_engine as sem
from src.skills_gap import ROLE_REQUIREMENTS

try:
    from sklearn.linear_model import LogisticRegression
    _SKLEARN_RANKER_OK = True
except Exception:
    LogisticRegression = None
    _SKLEARN_RANKER_OK = False

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "courses.csv")
LEVEL_RANK = {"beginner": 0, "intermediate": 1, "advanced": 2}
TIME_HOURS = {"light": 5, "moderate": 10, "intensive": 18}


def load_catalog():
    df = pd.read_csv(CATALOG_PATH)
    df["prerequisites"] = df["prerequisites"].fillna("")
    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce").fillna(1)
    return df


def _build_graph(df):
    prereq_map = {}
    for _, row in df.iterrows():
        prereq_map[row["course_id"]] = [
            p.strip() for p in str(row["prerequisites"]).split(";") if p.strip()
        ]
    return prereq_map


class DeterministicDAG:
    """Backward-compatible facade over the explicit planner DAG solver."""
    def __init__(self, df):
        from src.planner import PrerequisiteDAGSolver
        self._solver = PrerequisiteDAGSolver(df)
        self.prereq = self._solver.prereq
        self.children = self._solver.children
        self.nodes = self._solver.nodes

    def ancestors(self, target):
        return self._solver.ancestors([target])

    def solve(self, targets, priority=None, excluded=None):
        return self._solver.topological_order(targets, satisfied=excluded, priority=priority)


def _all_ancestors(course_id, prereq_map, seen=None):
    if seen is None:
        seen = set()
    for p in prereq_map.get(course_id, []):
        if p not in seen:
            seen.add(p)
            _all_ancestors(p, prereq_map, seen)
    return seen


def _normalise(values):
    return {str(v).strip().lower() for v in (values or []) if str(v).strip()}


def _interest_tokens(interests):
    text = " ".join(str(x) for x in (interests or [])).lower()
    return {t.strip() for t in text.replace("/", " ").replace(",", " ").split() if t.strip()}


def _course_feature_vector(row, goal_text, experience_level, known_skills, interests, time_commitment, semantic_scores):
    """Numerical feature vector used by the online feedback ranker."""
    known = _normalise(known_skills)
    skill = str(row["skill_tag"]).lower()
    title_desc = f"{row['title']} {row['description']}".lower()
    level = LEVEL_RANK.get(str(row["level"]).lower(), 0)
    target = LEVEL_RANK.get(experience_level, 0)
    weekly = TIME_HOURS.get(time_commitment, TIME_HOURS["moderate"])
    duration = float(row["duration_hours"])
    return [
        float(semantic_scores.get(row["course_id"], 0.0)),
        float(_skill_gap_priority(skill, known, experience_level, row["domain"])),
        float(domain_fit(row["domain"], goal_text)),
        float(1.0 if level == target else (0.82 if level < target else 0.68)),
        float(_semantic_interest_fit(interests, title_desc, float(semantic_scores.get(row["course_id"], 0.0)))),
        float(1.0 if duration <= weekly else 0.88 if duration <= weekly * 2 else 0.70),
        float(str(row.get("type", "course")).lower() == "project"),
        float(min(duration / max(weekly, 1.0), 3.0) / 3.0),
    ]

def learned_feedback_scores(df, goal_text, experience_level, known_skills, interests, time_commitment, semantic_scores, feedback):
    """Train a small online logistic ranker from this learner's explicit feedback.

    It activates only after enough labelled interactions exist, preserving a deterministic
    cold-start fallback. This is a real supervised ML layer, not a hand-written feedback rule.
    """
    if not _SKLEARN_RANKER_OK or not feedback:
        return {}
    rows = {str(r["course_id"]): r for _, r in df.iterrows()}
    samples=[]; labels=[]
    for fb in feedback:
        cid=str(fb.get("course_id") or "")
        if cid not in rows or fb.get("rating") not in {"up","down"}:
            continue
        samples.append(_course_feature_vector(rows[cid], goal_text, experience_level, known_skills, interests, time_commitment, semantic_scores))
        labels.append(1 if fb["rating"] == "up" else 0)
    if len(samples) < 4 or len(set(labels)) < 2:
        return {}
    model=LogisticRegression(max_iter=300, class_weight="balanced", random_state=42)
    model.fit(samples, labels)
    X=[_course_feature_vector(row, goal_text, experience_level, known_skills, interests, time_commitment, semantic_scores) for _,row in df.iterrows()]
    probs=model.predict_proba(X)[:,1]
    return dict(zip(df["course_id"], probs.astype(float)))

def _personalization_scores(df, goal_text, experience_level, known_skills, interests,
                            time_commitment, semantic_scores, feedback=None):
    """Return explainable 0–1 preference scores for each course."""
    known = _normalise(known_skills)
    interest_text = " ".join(str(x).lower() for x in (interests or []))
    target_level = LEVEL_RANK.get(experience_level, 0)
    weekly = TIME_HOURS.get(time_commitment, TIME_HOURS["moderate"])
    scores = {}
    feedback = feedback or []
    # Interpretable preference memory inferred from explicit feedback/comments.
    pref = feedback_preferences(feedback, df)
    learned = learned_feedback_scores(df, goal_text, experience_level, known_skills, interests, time_commitment, semantic_scores, feedback)

    for _, row in df.iterrows():
        level = LEVEL_RANK.get(str(row["level"]).lower(), 0)
        skill = str(row["skill_tag"]).lower()
        title_desc = f"{row['title']} {row['description']}".lower()

        semantic = float(semantic_scores.get(row["course_id"], 0.0))
        level_fit = 1.0 if level == target_level else (0.82 if level < target_level else 0.68)
        if skill in known and level == 0:
            level_fit *= 0.45  # avoid repeating a known foundation
        interest_fit = _semantic_interest_fit(interests, title_desc, semantic)
        duration = float(row["duration_hours"])
        pace_fit = 1.0 if duration <= weekly else 0.88 if duration <= weekly * 2 else 0.70
        type_fit = 1.0
        if pref["project_preference"] > 0 and str(row["type"]).lower() == "project": type_fit += 0.10 * pref["project_preference"]
        if pref["short_preference"] > 0 and duration <= weekly: type_fit += 0.08 * pref["short_preference"]
        if pref["short_preference"] < 0 and duration > weekly * 2: type_fit += 0.08 * (-pref["short_preference"])
        feedback_fit = max(0.0, min(1.0, 0.5 + 0.25 * pref["course_type_bias"] + 0.25 * pref["skill_bias"].get(skill, 0.0)))

        # Explicit, auditable hybrid ranker. Weights sum to 1.0 before feedback memory.
        hybrid = (
            0.35 * semantic + 0.25 * _skill_gap_priority(skill, known, experience_level, row["domain"]) +
            0.12 * domain_fit(row["domain"], goal_text) + 0.10 * level_fit +
            0.08 * interest_fit + 0.05 * pace_fit + 0.05 * feedback_fit
        ) * type_fit
        # Blend learned preference only when sufficient labelled feedback exists.
        learned_score = float(learned.get(row["course_id"], hybrid))
        scores[row["course_id"]] = 0.80 * hybrid + 0.20 * learned_score if learned else hybrid
    return scores


def _semantic_interest_fit(interests, title_desc, semantic):
    tokens=_interest_tokens(interests)
    if not tokens: return 0.5
    hits=sum(1 for t in tokens if len(t)>2 and t in title_desc)
    lexical=min(1.0, hits/max(1,len(tokens))*2.0)
    return min(1.0, 0.55*lexical + 0.45*max(0.0,float(semantic)))

def _skill_gap_priority(skill, known, experience_level, domain=None):
    if skill in known: return 0.15
    req=(ROLE_REQUIREMENTS.get(domain or "",{}) or {}).get(skill, 2)
    level={"beginner":1,"intermediate":2,"advanced":3}.get(experience_level,1)
    gap=max(0,req-min(req,level))
    return min(1.0,0.55+0.15*gap)

def domain_fit(domain, goal_text):
    text=str(goal_text).lower(); d=str(domain).lower()
    aliases={"data science":["data","ml","machine learning","ai","analytics"],"web development":["web","frontend","backend","javascript","react"],"cloud & devops":["cloud","devops","docker","kubernetes","aws"],"cybersecurity":["cyber","security","pentest","ethical hacking"],"product management":["product","roadmap","prd","product manager"],"digital marketing":["marketing","seo","growth","ads"]}
    keys=aliases.get(d,[d]); return 1.0 if any(k in text for k in keys) else 0.35

def feedback_preferences(feedback, catalog=None):
    """Convert feedback into bounded, interpretable course/type/skill signals."""
    pref={"course_type_bias":0.0,"project_preference":0.0,"short_preference":0.0,"skill_bias":{},"course_bias":{}}
    skill_by_course={}
    if catalog is not None:
        skill_by_course={str(r["course_id"]):str(r["skill_tag"]).lower() for _,r in catalog.iterrows()}
    for fb in feedback or []:
        sign=1.0 if fb.get("rating")=="up" else -1.0
        comment=str(fb.get("comment","")).lower()
        if any(x in comment for x in ("project","hands-on","practical","build")): pref["project_preference"] += .25*sign
        if "too long" in comment or "long" in comment: pref["short_preference"] += -.25*sign
        if "short" in comment or "quick" in comment: pref["short_preference"] += .25*sign
        cid=str(fb.get("course_id") or "")
        if cid:
            pref["course_bias"][cid]=pref["course_bias"].get(cid,0)+.20*sign
            skill=skill_by_course.get(cid)
            if skill: pref["skill_bias"][skill]=pref["skill_bias"].get(skill,0)+.15*sign
        pref["course_type_bias"] += .15*sign
    for k in ("project_preference","short_preference","course_type_bias"):
        pref[k]=max(-1,min(1,pref[k]))
    pref["skill_bias"]={k:max(-1,min(1,v)) for k,v in pref["skill_bias"].items()}
    return pref

def course_score_breakdown(course_id, target_domain, experience_level, known_skills, interests,
                            goal_text, time_commitment="moderate", feedback=None):
    df=load_catalog(); row=df[df["course_id"]==course_id]
    if row.empty: return {}
    row=row.iloc[0]; sem_scores=sem.semantic_similarity_scores(goal_text or target_domain,df)
    semantic=float(sem_scores.get(course_id,0.0)); known=_normalise(known_skills)
    level=LEVEL_RANK.get(str(row["level"]).lower(),0); target=LEVEL_RANK.get(experience_level,0)
    level_fit=1.0 if level==target else (0.82 if level<target else 0.68)
    skill_gap=_skill_gap_priority(str(row["skill_tag"]).lower(),known,experience_level,row["domain"])
    interest=_semantic_interest_fit(interests,f"{row['title']} {row['description']}".lower(),semantic)
    weekly=TIME_HOURS.get(time_commitment,10); duration=float(row["duration_hours"]); pace=1.0 if duration<=weekly else .88 if duration<=weekly*2 else .70
    dom=domain_fit(row["domain"],goal_text); pref=feedback_preferences(feedback or [], df)
    fb=max(0,min(1,0.5+0.25*pref["course_type_bias"]+0.25*pref["skill_bias"].get(str(row["skill_tag"]).lower(),0)))
    total=(.35*semantic+.25*skill_gap+.12*dom+.10*level_fit+.08*interest+.05*pace+.05*fb)
    learned = learned_feedback_scores(df, goal_text or target_domain, experience_level, known_skills, interests, time_commitment, sem_scores, feedback or [])
    learned_pref=float(learned.get(course_id, total)) if learned else total
    overall=(0.80*total+0.20*learned_pref) if learned else total
    return {"semantic_relevance":semantic,"skill_gap":skill_gap,"domain_fit":dom,"difficulty_fit":level_fit,"interest_match":interest,"time_fit":pace,"feedback_fit":fb,"learned_preference":learned_pref,"overall":max(0,min(1,overall))}

def _stable_topological_sort(required, prereq_map, id_to_row, priority):
    """Kahn topological sort with deterministic personalized tie-breaking."""
    required = set(required)
    in_degree = {cid: 0 for cid in required}
    graph = defaultdict(list)
    for cid in required:
        for p in prereq_map.get(cid, []):
            if p in required:
                graph[p].append(cid)
                in_degree[cid] += 1

    def key(cid):
        row = id_to_row[cid]
        return (-priority.get(cid, 0), LEVEL_RANK.get(str(row["level"]).lower(), 0), cid)

    ready = sorted([cid for cid in required if in_degree[cid] == 0], key=key)
    ordered = []
    while ready:
        cid = ready.pop(0)
        ordered.append(cid)
        newly_ready = []
        for nxt in graph[cid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                newly_ready.append(nxt)
        ready.extend(newly_ready)
        ready.sort(key=key)

    # A malformed catalog should never silently produce a partial curriculum.
    if len(ordered) != len(required):
        raise ValueError("Course catalog contains a prerequisite cycle or invalid dependency graph.")
    return ordered


def generate_path(target_domain, experience_level, known_skills, completed_courses,
                  interests=None, goal_text="", time_commitment="moderate", feedback=None,
                  assessed_skill_scores=None):
    """Build a personalized path while guaranteeing prerequisite validity."""
    df = load_catalog()
    dag = DeterministicDAG(df)
    prereq_map = dag.prereq
    id_to_row = {row["course_id"]: row for _, row in df.iterrows()}
    known = _normalise(known_skills)
    completed = set(completed_courses or [])

    domain_items = df[df["domain"] == target_domain]
    if domain_items.empty:
        domain_items = df

    # The catalog's advanced project is the destination. Its transitive prerequisite
    # closure defines what is actually necessary to reach the destination.
    goals = list(domain_items[domain_items["type"].eq("project") & domain_items["level"].eq("advanced")]["course_id"])
    if not goals:
        goals = list(domain_items[domain_items["level"].eq("advanced")]["course_id"])
    if not goals:
        goals = list(domain_items["course_id"])[-3:]

    # Explicit DAG closure is resolved by the graph solver itself.
    # Completed courses are treated as already-satisfied prerequisites.
    excluded = set(completed)
    if experience_level in ("intermediate", "advanced"):
        # A demonstrated foundation is also treated as satisfied, so it can be skipped
        # without accidentally dropping its downstream courses.
        excluded |= {
            cid for cid in dag.nodes if cid in id_to_row
            and str(id_to_row[cid]["level"]).lower() == "beginner"
            and str(id_to_row[cid]["skill_tag"]).lower() in known
        }

    semantic_scores = sem.semantic_similarity_scores(goal_text or target_domain, df)
    priority = _personalization_scores(
        df, goal_text or target_domain, experience_level, known_skills,
        interests, time_commitment, semantic_scores, feedback=feedback
    )

    # Feedback is converted into small, transparent ranking nudges. Structural dependencies
    # still win because sorting is always topological.
    for fb in (feedback or []):
        cid = fb.get("course_id")
        if cid not in priority:
            continue
        delta = 0.07 if fb.get("rating") == "up" else -0.07 if fb.get("rating") == "down" else 0
        priority[cid] += delta
        row = id_to_row[cid]
        skill = str(row["skill_tag"]).lower()
        for other, other_row in id_to_row.items():
            if other in priority and str(other_row["skill_tag"]).lower() == skill and other != cid:
                priority[other] += delta * 0.45

    # Deterministic DAG traversal is the final authority. Priority only breaks ties
    # between currently-ready independent nodes; it can never move a node ahead of a prerequisite.
    # Cost-weighted A* chooses the least-friction legal frontier using assessed
    # baseline proficiency while the DAG remains the hard safety boundary.
    from src.planner import CostWeightedAStar
    baseline = {str(k).lower(): float(v) for k, v in (assessed_skill_scores or {}).items()}
    ordered = CostWeightedAStar(df, dag_solver=dag._solver).search(
        goals, satisfied=excluded, baseline=baseline, priority=priority
    )
    return ordered, id_to_row


def assign_milestones(ordered_ids, id_to_row):
    milestones = []
    for cid in ordered_ids:
        level = str(id_to_row[cid]["level"]).lower()
        item_type = str(id_to_row[cid]["type"]).lower()
        if item_type == "project" and level == "advanced":
            m = "Capstone"
        elif level == "beginner":
            m = "Foundations"
        elif level == "intermediate":
            m = "Core Skills"
        else:
            m = "Applied / Advanced"
        milestones.append(m)
    return milestones


def path_progress_stats(path_items):
    total = len(path_items)
    completed = sum(1 for p in path_items if p["status"] == "completed")
    in_progress = sum(1 for p in path_items if p["status"] == "in_progress")
    pct = round(100 * completed / total, 1) if total else 0
    return {"total": total, "completed": completed, "in_progress": in_progress,
            "not_started": total - completed - in_progress, "pct_complete": pct}


def skill_breakdown(path_items, id_to_row):
    stats = defaultdict(lambda: {"total": 0, "completed": 0, "in_progress": 0})
    for p in path_items:
        row = id_to_row.get(p["course_id"])
        if row is None:
            continue
        tag = row["skill_tag"]
        stats[tag]["total"] += 1
        if p["status"] == "completed":
            stats[tag]["completed"] += 1
        elif p["status"] == "in_progress":
            stats[tag]["in_progress"] += 1
    return dict(stats)


def adjust_path_on_feedback(learner_profile, path_items, course_id, rating, feedback_items=None):
    """Re-rank remaining items without ever breaking prerequisite order."""
    df = load_catalog()
    id_to_row = {row["course_id"]: row for _, row in df.iterrows()}
    if course_id not in id_to_row:
        return [p["course_id"] for p in path_items]

    feedback = list(feedback_items or []) + [{"course_id": course_id, "rating": rating}]
    priority = {p["course_id"]: 0.0 for p in path_items}
    target_skill = str(id_to_row[course_id]["skill_tag"]).lower()
    for fb in feedback:
        cid = fb.get("course_id")
        if cid not in priority:
            continue
        delta = 1.0 if fb.get("rating") == "up" else -1.0 if fb.get("rating") == "down" else 0.0
        priority[cid] += delta
        if cid in id_to_row:
            skill = str(id_to_row[cid]["skill_tag"]).lower()
            for pid in priority:
                if pid != cid and str(id_to_row.get(pid, {}).get("skill_tag", "")).lower() == skill:
                    priority[pid] += 0.35 * delta

    # Keep completed/in-progress order fixed; re-sort only the untouched suffix,
    # using the actual prerequisite graph for safety.
    fixed = [p for p in path_items if p["status"] != "not_started"]
    pending = [p for p in path_items if p["status"] == "not_started"]
    pending_ids = {p["course_id"] for p in pending}
    prereq_map = _build_graph(df)
    ordered_pending = _stable_topological_sort(pending_ids, prereq_map, id_to_row, priority)

    # Preserve the existing milestone/status records; only order changes.
    return [p["course_id"] for p in fixed] + ordered_pending


def next_best_action(path_items, id_to_row):
    """Return the most actionable current step, preferring in-progress work."""
    current = next((p for p in path_items if p["status"] == "in_progress"), None)
    if current is None:
        current = next((p for p in path_items if p["status"] == "not_started"), None)
    if current is None:
        return None
    row = id_to_row.get(current["course_id"])
    if row is None:
        return None
    return {"course_id": current["course_id"], "title": row["title"],
            "skill": row["skill_tag"], "duration_hours": float(row["duration_hours"]),
            "status": current["status"], "step_order": current["step_order"]}
