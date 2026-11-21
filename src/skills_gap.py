"""
skills_gap.py — computes a "current vs required" skill profile for the
learner's target domain, used to power the radar/gap chart on the dashboard.

Proficiency scale: 0 = none, 1 = beginner, 2 = intermediate, 3 = advanced.
"""

# What proficiency each domain's core skills need at a competent/employable level.
ROLE_REQUIREMENTS = {
    "Data Science": {
        "python": 3, "statistics": 2, "data_wrangling": 2,
        "data_viz": 2, "machine_learning": 3, "deep_learning": 2, "nlp": 1,
    },
    "Web Development": {
        "frontend_basics": 2, "javascript": 3, "frontend_framework": 3,
        "backend": 3, "databases": 2, "devops": 1,
    },
    "Cloud & DevOps": {
        "cloud_basics": 3, "linux": 2, "containers": 3,
        "orchestration": 2, "devops": 3,
    },
    "Cybersecurity": {
        "security_basics": 3, "network_security": 3, "pentesting": 2,
    },
    "Product Management": {
        "pm_basics": 3, "ux": 2, "analytics": 2,
    },
    "Digital Marketing": {
        "marketing_basics": 3, "seo": 2, "paid_ads": 2,
    },
}

LEVEL_TO_SCORE = {"beginner": 1, "intermediate": 2, "advanced": 3}


def compute_skill_gap(learner, path_items, id_to_row):
    """
    Returns {skill_tag: {'current': int, 'required': int}} for the learner's
    target domain. 'current' blends:
      - a base score if the learner self-reported the skill during onboarding
      - boosts from any catalog items they've marked completed
    """
    domain = learner.get("target_domain")
    required = ROLE_REQUIREMENTS.get(domain, {})
    if not required:
        return {}

    known_skills = set(s.lower().strip() for s in (learner.get("known_skills") or []))
    base_level = LEVEL_TO_SCORE.get(learner.get("experience_level", "beginner"), 1)

    current = {skill: (base_level if skill in known_skills else 0) for skill in required}

    # Prior learning completed before joining PathFinder counts as real evidence too.
    for cid in learner.get("completed_courses", []) or []:
        row = id_to_row.get(cid)
        if row is None:
            continue
        skill = row["skill_tag"]
        if skill in current:
            current[skill] = max(current[skill], LEVEL_TO_SCORE.get(row["level"], 1))

    # Diagnostic assessment is stronger evidence than self-report.
    assessment = learner.get("assessment_skill_scores") or {}
    for skill, score in assessment.items():
        if skill in current:
            current[skill] = max(current[skill], int(score))

    # Then layer in progress made inside the generated path.
    for item in path_items:
        row = id_to_row.get(item["course_id"])
        if row is None or item["status"] != "completed":
            continue
        skill = row["skill_tag"]
        if skill in current:
            level_score = LEVEL_TO_SCORE.get(row["level"], 1)
            current[skill] = max(current[skill], level_score)

    return {skill: {"current": current[skill], "required": req} for skill, req in required.items()}


def gap_summary(gap_dict):
    """Returns overall readiness % and the single biggest gap skill."""
    if not gap_dict:
        return {"readiness_pct": 0, "biggest_gap_skill": None}
    total_required = sum(v["required"] for v in gap_dict.values())
    total_current = sum(min(v["current"], v["required"]) for v in gap_dict.values())
    readiness_pct = round(100 * total_current / total_required, 1) if total_required else 0
    biggest_gap = max(gap_dict.items(), key=lambda kv: kv[1]["required"] - kv[1]["current"])
    return {"readiness_pct": readiness_pct, "biggest_gap_skill": biggest_gap[0]}
