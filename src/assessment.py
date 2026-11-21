"""Deterministic, domain-aware diagnostic checks used as evidence for personalization.
No external API is required: questions are tied directly to catalog skills and each
answer maps to a proficiency level (0-3)."""

QUESTION_BANK = {
    "Data Science": [
        ("python", "Which Python structure stores key-value pairs?", ["List", "Tuple", "Dictionary", "Set"], 2),
        ("data_wrangling", "Which Pandas operation is commonly used to combine rows from two tables by a key?", ["merge", "plot", "describe", "astype"], 0),
        ("statistics", "A p-value below a chosen significance level is commonly used to indicate:", ["Evidence against the null hypothesis", "A larger sample automatically", "Causation", "Zero variance"], 0),
        ("machine_learning", "Which metric is especially useful for imbalanced binary classification?", ["Accuracy only", "F1-score", "R-squared only", "Mean absolute error only"], 1),
        ("deep_learning", "Backpropagation is primarily used to:", ["Store datasets", "Compute gradients for parameter updates", "Create SQL tables", "Render charts"], 1),
        ("nlp", "Embeddings are primarily used to represent:", ["Text/items as dense numeric vectors", "Passwords as plain text", "Only images", "Database schemas"], 0),
    ],
    "Web Development": [
        ("frontend_basics", "Which HTML element is commonly used for the main page heading?", ["<h1>", "<p>", "<div>", "<span>"], 0),
        ("javascript", "Which keyword declares a block-scoped variable that can be reassigned?", ["const", "let", "static", "define"], 1),
        ("frontend_framework", "In React, state is primarily used to:", ["Store changing UI data", "Create DNS records", "Compile CSS", "Replace HTTP"], 0),
        ("backend", "HTTP status 201 generally means:", ["Created", "Unauthorized", "Not Found", "Server Error"], 0),
        ("databases", "Which SQL clause filters rows before grouping?", ["WHERE", "ORDER BY", "HAVING", "LIMIT"], 0),
        ("devops", "CI/CD is mainly intended to automate:", ["Build, test and delivery/deployment", "Graphic design", "User interviews", "DNS ownership"], 0),
    ],
    "Cloud & DevOps": [
        ("cloud_basics", "Which cloud model provides virtual machines on demand?", ["IaaS", "SaaS", "DaaS only", "Manual hosting"], 0),
        ("linux", "Which command lists files in a Linux directory?", ["ls", "pwdx", "showfiles", "dirlist"], 0),
        ("containers", "A Docker image is best described as:", ["A packaged filesystem and metadata for creating containers", "A physical server", "A DNS record", "A database row"], 0),
        ("orchestration", "Kubernetes primarily helps with:", ["Container orchestration", "Spreadsheet editing", "Email delivery", "Image compression"], 0),
        ("devops", "A CI pipeline should ideally run tests:", ["Automatically on relevant changes", "Only after production failure", "Never", "Only manually once a year"], 0),
    ],
    "Cybersecurity": [
        ("security_basics", "The CIA triad stands for:", ["Confidentiality, Integrity, Availability", "Control, Inspection, Access", "Cloud, Identity, Audit", "Code, Internet, Authentication"], 0),
        ("network_security", "What does a firewall primarily control?", ["Network traffic based on rules", "CPU temperature", "Source code formatting", "Keyboard input"], 0),
        ("pentesting", "A penetration test should be performed:", ["With explicit authorization and scope", "Against any target without permission", "Only after deleting logs", "Without documenting findings"], 0),
    ],
    "Product Management": [
        ("pm_basics", "A product roadmap primarily communicates:", ["Planned product direction and priorities", "Source-code binaries", "Server CPU usage", "Legal contracts only"], 0),
        ("ux", "User interviews are mainly useful for understanding:", ["User needs and pain points", "Database indexes", "Compiler tokens", "Cloud regions"], 0),
        ("analytics", "A north-star metric should represent:", ["Core customer/product value", "Every possible event", "Only engineering cost", "A random vanity number"], 0),
    ],
    "Digital Marketing": [
        ("marketing_basics", "A conversion funnel describes:", ["Stages from awareness toward desired action", "Only server requests", "Database normalization", "Compiler phases"], 0),
        ("seo", "A search engine can use page titles to understand:", ["The page topic", "CPU speed", "A user's password", "Cloud billing"], 0),
        ("paid_ads", "CTR commonly means:", ["Click-through rate", "Customer tax ratio", "Cloud transfer request", "Content tracking record"], 0),
    ],
}


def questions(domain):
    return [
        {"skill": s, "question": q, "options": opts, "answer": ans}
        for s, q, opts, ans in QUESTION_BANK.get(domain, [])
    ]


def grade(domain, answers):
    qs = questions(domain)
    skill_scores = {}
    correct = 0
    for i, q in enumerate(qs):
        selected = answers.get(str(i)) if isinstance(answers, dict) else None
        ok = selected is not None and int(selected) == q["answer"]
        if ok:
            correct += 1
            skill_scores[q["skill"]] = 3
        else:
            skill_scores[q["skill"]] = 1 if selected is not None else 0
    total = len(qs)
    return {"score": correct, "total": total, "pct": round(correct * 100 / total) if total else 0, "skill_scores": skill_scores}

# --- Adaptive assessment layer -------------------------------------------------
# Difficulty is attached to the existing question bank without changing the public
# questions()/grade() API. The controller selects unseen questions near the learner's
# current estimated ability, then moves harder/easier after correct/incorrect answers.
DIFFICULTY_MAP = {
    "python": 1, "data_wrangling": 2, "statistics": 2, "machine_learning": 2,
    "deep_learning": 3, "nlp": 3, "frontend_basics": 1, "javascript": 2,
    "frontend_framework": 2, "backend": 2, "databases": 2, "devops": 3,
    "cloud_basics": 1, "linux": 1, "containers": 2, "orchestration": 3,
    "security_basics": 1, "network_security": 2, "pentesting": 3,
    "pm_basics": 1, "ux": 2, "analytics": 3, "marketing_basics": 1, "seo": 2,
    "paid_ads": 2,
}


def adaptive_questions(domain):
    """Return diagnostic questions with explicit 1..3 difficulty metadata."""
    out=[]
    for i,q in enumerate(questions(domain)):
        item=dict(q)
        item["id"]=f"{domain}:{i}"
        item["difficulty"]=DIFFICULTY_MAP.get(q["skill"], 2)
        out.append(item)
    return out


def estimate_ability(answered):
    """Estimate ability on a 1..3 scale from adaptive answer history."""
    if not answered:
        return 1.5
    total_weight=0.0; weighted=0.0
    for item in answered:
        difficulty=float(item.get("difficulty",2))
        correct=bool(item.get("correct",False))
        # Correct answers pull ability above the item difficulty; incorrect answers below it.
        target=difficulty + (0.45 if correct else -0.45)
        weight=1.0 + 0.15*len(answered)
        weighted += target*weight; total_weight += weight
    return max(1.0,min(3.0,weighted/max(total_weight,1e-9)))


def next_adaptive_question(domain, answered=None):
    """Select the next unseen item closest to current ability, with difficulty adaptation."""
    answered=list(answered or [])
    pool=adaptive_questions(domain)
    used={x.get("id") for x in answered}
    unseen=[q for q in pool if q["id"] not in used]
    if not unseen:
        return None
    ability=estimate_ability(answered)
    # Prefer the closest difficulty, then a not-yet-tested skill, then deterministic id.
    seen_skills={x.get("skill") for x in answered}
    unseen.sort(key=lambda q:(abs(q["difficulty"]-ability), q["skill"] in seen_skills, q["id"]))
    return unseen[0]


def grade_adaptive_answer(question, selected_index):
    """Grade one adaptive item and return bounded evidence for the controller."""
    if selected_index is None:
        correct=False
    else:
        try: correct=int(selected_index)==int(question["answer"])
        except (TypeError,ValueError): correct=False
    d=int(question.get("difficulty",2))
    evidence=max(0,min(3, round(d + (0.6 if correct else -0.6))))
    return {"id":question["id"],"skill":question["skill"],"difficulty":d,"correct":correct,"evidence":evidence}
