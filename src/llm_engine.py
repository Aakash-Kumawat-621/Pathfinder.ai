"""LLM layer with structured extraction, grounded explanations and safe fallbacks."""
import json
import os
import re
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    from openai import OpenAI
    _openai_import_ok = True
except ImportError:
    _openai_import_ok = False

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DOMAINS = ["Data Science", "Web Development", "Cloud & DevOps", "Cybersecurity", "Product Management", "Digital Marketing"]
LEVELS = ["beginner", "intermediate", "advanced"]
TIMES = ["light", "moderate", "intensive"]

DOMAIN_KEYWORDS = {
    "Data Science": ["data scien", "machine learning", "ml", "artificial intelligence", "ai", "analytics", "data analy", "deep learning", "nlp", "data scientist"],
    "Web Development": ["web dev", "frontend", "front-end", "backend", "back-end", "full stack", "full-stack", "react", "javascript", "website", "web app"],
    "Cloud & DevOps": ["cloud", "devops", "aws", "azure", "gcp", "kubernetes", "docker", "ci/cd", "deployment", "infrastructure"],
    "Cybersecurity": ["cyber", "security", "ethical hack", "penetration", "pentest", "soc", "vulnerability"],
    "Product Management": ["product manag", "product owner", "pm role", "product design", "roadmap", "prd"],
    "Digital Marketing": ["marketing", "seo", "social media", "growth", "paid ads", "campaign"],
}
LEVEL_KEYWORDS = {
    "advanced": ["advanced", "expert", "years of experience", "senior", "professional"],
    "intermediate": ["some experience", "intermediate", "know the basics", "worked with", "familiar with", "have built"],
    "beginner": ["beginner", "new to", "never coded", "no experience", "just starting", "from scratch", "newbie"],
}
TIME_KEYWORDS = {
    "light": ["1-2", "2-3", "few hours", "couple hours", "little time", "light"],
    "intensive": ["20", "30", "40", "full time", "intensive", "as much as possible", "many hours"],
}

PROMPT_DIR = Path(__file__).resolve().parents[1] / "config" / "prompts"


def load_prompt(name):
    """Load versioned prompt templates from config/prompts without code edits."""
    path = PROMPT_DIR / f"{name}.yaml"
    try:
        if yaml is None:
            raise RuntimeError("PyYAML unavailable")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict) or not data.get("system"):
            raise ValueError(f"Invalid prompt config: {path}")
        return data
    except Exception:
        # Safe fallback keeps offline/demo mode functional if configuration is missing.
        return {"version": "fallback", "system": "You are a helpful learning assistant."}


def _get_secret(key, default=None):
    val = os.environ.get(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


def _get_client():
    if not _openai_import_ok:
        return None
    key = _get_secret("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


def is_ai_available():
    return _get_client() is not None


def _normalise_skill_list(values):
    seen = set()
    out = []
    for value in values or []:
        v = re.sub(r"\s+", " ", str(value).strip().lower())
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _rule_based_parse(goal_text, known_skills_hint=None, completed_courses_hint=None):
    text = str(goal_text).lower()
    domain_scores = {
        dom: sum(1 for kw in kws if kw in text)
        for dom, kws in DOMAIN_KEYWORDS.items()
    }
    domain = max(domain_scores, key=domain_scores.get) if any(domain_scores.values()) else "Data Science"

    level = "beginner"
    for candidate in ("advanced", "intermediate", "beginner"):
        if any(kw in text for kw in LEVEL_KEYWORDS[candidate]):
            level = candidate
            break

    interests = [dom for dom, score in domain_scores.items() if score > 0]
    interests = interests or [domain]
    return {
        "target_domain": domain,
        "experience_level": level,
        "interests": interests,
        "known_skills": _normalise_skill_list(known_skills_hint),
        "completed_courses": list(completed_courses_hint or []),
        "time_commitment": _infer_time(text),
    }


def _infer_time(text):
    for level, kws in TIME_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return level
    return "moderate"


def _validate_profile(data, known_skills_hint=None, completed_courses_hint=None):
    if not isinstance(data, dict):
        raise ValueError("Profile is not an object")
    domain = data.get("target_domain")
    if domain not in DOMAINS:
        raise ValueError("Invalid target domain")
    level = data.get("experience_level")
    if level not in LEVELS:
        raise ValueError("Invalid experience level")
    time = data.get("time_commitment", "moderate")
    if time not in TIMES:
        time = "moderate"
    skills = _normalise_skill_list(data.get("known_skills", []))
    for s in known_skills_hint or []:
        if str(s).strip().lower() not in skills:
            skills.append(str(s).strip().lower())
    completed = list(dict.fromkeys([str(x).strip() for x in (data.get("completed_courses", []) + list(completed_courses_hint or [])) if str(x).strip()]))
    interests = [str(x).strip() for x in data.get("interests", []) if str(x).strip()][:8]
    return {
        "target_domain": domain,
        "experience_level": level,
        "interests": interests or [domain],
        "known_skills": skills,
        "completed_courses": completed,
        "time_commitment": time,
    }


def parse_learner_goal(goal_text, known_skills_hint=None, completed_courses_hint=None):
    """Extract a validated learner profile; deterministic fallback is always available."""
    client = _get_client()
    if client is None:
        return _rule_based_parse(goal_text, known_skills_hint, completed_courses_hint)

    system = load_prompt("profiling")["system"]
    try:
        resp = client.chat.completions.create(
            model=_get_secret("OPENAI_MODEL", MODEL),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": goal_text}],
            temperature=0.1,
            max_tokens=350,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        return _validate_profile(json.loads(raw), known_skills_hint, completed_courses_hint)
    except Exception:
        return _rule_based_parse(goal_text, known_skills_hint, completed_courses_hint)


def explain_recommendation(learner_profile, course_row, step_order, total_steps):
    client = _get_client()
    prereqs = str(course_row.get("prerequisites", "")) or "none"
    fallback = (
        f"Step {step_order} of {total_steps}: **{course_row['title']}** targets **{course_row['skill_tag']}**, "
        f"which supports your {learner_profile.get('target_domain', 'career')} goal. "
        f"Its prerequisites ({prereqs}) are satisfied by earlier steps or your existing background."
    )
    if client is None:
        return fallback
    cfg = load_prompt("advisor")
    prompt = cfg.get("template", "").format(
        goal_text=learner_profile.get("goal_text", ""), target_domain=learner_profile.get("target_domain"),
        experience_level=learner_profile.get("experience_level"), known_skills=learner_profile.get("known_skills"),
        title=course_row["title"], skill_tag=course_row["skill_tag"], level=course_row["level"],
        type=course_row["type"], prerequisites=prereqs
    )
    try:
        resp = client.chat.completions.create(
            model=_get_secret("OPENAI_MODEL", MODEL),
            messages=[{"role": "system", "content": cfg["system"]}, {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        return (resp.choices[0].message.content or fallback).strip()
    except Exception:
        return fallback


def chat_reply(learner_profile, path_items, chat_history, user_message):
    """Answer using only supplied learner/path context; never fail the demo because AI is unavailable."""
    completed = sum(p["status"] == "completed" for p in path_items)
    remaining = len(path_items) - completed
    if not _get_client():
        return (
            f"Your current path has **{len(path_items)} steps**, with **{completed} completed** and **{remaining} remaining**. "
            f"Your target is **{learner_profile.get('target_domain', 'your goal')}** at {learner_profile.get('experience_level', 'beginner')} level. "
            "Configure OPENAI_API_KEY to enable conversational AI answers; the core recommendation engine continues to work offline."
        )

    path_summary = "\n".join(
        f"{p['step_order']}. {p.get('course_id')} — {p.get('status')} — milestone {p.get('milestone')}" for p in path_items
    )
    context = (
        f"LEARNER: {json.dumps(learner_profile, default=str)}\n"
        f"CURRENT PATH:\n{path_summary}\n"
        f"USER QUESTION: {user_message}"
    )
    system = load_prompt("chat")["system"]
    messages = [{"role": "system", "content": system}, {"role": "system", "content": context}]
    for turn in (chat_history or [])[-8:]:
        if turn.get("role") in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})
    try:
        resp = _get_client().chat.completions.create(
            model=_get_secret("OPENAI_MODEL", MODEL), messages=messages,
            temperature=0.4, max_tokens=320,
        )
        return (resp.choices[0].message.content or "I couldn't generate a response just now.").strip()
    except Exception:
        return (
            f"I couldn't reach the AI service right now, but your path is still available: {completed}/{len(path_items)} steps completed. "
            "You can continue learning and use the feedback controls while the AI service is unavailable."
        )
