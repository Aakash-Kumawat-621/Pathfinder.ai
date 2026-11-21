"""PathFinder Streamlit application — polished demo UI around the AI recommendation engine."""
import os
import re
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering

sys.path.append(os.path.dirname(__file__))
from src import badges as bd
from src import database as db
from src import llm_engine as llm
from src import recommender as rec
from src import semantic_engine as sem
from src import skills_gap as sg
from src import planner as planner
from src import validators as val
from src import assessment as assess
from src.agents import Orchestrator

st.set_page_config(page_title="PathFinder | Personalized Learning", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")
db.init_db()
ORCH = Orchestrator()
CATALOG = rec.load_catalog()
ID_TO_ROW = {row["course_id"]: row for _, row in CATALOG.iterrows()}

@st.cache_resource(show_spinner=False)
def _warm_semantic_ann():
    """Load the catalog embedding ANN once per Streamlit process."""
    try:
        return sem.warm_catalog_index(CATALOG)
    except Exception:
        return None

SEMANTIC_INDEX = _warm_semantic_ann()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

:root {
  --pf-accent: #d4af37;
  --pf-accent-hover: #e5c158;
  --pf-accent-soft: rgba(212, 175, 55, 0.15);
  --pf-mint: #8b0000;
  --pf-mint-soft: rgba(139, 0, 0, 0.15);
  --pf-border: rgba(212, 175, 55, 0.4);
  --pf-card-bg: rgba(38, 28, 21, 0.7);
  --pf-card-hover: rgba(38, 28, 21, 0.9);
  --pf-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
  --font-family: 'Lora', serif;
  --heading-font: 'Cinzel', serif;
}

/* Typography & Layout */
html, body, [class*="css"] { font-family: var(--font-family) !important; color: #e8dcc5; }
.block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1400px; }
h1, h2, h3, h4, h5, h6 { font-family: var(--heading-font) !important; font-weight: 700; color: #d4af37; text-transform: uppercase; letter-spacing: 0.05em; }

/* Hero Banner - Parchment / Fantasy style */
.hero {
  padding: 2.5rem 2.5rem; border: 2px solid var(--pf-accent); border-radius: 8px;
  background: linear-gradient(135deg, rgba(26,22,20,0.9), rgba(38,28,21,0.9));
  margin-bottom: 2rem; box-shadow: var(--pf-shadow); position: relative; overflow: hidden;
}
.hero::before {
  content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background: url('https://www.transparenttextures.com/patterns/old-wall.png'); opacity: 0.2; z-index: 0; pointer-events: none;
}
.hero h1 { margin: 0 0 0.5rem 0; font-size: 2.8rem; position: relative; z-index: 1; text-shadow: 2px 2px 4px rgba(0,0,0,0.8); }
.hero p { margin: 0; opacity: 0.9; font-size: 1.15rem; font-weight: 400; position: relative; z-index: 1; color: #e8dcc5; }

/* Cards - Leather/Parchment feel */
.card {
  padding: 1.5rem; border: 1px solid var(--pf-border); border-radius: 8px; height: 100%;
  background: var(--pf-card-bg); transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
  position: relative;
}
.card::before {
  content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  background: url('https://www.transparenttextures.com/patterns/black-scales.png'); opacity: 0.15; z-index: 0; pointer-events: none;
}
.card > * { position: relative; z-index: 1; }
.card:hover {
  border-color: var(--pf-accent); transform: translateY(-4px);
  background: var(--pf-card-hover); box-shadow: 0 12px 24px rgba(0,0,0,0.5);
}
.card b { display: block; margin-top: 0.5rem; font-size: 1.1rem; color: #d4af37; font-family: var(--heading-font); }

/* Labels & Pills */
.small-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.8; font-weight: 600; font-family: var(--heading-font); color: #d4af37; }
.step-pill {
  display: inline-block; padding: 0.3rem 0.8rem; border-radius: 4px; background: rgba(139,0,0,0.3); 
  font-size: 0.8rem; font-weight: 600; color: #e8dcc5; border: 1px solid var(--pf-accent); font-family: var(--heading-font);
}

/* Sidebar styling */
[data-testid="stSidebar"] { border-right: 2px solid var(--pf-border); background-color: #1a1614 !important; }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid var(--pf-border); }
.stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0 0; padding: 0.75rem 1.25rem; font-weight: 600; transition: background 0.2s; font-family: var(--heading-font); }
.stTabs [aria-selected="true"] { background: var(--pf-accent-soft); color: var(--pf-accent) !important; border-bottom-color: var(--pf-accent) !important; }

/* Buttons with interactions */
.stButton > button, .stDownloadButton > button {
  border-radius: 4px; border: 1px solid var(--pf-accent); font-weight: 600; transition: all 0.2s ease; padding: 0.5rem 1.2rem;
  background: transparent; color: var(--pf-accent); font-family: var(--heading-font); text-transform: uppercase;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--pf-accent-hover); transform: translateY(-2px); background: rgba(212,175,55,0.1); color: var(--pf-accent-hover); box-shadow: 0 4px 12px rgba(212, 175, 55, 0.2);
}
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #8b0000, #5c0000); border: 1px solid var(--pf-accent); color: #e8dcc5 !important; }
.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #a50000, #7a0000); color: #fff !important; }

/* Gradient Progress Bars */
.stProgress > div > div > div { background: linear-gradient(90deg, #8b0000, var(--pf-accent)); border-radius: 2px; }

/* Metrics */
[data-testid="stMetric"] {
  background: var(--pf-card-bg); border: 1px solid var(--pf-border); border-radius: 8px; padding: 1rem 1.25rem;
  box-shadow: inset 0 0 10px rgba(0,0,0,0.5); transition: transform 0.2s ease; position: relative; overflow: hidden;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: var(--pf-accent); }
[data-testid="stMetricLabel"] { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.9; font-weight: 600; color: #d4af37; font-family: var(--heading-font); }
[data-testid="stMetricValue"] { font-family: var(--heading-font) !important; color: #e8dcc5 !important; }

/* Expanders */
.streamlit-expanderHeader, [data-testid="stExpander"] summary { border-radius: 4px !important; font-weight: 600; transition: background 0.2s; font-family: var(--heading-font); color: var(--pf-accent); border: 1px solid transparent; }
.streamlit-expanderHeader:hover, [data-testid="stExpander"] summary:hover { background: rgba(212,175,55,0.05); border: 1px solid var(--pf-accent-soft); }

/* Chat Bubbles */
[data-testid="stChatMessage"] { border-radius: 4px; border: 1px solid var(--pf-border); background: var(--pf-card-bg); padding: 1rem; margin-bottom: 1rem; border-left: 4px solid var(--pf-accent); }
</style>
""", unsafe_allow_html=True)

DEFAULTS = {
    "learner_id": None, "onboard_stage": "collect_name", "onboard_chat": [],
    "onboard_data": {}, "onboard_profile": None, "onboard_missing": [],
    "last_trace": [], "related_courses": [], "plan_meta": {}, "assessment_result": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value


def _reset_onboarding():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value
    _bot_say("Hi! I'm PathFinder 👋 What's your name?")


def _bot_say(text):
    st.session_state.onboard_chat.append({"role": "assistant", "content": text})


def _user_say(text):
    st.session_state.onboard_chat.append({"role": "user", "content": text})


if not st.session_state.onboard_chat:
    _bot_say("Hi! I'm PathFinder 👋 What's your name?")


def _extract_background(text):
    """Map free-text background to catalog skill tags and exact prior course IDs."""
    raw = str(text).strip()
    low = raw.lower()
    skills = set()
    aliases = {
        "python": "python", "pandas": "data_wrangling", "numpy": "data_wrangling", "statistics": "statistics",
        "stats": "statistics", "matplotlib": "data_viz", "visualization": "data_viz", "machine learning": "machine_learning",
        "ml": "machine_learning", "deep learning": "deep_learning", "nlp": "nlp", "javascript": "javascript",
        "js": "javascript", "html": "frontend_basics", "css": "frontend_basics", "react": "frontend_framework",
        "node": "backend", "sql": "databases", "database": "databases", "linux": "linux", "docker": "containers",
        "kubernetes": "orchestration", "k8s": "orchestration", "devops": "devops", "aws": "cloud_basics",
        "azure": "cloud_basics", "gcp": "cloud_basics", "security": "security_basics", "cybersecurity": "security_basics",
        "networking": "network_security", "penetration testing": "pentesting", "pentesting": "pentesting",
        "product management": "pm_basics", "ux": "ux", "seo": "seo", "marketing": "marketing_basics", "paid ads": "paid_ads",
    }
    for phrase, skill in aliases.items():
        if phrase in low:
            skills.add(skill)

    completed = set()
    for _, row in CATALOG.iterrows():
        cid = str(row["course_id"])
        title = str(row["title"]).lower()
        if cid.lower() in low or title in low:
            completed.add(cid)
        else:
            # Match distinctive title phrases, e.g. "Python Programming Fundamentals" -> py101.
            words = [w for w in re.findall(r"[a-z0-9]+", title) if len(w) > 3]
            if len(words) >= 3 and sum(w in low for w in words) >= max(3, int(len(words) * .65)):
                completed.add(cid)
    return sorted(skills), sorted(completed)


def _parse_time_commitment(text):
    t = text.lower()
    m = re.search(r"(\d+)\s*(?:-|to)?\s*(\d+)?\s*(?:hours?|hrs?)", t)
    if m:
        hours = int(m.group(2) or m.group(1))
        return "light" if hours <= 4 else "intensive" if hours >= 18 else "moderate"
    if any(w in t for w in ["few", "couple", "light", "little time"]):
        return "light"
    if any(w in t for w in ["lot", "full time", "intensive", "as much as possible"]):
        return "intensive"
    return "moderate"


def _confirmation_message(profile):
    known = ", ".join(profile.get("known_skills") or []) or "none listed"
    completed = len(profile.get("completed_courses") or [])
    weekly = rec.TIME_HOURS.get(profile.get("time_commitment", "moderate"), 10)
    return (
        f"### Your learning profile\n"
        f"- **Goal:** {profile.get('target_domain')}\n"
        f"- **Experience:** {profile.get('experience_level')}\n"
        f"- **Known skills:** {known}\n"
        f"- **Prior catalog courses:** {completed}\n"
        f"- **Study pace:** ~{weekly} hrs/week ({profile.get('time_commitment')})\n\n"
        "If this looks right, type **yes** and I'll generate a prerequisite-safe roadmap. 🚀"
    )


def _finalize_onboarding():
    data = st.session_state.onboard_data
    profile = st.session_state.onboard_profile or {}
    learner_id = db.create_learner(
        name=data.get("name", "Learner"), goal_text=data.get("goal_text", ""),
        target_domain=profile.get("target_domain", "Data Science"),
        experience_level=profile.get("experience_level", "beginner"),
        interests=profile.get("interests", []), known_skills=profile.get("known_skills", []),
        completed_courses=profile.get("completed_courses", []),
        time_commitment=profile.get("time_commitment", "moderate"),
    )
    for turn in st.session_state.onboard_chat:
        db.add_chat_message(learner_id, turn["role"], turn["content"])

    learner = db.get_learner(learner_id)
    learner["feedback"] = db.get_feedback(learner_id)
    with st.status("🧠 Building your personalized roadmap…", expanded=True) as pipeline_status:
        def _stream_step(title, detail):
            pipeline_status.update(label=f"{title} — {detail}", state="running")
            st.write(f"**{title}**  \n{detail}")
        path_items, related, trace = ORCH.run_path_generation(learner, progress_callback=_stream_step)
        pipeline_status.update(label="✅ Multi-agent roadmap ready", state="complete")
    total_hours = sum(float(ID_TO_ROW[p["course_id"]]["duration_hours"]) for p in path_items)
    weekly = rec.TIME_HOURS.get(learner.get("time_commitment", "moderate"), 10)
    meta = {"total_hours": total_hours, "estimated_weeks": max(1, int(round(total_hours / weekly)))}
    ordered = [p["course_id"] for p in path_items]
    try:
        val.validate_prerequisite_order(ordered, CATALOG, learner["completed_courses"])
        warnings = val.validate_no_duplicate_milestones_out_of_order([p["milestone"] for p in path_items])
        if warnings:
            trace.append(f"🛡️ Validator: generated path passed dependency checks with {len(warnings)} soft milestone warnings.")
        else:
            trace.append("🛡️ Validator: all prerequisite and milestone-order checks passed.")
    except val.PathValidationError as exc:
        st.error(f"Path validation stopped the save: {exc}")
        return

    db.save_learning_path(learner_id, path_items)
    st.session_state.learner_id = learner_id
    st.session_state.last_trace = st.session_state.last_trace + trace
    st.session_state.related_courses = related
    st.session_state.plan_meta = meta


def page_onboarding():
    st.markdown('<div class="hero"><h1>🧭 PathFinder</h1><p>Tell me where you want to go. I’ll turn it into a personalized, prerequisite-aware learning plan.</p></div>', unsafe_allow_html=True)
    for turn in st.session_state.onboard_chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_msg = st.chat_input("Type your reply…")
    if not user_msg:
        return
    _user_say(user_msg)
    stage = st.session_state.onboard_stage
    data = st.session_state.onboard_data

    if stage == "collect_name":
        data["name"] = user_msg.strip()
        st.session_state.onboard_stage = "collect_goal"
        _bot_say(f"Nice to meet you, **{data['name']}**! What do you want to learn, and what’s your current experience?\n\n*Example: “I want to become a data scientist; I know basic Python but I’m new to ML.”*")

    elif stage == "collect_goal":
        data["goal_text"] = user_msg
        with st.spinner("Understanding your goal…"):
            profile, _, trace = ORCH.run_onboarding(user_msg)
        st.session_state.onboard_profile = profile
        st.session_state.last_trace = trace
        st.session_state.onboard_stage = "followup_background"
        _bot_say(
            f"Got it — **{profile.get('target_domain')}** looks like the best fit at **{profile.get('experience_level')}** level. "
            "What do you already know or have you completed? You can write it naturally, e.g. **“Python, statistics; completed py101”** — or say **none**."
        )

    elif stage == "followup_background":
        skills, completed = _extract_background(user_msg)
        profile = st.session_state.onboard_profile
        if user_msg.strip().lower() not in {"none", "no", "nothing", "n/a"}:
            profile["known_skills"] = skills or [x.strip() for x in user_msg.split(",") if x.strip()]
            profile["completed_courses"] = completed
        else:
            profile["known_skills"] = []
            profile["completed_courses"] = []
        st.session_state.onboard_stage = "followup_time"
        _bot_say("Perfect. Roughly how many hours per week can you commit? **1–3**, **5–10**, **10–18**, or **as much as possible** all work.")

    elif stage == "followup_time":
        profile = st.session_state.onboard_profile
        profile["time_commitment"] = _parse_time_commitment(user_msg)
        st.session_state.onboard_stage = "confirm"
        _bot_say(_confirmation_message(profile))

    elif stage == "confirm":
        if user_msg.strip().lower() not in {"no", "change", "edit"}:
            _bot_say("Awesome — generating your personalized roadmap now…")
            _finalize_onboarding()
        else:
            st.session_state.onboard_stage = "followup_time"
            _bot_say("No problem. Tell me what you’d like to change (skills, courses, or weekly time), and I’ll update the profile.")
    st.rerun()


def _status_icon(status):
    return {"completed": "✅", "in_progress": "🔵", "not_started": "⬜"}.get(status, "⬜")


def _course_card(row, item):
    st.markdown(f"<span class='step-pill'>STEP {item['step_order']}</span> &nbsp; **{row['title']}**", unsafe_allow_html=True)
    st.caption(f"{row['type'].title()} · {row['level'].title()} · {row['skill_tag']} · ⏱ {float(row['duration_hours']):g}h")
    st.write(row["description"])
    prereqs = [ID_TO_ROW[p]["title"] if p in ID_TO_ROW else p for p in str(row["prerequisites"]).split(";") if p]
    st.caption("Prerequisites: " + (", ".join(prereqs) if prereqs else "None"))
    st.markdown(f"[Open learning resource ↗]({row['resource_url']})")
    st.info("**Why this is here:** " + item["explanation"])
    with st.expander("📐 Recommendation score breakdown"):
        learner=db.get_learner(st.session_state.learner_id)
        breakdown=rec.course_score_breakdown(row["course_id"],learner["target_domain"],learner["experience_level"],learner.get("known_skills",[]),learner.get("interests",[]),learner.get("goal_text", ""),learner.get("time_commitment","moderate"),db.get_feedback(learner["id"]))
        labels={"semantic_relevance":"Goal relevance","skill_gap":"Skill-gap relevance","domain_fit":"Domain fit","difficulty_fit":"Difficulty fit","interest_match":"Interest match","time_fit":"Time fit","feedback_fit":"Feedback fit"}
        cols=st.columns(4)
        for idx,(key,label) in enumerate(labels.items()):
            cols[idx%4].metric(label,f"{breakdown.get(key,0)*100:.0f}%")
        st.progress(float(breakdown.get("overall",0)))
        st.caption(f"Overall hybrid recommendation score: **{breakdown.get('overall',0)*100:.0f}/100**")


def _reapply_order(learner_id, new_order_ids, old_path_items):
    old_by_id = {p["course_id"]: p for p in old_path_items}
    new_items = []
    for i, cid in enumerate(new_order_ids, 1):
        old = old_by_id[cid]
        new_items.append({**old, "step_order": i})
    db.save_learning_path(learner_id, new_items)


def _regenerate_current_path(reason="profile update"):
    learner = db.get_learner(st.session_state.learner_id)
    learner["feedback"] = db.get_feedback(learner["id"])
    latest = db.get_latest_assessment(learner["id"])
    learner["assessment_skill_scores"] = (latest or {}).get("skill_scores", {})
    with st.status(f"🧠 Adapting your roadmap to {reason}…", expanded=True) as pipeline_status:
        def _stream_step(title, detail):
            pipeline_status.update(label=f"{title} — {detail}", state="running")
            st.write(f"**{title}**  \n{detail}")
        path_items, related, trace = ORCH.run_path_generation(learner, progress_callback=_stream_step)
        pipeline_status.update(label="✅ Adaptive roadmap rebuilt", state="complete")
    ordered=[p["course_id"] for p in path_items]
    try:
        val.validate_prerequisite_order(ordered, CATALOG, learner.get("completed_courses", []))
    except val.PathValidationError as exc:
        st.error(f"Safety validator blocked the update: {exc}")
        return
    db.save_learning_path(learner["id"], path_items)
    total_hours=sum(float(ID_TO_ROW[p["course_id"]]["duration_hours"]) for p in path_items)
    weekly=rec.TIME_HOURS.get(learner.get("time_commitment","moderate"),10)
    st.session_state.related_courses=related
    st.session_state.last_trace=trace
    st.session_state.plan_meta={"total_hours":total_hours,"estimated_weeks":max(1,int(round(total_hours/weekly)))}
    st.success("Your roadmap was re-personalized successfully — progress and prerequisites were preserved.")
    st.rerun()

def page_skill_check():
    learner=db.get_learner(st.session_state.learner_id)
    latest=db.get_latest_assessment(learner["id"])
    st.markdown('<div class="hero"><h1>🧪 Adaptive Skill Check</h1><p>Questions change difficulty from your answers, producing evidence that directly updates your roadmap.</p></div>', unsafe_allow_html=True)
    domain=learner["target_domain"]
    all_qs=assess.adaptive_questions(domain)
    st.info(f"Domain: **{domain}** · adaptive diagnostic · No API key required")

    state=st.session_state.setdefault("adaptive_check", {"domain":domain,"answered":[],"current":None,"finished":False})
    if state.get("domain") != domain:
        state={"domain":domain,"answered":[],"current":None,"finished":False}
        st.session_state.adaptive_check=state

    if st.button("↻ Start / restart adaptive check"):
        state={"domain":domain,"answered":[],"current":None,"finished":False}
        st.session_state.adaptive_check=state
        st.rerun()

    if not state["finished"]:
        if state["current"] is None:
            state["current"]=assess.next_adaptive_question(domain,state["answered"])
        q=state["current"]
        if q is None:
            state["finished"]=True
            st.rerun()
        st.progress(min(1.0,len(state["answered"])/min(6,max(1,len(all_qs)))))
        st.caption(f"Question {len(state['answered'])+1} of up to {min(6,len(all_qs))} · difficulty {q['difficulty']}/3 · skill: {q['skill'].replace('_',' ').title()}")
        with st.form("adaptive_question_form"):
            choice=st.radio(q["question"],q["options"],index=None)
            submitted=st.form_submit_button("Submit answer →",type="primary")
        if submitted:
            idx=q["options"].index(choice) if choice in q["options"] else None
            evidence=assess.grade_adaptive_answer(q,idx)
            state["answered"].append(evidence)
            if len(state["answered"]) >= min(6,len(all_qs)):
                state["finished"]=True
                skill_scores={}
                for item in state["answered"]:
                    skill_scores[item["skill"]]=max(skill_scores.get(item["skill"],0),item["evidence"])
                correct=sum(1 for x in state["answered"] if x["correct"])
                result={"score":correct,"total":len(state["answered"]),"pct":round(100*correct/max(1,len(state["answered"]))),"skill_scores":skill_scores,"adaptive":True}
                db.add_assessment(learner["id"],domain,result)
                st.session_state.assessment_result=result
            else:
                state["current"]=assess.next_adaptive_question(domain,state["answered"])
            st.rerun()

    if state.get("finished"):
        result=st.session_state.get("assessment_result") or latest
        if result:
            st.subheader(f"Adaptive diagnostic: {result.get('pct',0):g}%")
            st.progress(float(result.get("pct",0))/100)
            cols=st.columns(min(4,max(1,len(result.get("skill_scores",{})))))
            for col,(skill,score) in zip(cols,result.get("skill_scores",{}).items()):
                col.metric(skill.replace("_"," ").title(),f"{score}/3")
            st.success("Evidence captured. The next roadmap rebuild will use these proficiency estimates while preserving prerequisite safety.")
            if st.button("🔄 Rebuild my roadmap from this evidence",type="primary"):
                _regenerate_current_path("your adaptive diagnostic results")


def _render_prerequisite_graph(path_items, learner=None):
    """Interactive DAG with hierarchical clustering/collapse for dense roadmaps."""
    if not path_items:
        return
    st.subheader("🕸️ Interactive prerequisite roadmap")
    st.caption("Explore acquired skills → course milestones → target requirements. Dense branches can be collapsed into hierarchical clusters.")
    path_ids = [p["course_id"] for p in path_items]
    path_set = set(path_ids)
    dag = rec.DeterministicDAG(CATALOG)
    edges = [(parent, child) for child in path_ids for parent in dag.prereq.get(child, []) if parent in path_set]

    depth = {cid: 0 for cid in path_ids}
    for cid in dag.solve(path_ids, priority={cid: 0 for cid in path_ids}):
        parents = [p for p in dag.prereq.get(cid, []) if p in path_set]
        depth[cid] = max((depth[p] + 1 for p in parents), default=0)
    layers = {}
    for cid in path_ids:
        layers.setdefault(depth[cid], []).append(cid)
    for layer in layers.values(): layer.sort(key=lambda cid: path_ids.index(cid))
    x, y = {}, {}
    for d, nodes in layers.items():
        for j, cid in enumerate(nodes):
            x[cid] = d; y[cid] = (len(nodes)-1)/2-j
    max_layer = max(layers, default=0)

    collapse = st.toggle("🧩 Collapse dense prerequisite clusters", value=len(path_ids) > 12,
                         help="Uses hierarchical agglomerative clustering to group nearby prerequisite milestones. Turn off to inspect every node.")
    expanded = set()
    if collapse and len(path_ids) >= 6:
        labels = {}
        for d, nodes in layers.items():
            # Cluster by prerequisite depth + vertical proximity. This preserves
            # curriculum stages while reducing visual density.
            n_clusters = max(1, min(len(nodes), (len(nodes)+4)//5))
            if len(nodes) == 1:
                labels[nodes[0]] = f"Stage {d+1}"
            else:
                import numpy as np
                X = np.array([[float(d), float(j)] for j, _ in enumerate(nodes)])
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
                labs = model.fit_predict(X)
                for cid, lab in zip(nodes, labs): labels[cid] = f"Stage {d+1} · Cluster {int(lab)+1}"
        cluster_names = sorted(set(labels.values()))
        chosen = st.multiselect("Expand clusters", cluster_names, default=[],
                                help="Expanded clusters show their individual course milestones.")
        expanded = {cid for cid in path_ids if labels[cid] in chosen}
        cluster_x, cluster_y = {}, {}
        for name in cluster_names:
            members=[cid for cid in path_ids if labels[cid]==name]
            cluster_x[name]=sum(x[c] for c in members)/len(members)
            cluster_y[name]=sum(y[c] for c in members)/len(members)

        # Render collapsed clusters, with expanded members restored as individual nodes.
        node_x=[]; node_y=[]; texts=[]; hovers=[]; shown=set()
        for name in cluster_names:
            members=[cid for cid in path_ids if labels[cid]==name]
            collapsed=[c for c in members if c not in expanded]
            if collapsed:
                node_x.append(cluster_x[name]); node_y.append(cluster_y[name]); texts.append(f"🧩 {name} ({len(collapsed)})")
                hovers.append("<b>Hierarchical cluster</b><br>" + "<br>".join(ID_TO_ROW[c]["title"] for c in collapsed))
                shown.update(collapsed)
            for cid in members:
                if cid in expanded:
                    title = ID_TO_ROW[cid]['title']
                    short_title = title if len(title) <= 22 else title[:20] + "..."
                    node_x.append(x[cid]); node_y.append(y[cid]); texts.append(f"M{path_ids.index(cid)+1} · {short_title}")
                    hovers.append(f"<b>{title}</b><br>Skill: {ID_TO_ROW[cid]['skill_tag']}<br>Status: {next(p['status'] for p in path_items if p['course_id']==cid)}")
                    shown.add(cid)
        # Aggregate edges through cluster representatives.
        def rep(cid):
            return cid if cid in expanded else labels[cid]
        fig=go.Figure()
        drawn=set()
        for parent, child in edges:
            a,b=rep(parent),rep(child)
            if a==b or (a,b) in drawn: continue
            drawn.add((a,b))
            ax,ay=(x[a],y[a]) if a in x else (cluster_x[a],cluster_y[a])
            bx,by=(x[b],y[b]) if b in x else (cluster_x[b],cluster_y[b])
            fig.add_trace(go.Scatter(x=[ax,bx],y=[ay,by],mode="lines",line=dict(width=2),hoverinfo="skip",showlegend=False))
        fig.add_trace(go.Scatter(x=node_x,y=node_y,mode="markers+text",text=texts,textposition="bottom center",
                                 customdata=hovers,hovertemplate="%{customdata}<extra></extra>",
                                 marker=dict(size=28,line=dict(width=2)),name="Roadmap milestone"))
    else:
        fig=go.Figure()
        for parent, child in edges:
            fig.add_trace(go.Scatter(x=[x[parent],x[child]],y=[y[parent],y[child]],mode="lines",line=dict(width=2),hoverinfo="skip",showlegend=False))
        status_marker={"completed":"Acquired / completed","in_progress":"In progress","not_started":"Milestone"}
        node_x=[]; node_y=[]; texts=[]; hover=[]
        for item in path_items:
            cid=item["course_id"]; row=ID_TO_ROW[cid]
            title = row['title']
            short_title = title if len(title) <= 22 else title[:20] + "..."
            node_x.append(x[cid]); node_y.append(y[cid]); texts.append(f"M{item['step_order']} · {short_title}")
            hover.append(f"<b>Milestone {item['step_order']}: {title}</b><br>Status: {status_marker.get(item.get('status'),item.get('status'))}<br>Skill delivered: {row['skill_tag']}<br>Level: {row['level'].title()} · {float(row['duration_hours']):g}h<br>Prerequisites: {', '.join(dag.prereq.get(cid, [])) or 'None'}")
        fig.add_trace(go.Scatter(x=node_x,y=node_y,mode="markers+text",text=texts,textposition="bottom center",customdata=hover,hovertemplate="%{customdata}<extra></extra>",marker=dict(size=26,line=dict(width=2)),name="Course milestone"))

    # Calculate dynamic height based on the most dense vertical column
    max_nodes_vertically = max([len(nodes) for nodes in layers.values()] + [0])
    
    # Acquired/target skill nodes remain visible in both modes.
    if learner:
        gap=sg.compute_skill_gap(learner,path_items,ID_TO_ROW)
        acquired=[k for k,v in gap.items() if v["current"]>0]
        targets=[k for k,v in gap.items() if v["required"]>v["current"]]
        
        max_nodes_vertically = max(max_nodes_vertically, len(acquired), len(targets))
        
        for yy,skill in enumerate(acquired):
            pos_y = (len(acquired)-1)/2 - yy
            fig.add_trace(go.Scatter(x=[-1],y=[pos_y],mode="markers+text",text=[f"✓ {skill}"],textposition="middle left",marker=dict(size=18,symbol="circle",color="#d4af37",line=dict(width=1,color="white")),showlegend=False,hovertemplate=f"<b>Acquired skill: {skill}</b><br>Current proficiency: {gap[skill]['current']}/3<extra></extra>"))
        for yy,skill in enumerate(targets):
            pos_y = (len(targets)-1)/2 - yy
            fig.add_trace(go.Scatter(x=[max_layer+1],y=[pos_y],mode="markers+text",text=[f"🎯 {skill}"],textposition="middle right",marker=dict(size=18,symbol="diamond",color="#8b0000",line=dict(width=1,color="white")),showlegend=False,hovertemplate=f"<b>Target requirement: {skill}</b><br>Required: {gap[skill]['required']}/3<br>Current: {gap[skill]['current']}/3<extra></extra>"))
            
    fig_height = max(550, max_nodes_vertically * 90)
    fig.update_layout(height=fig_height,margin=dict(l=40,r=40,t=40,b=40),xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),hovermode="closest",showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color="#fafafa"))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"scrollZoom":True})
    mode="hierarchically clustered" if collapse and len(path_ids)>=6 else "fully expanded"
    st.caption(f"{len(path_ids)} milestones · {len(edges)} prerequisite edges · {mode} · acquired skills → milestones → target requirements")


def page_learning_path():
    learner = db.get_learner(st.session_state.learner_id)
    path_items = db.get_learning_path(st.session_state.learner_id)
    stats = rec.path_progress_stats(path_items)
    st.markdown(f'<div class="hero"><h1>🗺️ {learner["name"]}’s Learning Path</h1><p>{learner["goal_text"]}</p></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Progress", f"{stats['pct_complete']}%")
    c2.metric("Completed", stats["completed"])
    c3.metric("In progress", stats["in_progress"])
    c4.metric("Remaining", stats["not_started"])
    st.progress(stats["pct_complete"] / 100)

    _render_prerequisite_graph(path_items, learner)

    meta = st.session_state.plan_meta
    if meta:
        st.caption(f"Estimated workload: **{meta.get('total_hours', 0):.0f} hours** · Suggested pace: **~{meta.get('estimated_weeks', 1)} weeks**")
    action = rec.next_best_action(path_items, ID_TO_ROW)
    if action:
        st.success(f"🎯 **Next best action:** Step {action['step_order']} — {action['title']} ({action['duration_hours']:g}h).")

    b1,b2,b3=st.columns(3)
    if b1.button("🧪 Take skill diagnostic",key="path_diag"):
        st.session_state.workspace_page="🧪 Skill Check"
        st.rerun()
    if b2.button("🔄 Re-personalize path",key="repath"):
        _regenerate_current_path("your latest progress and feedback")
    if b3.button("📋 View learner evidence",key="evidence"):
        latest=db.get_latest_assessment(learner["id"])
        if latest: st.info(f"Latest diagnostic: {latest['pct']:.0f}% on {latest['domain']}. Skill evidence: {latest['skill_scores']}")
        else: st.info("No diagnostic completed yet. Take the skill check to add objective evidence.")

    weekly=rec.TIME_HOURS.get(learner.get("time_commitment","moderate"),10)
    weeks=planner.weekly_plan(path_items,ID_TO_ROW,weekly)
    if weeks:
        st.subheader("📅 Personalized weekly schedule")
        st.caption(f"Built from your ~{weekly} hours/week. Completed items are automatically excluded.")
        for w in weeks[:8]:
            with st.expander(f"Week {w['week']} · {w['hours']:.1f} planned hours", expanded=w['week']==1):
                for x in w['items']:
                    st.write(f"**Step {x['step']} — {x['title']}** · {x['duration_hours']:g}h · {x['type'].title()} · {x['skill']}")

    if st.session_state.last_trace:
        with st.expander("🧠 Explainable AI pipeline — see how the plan was built"):
            for line in st.session_state.last_trace:
                st.write(line)

    feedback = db.get_feedback(learner["id"])
    for item in path_items:
        row = ID_TO_ROW.get(item["course_id"])
        if row is None:
            continue
        with st.expander(f"{_status_icon(item['status'])} Step {item['step_order']}: {row['title']} · {row['level'].title()}", expanded=item["status"] == "in_progress"):
            left, right = st.columns([3.5, 1])
            with left:
                _course_card(row, item)
            with right:
                options = ["not_started", "in_progress", "completed"]
                selected = st.selectbox("Progress", options, index=options.index(item["status"]), key=f"status_{item['course_id']}")
                if selected != item["status"]:
                    db.update_progress(learner["id"], item["course_id"], selected)
                    st.rerun()
                st.write("**Was this recommendation useful?**")
                b1, b2 = st.columns(2)
                if b1.button("👍", key=f"up_{item['course_id']}"):
                    db.add_feedback(learner["id"], item["course_id"], "up")
                    st.toast("Feedback saved — similar skills will be prioritized.")
                if b2.button("👎", key=f"down_{item['course_id']}"):
                    db.add_feedback(learner["id"], item["course_id"], "down")
                    new_order = rec.adjust_path_on_feedback(learner, path_items, item["course_id"], "down", feedback)
                    _reapply_order(learner["id"], new_order, path_items)
                    st.toast("Path adapted while preserving prerequisites.")
                    st.rerun()
                comment = st.text_input("Optional feedback", key=f"comment_{item['course_id']}", placeholder="Too easy? Too long? Not relevant?")
                if comment and st.button("Save note", key=f"savefb_{item['course_id']}"):
                    rating = "down" if any(x in comment.lower() for x in ("too easy", "too long", "not relevant", "irrelevant", "confusing")) else "up"
                    db.add_feedback(learner["id"], item["course_id"], rating, comment)
                    st.toast("Thanks — your note is part of the learner feedback signal.")

    if st.session_state.related_courses:
        st.divider()
        st.subheader("✨ Optional semantic recommendations")
        cols = st.columns(len(st.session_state.related_courses))
        for col, (cid, score) in zip(cols, st.session_state.related_courses):
            row = ID_TO_ROW.get(cid)
            if row is None:
                continue
            with col:
                st.markdown(f"**{row['title']}**")
                st.caption(f"{row['domain']} · semantic match {score:.0%}")
                st.write(row["description"])

    csv = pd.DataFrame([{
        "step": p["step_order"], "milestone": p["milestone"], "title": ID_TO_ROW[p["course_id"]]["title"],
        "type": ID_TO_ROW[p["course_id"]]["type"], "skill": ID_TO_ROW[p["course_id"]]["skill_tag"],
        "duration_hours": ID_TO_ROW[p["course_id"]]["duration_hours"], "status": p["status"],
    } for p in path_items]).to_csv(index=False)
    st.download_button("⬇️ Export roadmap CSV", csv, file_name="pathfinder_learning_path.csv", mime="text/csv")


def page_dashboard():
    learner = db.get_learner(st.session_state.learner_id)
    path_items = db.get_learning_path(learner["id"])
    if not path_items:
        st.info("Generate a learning path first.")
        return
    stats = rec.path_progress_stats(path_items)
    st.markdown('<div class="hero"><h1>📊 Progress & Skill Intelligence</h1><p>Track outcomes, identify gaps, and know what to do next.</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall progress", f"{stats['pct_complete']}%")
    c2.metric("Completed", stats["completed"])
    c3.metric("Current pace", learner["time_commitment"].title())
    meta = st.session_state.plan_meta
    c4.metric("Est. weeks", meta.get("estimated_weeks", "—"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Skill coverage")
        breakdown = rec.skill_breakdown(path_items, ID_TO_ROW)
        skill_df = pd.DataFrame([{"Skill": k, "Completed": v["completed"], "Remaining": v["total"] - v["completed"]} for k, v in breakdown.items()])
        if not skill_df.empty:
            st.bar_chart(skill_df.set_index("Skill"))
    with col2:
        st.subheader("Role readiness")
        latest = db.get_latest_assessment(learner["id"])
        learner["assessment_skill_scores"] = (latest or {}).get("skill_scores", {})
        gap = sg.compute_skill_gap(learner, path_items, ID_TO_ROW)
        if gap:
            skills = list(gap)
            current = [gap[s]["current"] for s in skills]
            required = [gap[s]["required"] for s in skills]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=current + [current[0]], theta=skills + [skills[0]], fill="toself", name="Learner"))
            fig.add_trace(go.Scatterpolar(r=required + [required[0]], theta=skills + [skills[0]], name="Role target", line=dict(dash="dot")))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 3])), height=360, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            summary = sg.gap_summary(gap)
            st.success(f"Readiness: **{summary['readiness_pct']}%** · Biggest gap: **{summary['biggest_gap_skill']}**")

    st.divider()
    action = rec.next_best_action(path_items, ID_TO_ROW)
    next_item = next((p for p in path_items if p["course_id"] == action["course_id"]), None) if action else None
    if next_item:
        row = ID_TO_ROW[next_item["course_id"]]
        st.subheader("🎯 Next best action")
        st.info(f"**Step {next_item['step_order']} — {row['title']}** · {row['duration_hours']}h\n\n{next_item['explanation']}")
    else:
        st.success("🏆 Path complete! Set a more advanced goal to continue your growth.")

    st.subheader("🏅 Achievements")
    badges = bd.compute_badges(path_items, db.get_feedback(learner["id"]))
    cols = st.columns(len(badges))
    for col, badge in zip(cols, badges):
        with col:
            st.markdown(f"<div class='card'><div style='font-size:32px'>{badge['icon']}</div><b>{badge['name']}</b><br><small>{'Earned' if badge['earned'] else 'Locked'}</small></div>", unsafe_allow_html=True)

    with st.expander("👤 Learner profile"):
        a, b, c = st.columns(3)
        a.write(f"**Domain:** {learner['target_domain']}\n\n**Level:** {learner['experience_level']}")
        b.write(f"**Known skills:** {', '.join(learner['known_skills']) or 'None'}")
        c.write(f"**Completed before PathFinder:** {len(learner['completed_courses'])}\n\n**Interests:** {', '.join(learner['interests']) or '—'}")


def page_ask_ai():
    learner = db.get_learner(st.session_state.learner_id)
    path_items = db.get_learning_path(learner["id"])
    st.markdown('<div class="hero"><h1>🤖 Ask your AI Learning Advisor</h1><p>Ask why a step exists, how to study it, or how your progress changes the plan.</p></div>', unsafe_allow_html=True)
    history = db.get_chat_history(learner["id"])
    for turn in history:
        if turn["role"] in ("user", "assistant"):
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
    user_msg = st.chat_input("e.g. Why do I need statistics before machine learning?")
    if user_msg:
        db.add_chat_message(learner["id"], "user", user_msg)
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = ORCH.advisor.chat(learner, path_items, history, user_msg, trace=st.session_state.last_trace)
            st.markdown(reply)
        db.add_chat_message(learner["id"], "assistant", reply)


def main():
    st.sidebar.markdown("# 🧭 PathFinder")
    st.sidebar.caption("AI-powered personalized learning path recommender")
    if llm.is_ai_available():
        st.sidebar.success("AI engine: connected")
    else:
        st.sidebar.info("AI engine: offline-safe mode")
    st.sidebar.caption(f"Semantic retrieval: `{sem.backend_name()}`")

    if st.session_state.learner_id:
        learner = db.get_learner(st.session_state.learner_id)
        st.sidebar.divider()
        st.sidebar.write(f"**{learner['name']}**")
        st.sidebar.caption(f"{learner['target_domain']} · {learner['experience_level']}")
        pages=["💬 Onboarding Chat", "🗺️ Learning Path", "🧪 Skill Check", "📊 Dashboard", "🤖 Ask AI"]
        default_page=pages.index(st.session_state.get("workspace_page","🗺️ Learning Path")) if st.session_state.get("workspace_page","🗺️ Learning Path") in pages else 1
        page = st.sidebar.radio("Workspace", pages, index=default_page)
        st.session_state.workspace_page=page
        if st.sidebar.button("🔄 Start a new learner"):
            _reset_onboarding()
            st.rerun()
    else:
        page = "💬 Onboarding Chat"
        st.sidebar.info("Start with a short conversation to create your learner profile.")

    {"💬 Onboarding Chat": page_onboarding, "🗺️ Learning Path": page_learning_path, "🧪 Skill Check": page_skill_check,
     "📊 Dashboard": page_dashboard, "🤖 Ask AI": page_ask_ai}[page]()


if __name__ == "__main__":
    main()
