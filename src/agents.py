"""Explainable multi-agent orchestration for PathFinder."""
from src import llm_engine as llm
from src import recommender as rec
from src import semantic_engine as sem


class ProfilerAgent:
    name = "ProfilerAgent"

    def parse(self, goal_text, known_skills_hint=None, completed_courses_hint=None, trace=None):
        profile = llm.parse_learner_goal(goal_text, known_skills_hint, completed_courses_hint)
        trace = trace if trace is not None else []
        missing = []
        if not profile.get("known_skills"):
            missing.append("known_skills")
        if not profile.get("completed_courses"):
            missing.append("completed_courses")
        if profile.get("time_commitment") in (None, "", "moderate"):
            missing.append("time_commitment")
        trace.append(
            f"🧑‍💼 {self.name}: profile extracted → {profile.get('target_domain')} / "
            f"{profile.get('experience_level')} / {profile.get('time_commitment')} pace; "
            f"known skills={len(profile.get('known_skills', []))}, prior courses={len(profile.get('completed_courses', []))}."
        )
        return profile, missing


class RecommenderAgent:
    name = "RecommenderAgent"

    def build_path(self, learner, trace, progress_callback=None):
        df = rec.load_catalog()
        if progress_callback: progress_callback("🔎 Semantic retrieval", "Finding the most relevant learning signals…")
        sem_domain, sem_scores = sem.best_matching_domain(learner["goal_text"], df)
        chosen = learner["target_domain"]
        confidence = sem_scores.get(chosen, 0.0)
        trace.append(
            f"🧭 {self.name}: semantic retrieval via {sem.backend_name()} → "
            f"profile domain='{chosen}', best semantic domain='{sem_domain}', confidence={confidence:.2f}."
        )
        if progress_callback: progress_callback("🧩 Dependency planning", "Solving prerequisites and selecting the next legal milestones…")
        ordered_ids, id_to_row = rec.generate_path(
            target_domain=chosen,
            experience_level=learner["experience_level"],
            known_skills=list(learner.get("known_skills", [])) + [k for k,v in (learner.get("assessment_skill_scores") or {}).items() if int(v) >= 3],
            completed_courses=learner.get("completed_courses", []),
            interests=learner.get("interests", []),
            goal_text=learner.get("goal_text", ""),
            time_commitment=learner.get("time_commitment", "moderate"),
            feedback=learner.get("feedback", []),
            assessed_skill_scores=learner.get("assessment_skill_scores", {}),
        )
        milestones = rec.assign_milestones(ordered_ids, id_to_row)
        total_hours = sum(float(id_to_row[c]["duration_hours"]) for c in ordered_ids)
        weekly = rec.TIME_HOURS.get(learner.get("time_commitment", "moderate"), 10)
        weeks = max(1, int(round(total_hours / weekly)))
        trace.append(
            f"🗺️ {self.name}: ranked {len(ordered_ids)} required steps with prerequisite-safe "
            f"topological ordering; estimated workload ≈ {total_hours:.0f}h / {weeks} weeks."
        )
        if progress_callback: progress_callback("✨ Optional recommendations", "Retrieving relevant electives outside the core path…")
        related = sem.related_courses(learner["goal_text"], df, exclude_ids=ordered_ids, top_k=4)
        trace.append(f"✨ {self.name}: retrieved {len(related)} optional semantic electives outside the core path.")
        return ordered_ids, id_to_row, milestones, related


class AdvisorAgent:
    name = "AdvisorAgent"

    def explain_step(self, learner, course_row, step_order, total_steps):
        return llm.explain_recommendation(learner, course_row, step_order, total_steps)

    def chat(self, learner, path_items, chat_history, user_message, trace=None):
        reply = llm.chat_reply(learner, path_items, chat_history, user_message)
        if trace is not None:
            trace.append(f"🤖 {self.name}: answered from the current learner profile and path state.")
        return reply


class Orchestrator:
    def __init__(self):
        self.profiler = ProfilerAgent()
        self.recommender = RecommenderAgent()
        self.advisor = AdvisorAgent()

    def run_onboarding(self, goal_text, known_skills_hint=None, completed_courses_hint=None):
        trace = []
        profile, missing = self.profiler.parse(goal_text, known_skills_hint or [], completed_courses_hint or [], trace)
        return profile, missing, trace

    def run_path_generation(self, learner, progress_callback=None):
        trace = []
        if progress_callback: progress_callback("🧑‍💼 Agents initialized", "Profiler → recommender → advisor pipeline started.")
        ordered_ids, id_to_row, milestones, related = self.recommender.build_path(learner, trace, progress_callback)
        path_items = []
        total = len(ordered_ids)
        for i, cid in enumerate(ordered_ids, start=1):
            row = id_to_row[cid]
            if progress_callback: progress_callback("🤖 Advisor agent", f"Grounding explanation {i}/{total}: {row['title']}")
            path_items.append({
                "course_id": cid,
                "step_order": i,
                "milestone": milestones[i - 1],
                "explanation": self.advisor.explain_step(learner, row, i, total),
            })
        trace.append(f"📝 {self.advisor.name}: generated grounded explanations for all {total} steps.")
        return path_items, related, trace
