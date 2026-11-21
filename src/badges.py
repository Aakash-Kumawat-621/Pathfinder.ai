"""
badges.py — lightweight gamification layer. Badges are computed from real
progress/feedback data already in the DB, not a separate tracking system.
"""
from datetime import datetime

BADGE_DEFS = [
    {"id": "first_step", "name": "First Step", "icon": "🌱",
     "desc": "Completed your first learning item."},
    {"id": "halfway_hero", "name": "Halfway Hero", "icon": "🚀",
     "desc": "Reached 50% completion on your path."},
    {"id": "path_master", "name": "Path Master", "icon": "🏆",
     "desc": "Completed your entire learning path."},
    {"id": "feedback_champion", "name": "Feedback Champion", "icon": "💬",
     "desc": "Gave feedback on 3+ recommendations, helping PathFinder adapt."},
    {"id": "consistent_learner", "name": "Consistent Learner", "icon": "🔥",
     "desc": "Made progress on 3+ different days."},
]


def compute_badges(path_items, feedback_items):
    total = len(path_items)
    completed = sum(1 for p in path_items if p["status"] == "completed")
    pct = (100 * completed / total) if total else 0

    active_days = set()
    for p in path_items:
        if p["status"] != "not_started" and p.get("updated_at"):
            active_days.add(p["updated_at"][:10])  # date part of ISO timestamp

    earned_ids = set()
    if completed >= 1:
        earned_ids.add("first_step")
    if pct >= 50:
        earned_ids.add("halfway_hero")
    if total > 0 and completed == total:
        earned_ids.add("path_master")
    if len(feedback_items) >= 3:
        earned_ids.add("feedback_champion")
    if len(active_days) >= 3:
        earned_ids.add("consistent_learner")

    result = []
    for b in BADGE_DEFS:
        result.append({**b, "earned": b["id"] in earned_ids})
    return result
