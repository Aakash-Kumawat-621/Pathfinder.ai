"""test_badges.py — gamification badge logic."""
from src import badges as bd


def _path(statuses, updated_dates=None):
    items = []
    for i, s in enumerate(statuses):
        item = {"course_id": f"c{i}", "status": s}
        if updated_dates:
            item["updated_at"] = updated_dates[i]
        items.append(item)
    return items


def test_no_badges_for_untouched_path():
    path_items = _path(["not_started", "not_started", "not_started"])
    result = bd.compute_badges(path_items, [])
    assert all(not b["earned"] for b in result)


def test_first_step_badge():
    path_items = _path(["completed", "not_started", "not_started"])
    result = bd.compute_badges(path_items, [])
    earned = {b["id"]: b["earned"] for b in result}
    assert earned["first_step"] is True
    assert earned["path_master"] is False


def test_halfway_hero_badge():
    path_items = _path(["completed", "completed", "not_started", "not_started"])
    result = bd.compute_badges(path_items, [])
    earned = {b["id"]: b["earned"] for b in result}
    assert earned["halfway_hero"] is True


def test_path_master_requires_full_completion():
    path_items = _path(["completed", "completed"])
    result = bd.compute_badges(path_items, [])
    earned = {b["id"]: b["earned"] for b in result}
    assert earned["path_master"] is True

    path_items_partial = _path(["completed", "not_started"])
    result2 = bd.compute_badges(path_items_partial, [])
    earned2 = {b["id"]: b["earned"] for b in result2}
    assert earned2["path_master"] is False


def test_feedback_champion_badge():
    path_items = _path(["not_started"])
    feedback = [{"rating": "up"}, {"rating": "down"}, {"rating": "up"}]
    result = bd.compute_badges(path_items, feedback)
    earned = {b["id"]: b["earned"] for b in result}
    assert earned["feedback_champion"] is True


def test_consistent_learner_badge_counts_distinct_days():
    path_items = _path(
        ["completed", "completed", "completed"],
        updated_dates=["2026-01-01T10:00:00", "2026-01-01T11:00:00", "2026-01-02T10:00:00"],
    )
    # only 2 distinct days -> should NOT earn (needs 3+)
    result = bd.compute_badges(path_items, [])
    earned = {b["id"]: b["earned"] for b in result}
    assert earned["consistent_learner"] is False

    path_items_3days = _path(
        ["completed", "completed", "completed"],
        updated_dates=["2026-01-01T10:00:00", "2026-01-02T11:00:00", "2026-01-03T10:00:00"],
    )
    result2 = bd.compute_badges(path_items_3days, [])
    earned2 = {b["id"]: b["earned"] for b in result2}
    assert earned2["consistent_learner"] is True


def test_empty_path_does_not_crash():
    result = bd.compute_badges([], [])
    assert all(not b["earned"] for b in result)
