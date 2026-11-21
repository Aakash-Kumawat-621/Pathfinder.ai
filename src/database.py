"""SQLite persistence layer for learner profiles, paths, progress, feedback and chat."""
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

DB_PATH = os.environ.get(
    "PATHFINDER_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "pathfinder.db"),
)
VALID_STATUS = {"not_started", "in_progress", "completed"}
VALID_RATING = {"up", "down"}
ALLOWED_LEARNER_FIELDS = {
    "name", "goal_text", "target_domain", "experience_level", "interests",
    "known_skills", "completed_courses", "time_commitment",
}


class _RecordModel(BaseModel):
    """Pydantic-validated DB record with dict-compatible access for existing services."""
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.model_dump().keys()

    def items(self):
        return self.model_dump().items()

    def __iter__(self) -> Iterator:
        return iter(self.model_dump().items())


class LearnerRecord(_RecordModel):
    id: int
    name: str
    goal_text: str
    target_domain: str
    experience_level: str
    interests: list[str] = Field(default_factory=list)
    known_skills: list[str] = Field(default_factory=list)
    completed_courses: list[str] = Field(default_factory=list)
    time_commitment: str = "moderate"
    created_at: str
    feedback: list[Any] = Field(default_factory=list)
    assessment_skill_scores: dict[str, int] = Field(default_factory=dict)


class LearningPathRecord(_RecordModel):
    id: int
    learner_id: int
    course_id: str
    step_order: int = Field(ge=1)
    milestone: str = ""
    status: str = "not_started"
    explanation: str = ""
    updated_at: str

    @field_validator("status")
    @classmethod
    def valid_status(cls, v):
        if v not in VALID_STATUS:
            raise ValueError("invalid learning path status")
        return v


class FeedbackRecord(_RecordModel):
    id: int
    learner_id: int
    course_id: str
    rating: str
    comment: str = ""
    created_at: str

    @field_validator("rating")
    @classmethod
    def valid_rating(cls, v):
        if v not in VALID_RATING:
            raise ValueError("invalid feedback rating")
        return v


class ChatRecord(_RecordModel):
    id: int | None = None
    learner_id: int | None = None
    role: str
    content: str
    created_at: str | None = None

    @field_validator("role")
    @classmethod
    def valid_role(cls, v):
        if v not in {"user", "assistant"}:
            raise ValueError("invalid chat role")
        return v


class AssessmentRecord(_RecordModel):
    id: int
    learner_id: int
    domain: str
    score: int = Field(ge=0)
    total: int = Field(ge=0)
    pct: float = Field(ge=0, le=100)
    skill_scores: dict[str, int] = Field(default_factory=dict)
    created_at: str


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_conn():
    parent = os.path.dirname(os.path.abspath(DB_PATH))
    os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS learners (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            goal_text TEXT NOT NULL,
            target_domain TEXT NOT NULL,
            experience_level TEXT NOT NULL,
            interests TEXT NOT NULL DEFAULT '[]',
            known_skills TEXT NOT NULL DEFAULT '[]',
            completed_courses TEXT NOT NULL DEFAULT '[]',
            time_commitment TEXT NOT NULL DEFAULT 'moderate',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id INTEGER NOT NULL REFERENCES learners(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user','assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS learning_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id INTEGER NOT NULL REFERENCES learners(id) ON DELETE CASCADE,
            course_id TEXT NOT NULL,
            step_order INTEGER NOT NULL,
            milestone TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'not_started' CHECK(status IN ('not_started','in_progress','completed')),
            explanation TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            UNIQUE(learner_id, course_id)
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learner_id INTEGER NOT NULL REFERENCES learners(id) ON DELETE CASCADE,
            course_id TEXT NOT NULL,
            rating TEXT NOT NULL CHECK(rating IN ('up','down')),
            comment TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_path_learner_order ON learning_paths(learner_id, step_order);
        CREATE INDEX IF NOT EXISTS idx_feedback_learner ON feedback(learner_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_chat_learner ON chat_history(learner_id, id);
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, learner_id INTEGER NOT NULL REFERENCES learners(id) ON DELETE CASCADE,
            domain TEXT NOT NULL, score INTEGER NOT NULL, total INTEGER NOT NULL, pct REAL NOT NULL,
            skill_scores TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_assessment_learner ON assessments(learner_id, created_at);
    """)
    conn.commit()
    conn.close()


def create_learner(name, goal_text, target_domain, experience_level,
                   interests, known_skills, completed_courses, time_commitment):
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO learners
        (name, goal_text, target_domain, experience_level, interests, known_skills, completed_courses, time_commitment, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(name).strip() or "Learner", str(goal_text).strip(), target_domain, experience_level,
        json.dumps(interests or []), json.dumps(known_skills or []),
        json.dumps(completed_courses or []), time_commitment or "moderate", _now(),
    ))
    conn.commit()
    learner_id = cur.lastrowid
    conn.close()
    return learner_id


def get_learner(learner_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM learners WHERE id = ?", (learner_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    data = dict(row)
    for field in ("interests", "known_skills", "completed_courses"):
        try:
            data[field] = json.loads(data[field]) if data[field] else []
        except (TypeError, json.JSONDecodeError):
            data[field] = []
    return LearnerRecord.model_validate(data)


def update_learner_field(learner_id, field, value):
    if field not in ALLOWED_LEARNER_FIELDS:
        raise ValueError(f"Unsupported learner field: {field}")
    if isinstance(value, (list, dict)):
        value = json.dumps(value)
    conn = get_conn()
    conn.execute(f"UPDATE learners SET {field} = ? WHERE id = ?", (value, learner_id))
    conn.commit()
    conn.close()


def add_chat_message(learner_id, role, content):
    if role not in ("user", "assistant"):
        raise ValueError("role must be user or assistant")
    conn = get_conn()
    conn.execute("INSERT INTO chat_history(learner_id, role, content, created_at) VALUES(?,?,?,?)",
                 (learner_id, role, str(content), _now()))
    conn.commit()
    conn.close()


def get_chat_history(learner_id):
    conn = get_conn()
    rows = conn.execute("SELECT role, content FROM chat_history WHERE learner_id=? ORDER BY id", (learner_id,)).fetchall()
    conn.close()
    return [ChatRecord.model_validate(dict(r)) for r in rows]


def save_learning_path(learner_id, path_items):
    """Replace path structure atomically while preserving progress for unchanged courses."""
    conn = get_conn()
    try:
        conn.execute("BEGIN")
        old = {
            r["course_id"]: dict(r)
            for r in conn.execute("SELECT * FROM learning_paths WHERE learner_id=?", (learner_id,)).fetchall()
        }
        conn.execute("DELETE FROM learning_paths WHERE learner_id=?", (learner_id,))
        now = _now()
        for item in path_items:
            cid = item["course_id"]
            previous = old.get(cid, {})
            status = previous.get("status", item.get("status", "not_started"))
            if status not in VALID_STATUS:
                status = "not_started"
            conn.execute("""
                INSERT INTO learning_paths
                (learner_id, course_id, step_order, milestone, status, explanation, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (learner_id, cid, int(item["step_order"]), item.get("milestone", ""),
                  status, item.get("explanation", ""), previous.get("updated_at", now)))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_learning_path(learner_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM learning_paths WHERE learner_id=? ORDER BY step_order", (learner_id,)).fetchall()
    conn.close()
    return [LearningPathRecord.model_validate(dict(r)) for r in rows]


def update_progress(learner_id, course_id, status):
    if status not in VALID_STATUS:
        raise ValueError(f"Invalid status: {status}")
    conn = get_conn()
    conn.execute("UPDATE learning_paths SET status=?, updated_at=? WHERE learner_id=? AND course_id=?",
                 (status, _now(), learner_id, course_id))
    conn.commit()
    conn.close()


def add_feedback(learner_id, course_id, rating, comment=""):
    if rating not in VALID_RATING:
        raise ValueError("rating must be up or down")
    conn = get_conn()
    conn.execute("INSERT INTO feedback(learner_id,course_id,rating,comment,created_at) VALUES(?,?,?,?,?)",
                 (learner_id, course_id, rating, str(comment or "").strip()[:500], _now()))
    conn.commit()
    conn.close()


def get_feedback(learner_id):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM feedback WHERE learner_id=? ORDER BY id", (learner_id,)).fetchall()
    conn.close()
    return [FeedbackRecord.model_validate(dict(r)) for r in rows]


def list_learners():
    conn = get_conn()
    rows = conn.execute("SELECT id,name,goal_text,target_domain,created_at FROM learners ORDER BY id DESC").fetchall()
    conn.close()
    return [LearnerRecord.model_validate(dict(r)) for r in rows]


def add_assessment(learner_id, domain, result):
    conn=get_conn()
    conn.execute("INSERT INTO assessments(learner_id,domain,score,total,pct,skill_scores,created_at) VALUES(?,?,?,?,?,?,?)",
                 (learner_id,domain,int(result.get("score",0)),int(result.get("total",0)),float(result.get("pct",0)),json.dumps(result.get("skill_scores",{})),_now()))
    conn.commit(); conn.close()

def get_latest_assessment(learner_id):
    conn=get_conn(); row=conn.execute("SELECT * FROM assessments WHERE learner_id=? ORDER BY id DESC LIMIT 1",(learner_id,)).fetchone(); conn.close()
    if not row: return None
    d=dict(row)
    try: d["skill_scores"]=json.loads(d["skill_scores"])
    except Exception: d["skill_scores"]={}
    return AssessmentRecord.model_validate(d)
