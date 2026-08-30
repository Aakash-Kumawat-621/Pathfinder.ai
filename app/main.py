from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx
from pydantic import BaseModel, Field, field_validator

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    goal_statement: str
    skill_level: str
    interests: list[str]
    completed_courses: list[str]


class RecommendRequest(BaseModel):
    goal_statement: str
    skill_level: str = "beginner"
    completed_courses: list[str] = Field(default_factory=list)
    top_k: int = 10

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value < 1:
            raise ValueError("top_k must be at least 1")
        return min(value, 10)


class PathItem(BaseModel):
    step: int
    course: str
    similarity_score: float
    rationale: str
    milestone: str
    status: str = "not_started"

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        allowed = {"not_started", "in_progress", "complete"}
        if value not in allowed:
            raise ValueError("status must be one of not_started, in_progress, complete")
        return value


class RecommendResponse(BaseModel):
    predicted_track: str
    recommended_path: list[PathItem]


class ExplainRequest(BaseModel):
    goal_statement: str
    recommended_path: list[PathItem]


class PathRequest(BaseModel):
    recommended_path: list[PathItem]


class ProfileRequest(BaseModel):
    learner_id: str | None = None
    goal_statement: str
    skill_level: str
    interests: list[str] = Field(default_factory=list)
    completed_courses: list[str] = Field(default_factory=list)
    predicted_track: str | None = None
    recommended_path: list[PathItem] = Field(default_factory=list)

    @field_validator("skill_level")
    @classmethod
    def validate_skill_level(cls, value: str) -> str:
        if value not in {"beginner", "intermediate", "advanced"}:
            raise ValueError("skill_level must be beginner, intermediate, or advanced")
        return value


class ProfileResponse(BaseModel):
    learner_id: str
    goal_statement: str
    skill_level: str
    interests: list[str]
    completed_courses: list[str]
    predicted_track: str | None = None
    recommended_path: list[PathItem]
    progress_pct: float


class ProgressRequest(BaseModel):
    recommended_path: list[PathItem]


class ProgressResponse(BaseModel):
    progress_pct: float
    completed_steps: int
    total_steps: int
    next_actions: list[str]


@dataclass
class CourseRow:
    track: str
    course: str
    text: str


class LLMClient:
    def __init__(self) -> None:
        self.url = os.getenv("PATHIFY_LLM_URL", "").strip()
        self.api_key = os.getenv("PATHIFY_LLM_API_KEY", "").strip()
        self.model = os.getenv("PATHIFY_LLM_MODEL", "")
        self.timeout = float(os.getenv("PATHIFY_LLM_TIMEOUT", "10"))

    def available(self) -> bool:
        return bool(self.url)

    def ask_json(self, system: str, user: str) -> dict[str, Any] | None:
        if not self.available():
            return None

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"******"

        payload = {
            "model": self.model,
            "system": system,
            "input": user,
            "response_format": "json",
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(self.url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception:
            return None

        candidates: list[str] = []
        for key in ["output_text", "text", "content", "response"]:
            val = data.get(key)
            if isinstance(val, str):
                candidates.append(val)
        if isinstance(data.get("choices"), list) and data["choices"]:
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content")
            if isinstance(content, str):
                candidates.append(content)

        for text in candidates:
            parsed = _extract_json_object(text)
            if parsed is not None:
                return parsed
        return data if isinstance(data, dict) else None


class RecommendationEngine:
    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer
        self.classifier: MultinomialNB
        self.course_matrix: Any
        self.courses: list[CourseRow]
        self._load_or_bootstrap()

    def _load_or_bootstrap(self) -> None:
        artifacts_dir = os.getenv("PATHIFY_ARTIFACTS_DIR", "artifacts")
        vectorizer_path = os.path.join(artifacts_dir, "vectorizer.joblib")
        classifier_path = os.path.join(artifacts_dir, "classifier.joblib")
        courses_path = os.path.join(artifacts_dir, "courses.joblib")
        course_matrix_path = os.path.join(artifacts_dir, "course_matrix.joblib")

        if (
            joblib
            and os.path.exists(vectorizer_path)
            and os.path.exists(classifier_path)
            and os.path.exists(courses_path)
            and os.path.exists(course_matrix_path)
        ):
            self.vectorizer = joblib.load(vectorizer_path)
            self.classifier = joblib.load(classifier_path)
            self.courses = joblib.load(courses_path)
            self.course_matrix = joblib.load(course_matrix_path)
            return

        self.courses = [
            CourseRow("cloud", "Cloud Foundations", "intro beginner cloud fundamentals aws gcp azure"),
            CourseRow("cloud", "AWS Practitioner", "aws cloud practitioner identity networking storage"),
            CourseRow("cloud", "DevOps on Cloud", "advanced ci cd docker kubernetes cloud deployment"),
            CourseRow("machine learning", "ML Foundations", "intro machine learning basics supervised unsupervised"),
            CourseRow("machine learning", "Python for Data Science", "python pandas numpy data analysis"),
            CourseRow("machine learning", "Deep Learning Basics", "advanced neural network pytorch tensorflow"),
            CourseRow("web development", "HTML CSS JavaScript", "intro frontend html css javascript beginner"),
            CourseRow("web development", "React Development", "react hooks components routing frontend"),
            CourseRow("web development", "Backend APIs", "advanced fastapi flask rest api databases"),
            CourseRow("cybersecurity", "Security Fundamentals", "intro network security threat modeling auth basics"),
            CourseRow("cybersecurity", "Ethical Hacking", "advanced penetration testing kali reconnaissance"),
            CourseRow("cybersecurity", "Cloud Security", "iam encryption cloud compliance"),
        ]

        train_texts = [
            "i want to become a cloud engineer",
            "learn aws and deployment",
            "start devops in cloud",
            "i want machine learning career",
            "learn ai with python",
            "deep learning roadmap",
            "want to be a web developer",
            "learn frontend and backend",
            "build modern web apps",
            "start cybersecurity career",
            "ethical hacking beginner",
            "security analyst roadmap",
        ]
        train_labels = [
            "cloud",
            "cloud",
            "cloud",
            "machine learning",
            "machine learning",
            "machine learning",
            "web development",
            "web development",
            "web development",
            "cybersecurity",
            "cybersecurity",
            "cybersecurity",
        ]

        self.vectorizer = TfidfVectorizer(ngram_range=(2, 3), sublinear_tf=True, min_df=1)
        x_train = self.vectorizer.fit_transform(train_texts)
        self.classifier = MultinomialNB()
        self.classifier.fit(x_train, train_labels)
        self.course_matrix = self.vectorizer.transform([c.text for c in self.courses])

    def recommend(self, query: str, top_k: int = 10) -> tuple[str, list[PathItem]]:
        query_vec = self.vectorizer.transform([query])
        track = str(self.classifier.predict(query_vec)[0])

        track_courses = [c for c in self.courses if c.track == track]
        if not track_courses:
            return track, []

        track_matrix = self.vectorizer.transform([c.text for c in track_courses])
        sims = cosine_similarity(query_vec, track_matrix).flatten()
        ranked_idx = np.argsort(-sims)[: min(top_k, len(track_courses))]

        milestones = ["Foundations", "Core Skills", "Advanced Projects"]
        result: list[PathItem] = []
        for i, idx in enumerate(ranked_idx, start=1):
            milestone = milestones[min((i - 1) * len(milestones) // max(len(ranked_idx), 1), len(milestones) - 1)]
            course = track_courses[int(idx)]
            result.append(
                PathItem(
                    step=i,
                    course=course.course,
                    similarity_score=round(float(sims[int(idx)]), 4),
                    rationale=f"Matches your goal using {track} concepts found in your query.",
                    milestone=milestone,
                )
            )

        return track, result


engine = RecommendationEngine()
llm_client = LLMClient()
app = FastAPI(title="PathFinder API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _fallback_chat_parse(message: str) -> ChatResponse:
    text = message.lower()
    interests = [
        keyword
        for keyword in ["cloud", "machine learning", "ai", "web", "cybersecurity", "devops"]
        if keyword in text
    ]
    if not interests:
        interests = ["general"]

    skill_level = "beginner"
    if any(token in text for token in ["advanced", "experienced", "expert"]):
        skill_level = "advanced"
    elif any(token in text for token in ["intermediate", "some experience"]):
        skill_level = "intermediate"

    return ChatResponse(
        goal_statement=message.strip(),
        skill_level=skill_level,
        interests=interests,
        completed_courses=[],
    )


def _difficulty_rank(course_name: str) -> int:
    text = course_name.lower()
    if any(k in text for k in ["intro", "foundation", "beginner", "basics"]):
        return 0
    if any(k in text for k in ["advanced", "expert", "hacking", "projects"]):
        return 2
    return 1


@app.get("/")
def home() -> FileResponse:
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    system = (
        "Extract a learner profile from user text and return strict JSON with keys: "
        "goal_statement, skill_level(beginner|intermediate|advanced), interests(list), completed_courses(list)."
    )
    llm = llm_client.ask_json(system=system, user=req.message)
    if llm:
        try:
            return ChatResponse(**llm)
        except Exception:
            pass
    return _fallback_chat_parse(req.message)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if not req.goal_statement.strip():
        raise HTTPException(status_code=400, detail="goal_statement cannot be empty")
    track, path = engine.recommend(req.goal_statement, top_k=req.top_k)
    return RecommendResponse(predicted_track=track, recommended_path=path)


@app.post("/path", response_model=RecommendResponse)
def path(req: PathRequest) -> RecommendResponse:
    ranked = sorted(
        req.recommended_path,
        key=lambda item: (_difficulty_rank(item.course), -item.similarity_score, item.step),
    )
    milestones = ["Foundations", "Core Skills", "Advanced Projects"]
    ordered: list[PathItem] = []
    for i, item in enumerate(ranked, start=1):
        milestone = milestones[min((i - 1) * len(milestones) // max(len(ranked), 1), len(milestones) - 1)]
        payload = item.model_dump()
        payload["step"] = i
        payload["milestone"] = milestone
        ordered.append(PathItem(**payload))
    return RecommendResponse(predicted_track="custom", recommended_path=ordered)


@app.post("/explain", response_model=RecommendResponse)
def explain(req: ExplainRequest) -> RecommendResponse:
    explained: list[PathItem] = []
    for item in req.recommended_path:
        system = (
            "Write a 1-2 sentence explanation for why a learning course is recommended. "
            "Return strict JSON with key rationale only."
        )
        user = (
            f"Goal: {req.goal_statement}\n"
            f"Course: {item.course}\n"
            f"Similarity score: {item.similarity_score}\n"
            f"Milestone: {item.milestone}"
        )
        llm = llm_client.ask_json(system=system, user=user)
        rationale = None
        if isinstance(llm, dict):
            maybe = llm.get("rationale")
            if isinstance(maybe, str) and maybe.strip():
                rationale = maybe.strip()

        payload = item.model_dump()
        payload["rationale"] = rationale or (
            f"{item.course} aligns with your goal '{req.goal_statement}' "
            f"and its relevance score is {item.similarity_score:.2f}."
        )
        explained.append(PathItem(**payload))
    return RecommendResponse(predicted_track="custom", recommended_path=explained)


@app.post("/progress", response_model=ProgressResponse)
def progress(req: ProgressRequest) -> ProgressResponse:
    total = len(req.recommended_path)
    completed = sum(1 for item in req.recommended_path if item.status == "complete")
    progress_pct = round((completed / total) * 100, 2) if total else 0.0
    next_actions = [
        f"Start: {item.course}"
        for item in req.recommended_path
        if item.status == "not_started"
    ][:3]
    return ProgressResponse(
        progress_pct=progress_pct,
        completed_steps=completed,
        total_steps=total,
        next_actions=next_actions,
    )


@app.post("/profile", response_model=ProfileResponse)
def profile(req: ProfileRequest) -> ProfileResponse:
    learner_id = req.learner_id or str(uuid.uuid4())
    total = len(req.recommended_path)
    completed = sum(1 for item in req.recommended_path if item.status == "complete")
    progress_pct = round((completed / total) * 100, 2) if total else 0.0
    return ProfileResponse(
        learner_id=learner_id,
        goal_statement=req.goal_statement,
        skill_level=req.skill_level,
        interests=req.interests,
        completed_courses=req.completed_courses,
        predicted_track=req.predicted_track,
        recommended_path=req.recommended_path,
        progress_pct=progress_pct,
    )
