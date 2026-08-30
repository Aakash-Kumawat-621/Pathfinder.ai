from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

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


class PathItem(BaseModel):
    step: int
    course: str
    similarity_score: float
    rationale: str
    milestone: str
    status: str = "not_started"


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
            CourseRow("cloud", "Cloud Foundations", "beginner cloud fundamentals aws gcp azure"),
            CourseRow("cloud", "AWS Practitioner", "aws cloud practitioner identity networking storage"),
            CourseRow("cloud", "DevOps on Cloud", "ci cd docker kubernetes cloud deployment"),
            CourseRow("machine learning", "ML Foundations", "machine learning basics supervised unsupervised"),
            CourseRow("machine learning", "Python for Data Science", "python pandas numpy data analysis"),
            CourseRow("machine learning", "Deep Learning Basics", "neural network pytorch tensorflow"),
            CourseRow("web development", "HTML CSS JavaScript", "frontend html css javascript beginner"),
            CourseRow("web development", "React Development", "react hooks components routing frontend"),
            CourseRow("web development", "Backend APIs", "fastapi flask rest api databases"),
            CourseRow("cybersecurity", "Security Fundamentals", "network security threat modeling auth basics"),
            CourseRow("cybersecurity", "Ethical Hacking", "penetration testing kali reconnaissance"),
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
        mask = np.array([c.track == track for c in self.courses])
        if not mask.any():
            return track, []

        sims = cosine_similarity(query_vec, self.course_matrix[mask]).flatten()
        ranked_local_idx = np.argsort(-sims)[:top_k]
        track_courses = [course for course in self.courses if course.track == track]
        milestones = ["Foundations", "Core Skills", "Advanced Projects"]

        result: list[PathItem] = []
        for i, local_idx in enumerate(ranked_local_idx, start=1):
            milestone = milestones[min((i - 1) * len(milestones) // max(len(ranked_local_idx), 1), len(milestones) - 1)]
            course = track_courses[int(local_idx)]
            result.append(
                PathItem(
                    step=i,
                    course=course.course,
                    similarity_score=round(float(sims[int(local_idx)]), 4),
                    rationale=f"Matches your goal using {track} concepts found in your query.",
                    milestone=milestone,
                )
            )
        return track, result


engine = RecommendationEngine()
app = FastAPI(title="PathFinder API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    text = req.message.lower()
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
        goal_statement=req.message.strip(),
        skill_level=skill_level,
        interests=interests,
        completed_courses=[],
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    track, path = engine.recommend(req.goal_statement, top_k=req.top_k)
    return RecommendResponse(predicted_track=track, recommended_path=path)


@app.post("/path", response_model=RecommendResponse)
def path(req: PathRequest) -> RecommendResponse:
    ordered = sorted(req.recommended_path, key=lambda item: item.step)
    return RecommendResponse(
        predicted_track="custom",
        recommended_path=ordered,
    )


@app.post("/explain", response_model=RecommendResponse)
def explain(req: ExplainRequest) -> RecommendResponse:
    explained: list[PathItem] = []
    for item in req.recommended_path:
        explained.append(
            PathItem(
                **item.model_dump(),
                rationale=(
                    f"{item.course} is recommended because it aligns with your goal "
                    f"'{req.goal_statement}' and has relevance score {item.similarity_score:.2f}."
                ),
            )
        )
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
