# PathFinder — AI-Powered Personalized Learning Path Recommender

## 1. Problem Understanding & Solution Approach

Online learning platforms offer thousands of courses across diverse domains. While traditional recommendation systems can suggest relevant individual courses, learners often struggle to identify the *right sequence* of learning resources needed to achieve a specific long-term goal. Different learners have different skill levels, interests, career aspirations, and learning preferences, making a one-size-fits-all approach ineffective.

**The Solution: PathFinder**
PathFinder is a conversational learning assistant that bridges this gap. It turns a learner's natural-language goal into a **personalized, prerequisite-safe, milestone-based roadmap**. 

While generic LLMs can hallucinate resources and ignore necessary prerequisite ordering, PathFinder cleanly separates these responsibilities:
1. **Natural Language Understanding**: Uses LLMs to capture learner profiles (goals, experience, pace).
2. **Deterministic Graph Constraints**: Uses topological sorting and a DAG (Directed Acyclic Graph) solver to enforce strict prerequisite ordering.
3. **Adaptive Feedback**: Incorporates learner feedback (thumbs up/down) to adjust recommendations dynamically.

## 2. System Architecture

The architecture relies on a hybrid approach, combining multi-agent orchestration with semantic retrieval and graph-based validation.

- **Streamlit UI**: A polished interface providing onboarding, a personalized path view, progress dashboard, and an AI advisory chat.
- **Orchestrator Agent**: Manages the flow between specialized agents.
- **Profiler Agent**: Extracts structured profile data (interests, experience level, known skills) from the learner's natural language input.
- **Recommender Agent & Semantic Engine**: Retrieves relevant catalog items using OpenAI embeddings (or TF-IDF fallback) and ranks them using a customized logistic scoring model based on semantic relevance, experience fit, and time constraints.
- **Prerequisite DAG & Validator**: Computes transitive closures and strictly validates the learning path against cyclic dependencies and prerequisites.
- **Advisor Agent**: A grounded LLM that provides explanations for why specific courses were recommended, without hallucinating.

## 3. AI/ML Implementation

PathFinder implements a hybrid "cold-start" recommendation strategy that requires no proprietary training corpus:

1. **Structured Learner Extraction**: Uses LLM APIs (e.g., `gpt-4o-mini`) for structured data extraction.
2. **Semantic Retrieval**: Uses `text-embedding-3-small` for vector representations of courses. An in-memory FAISS HNSW index ensures lightning-fast similarity search. If offline, it gracefully falls back to a TF-IDF + cosine-similarity vector-space model.
3. **Personalized Ranking Algorithm**: The recommender mathematically weights several factors:
   - Semantic goal/course relevance
   - Experience-level appropriateness
   - Known-skill avoidance (to prevent redundant recommendations)
   - Learner interest matching
   - Historical feedback signals
4. **CostWeightedAStar Planner**: Dynamically calculates the path of least resistance through the curriculum DAG, accounting for the learner's assessed baseline proficiency.

## 4. Key Features & Workflows

- **Conversational Onboarding**: Natural language goal setting (e.g., "I want to become a data scientist...").
- **Prerequisite-Safe Curriculum**: Guarantees that the generated path makes pedagogical sense and is logically sound.
- **Grounded AI Explanations**: "Why this is here" rationale for every recommended step.
- **Skill Intelligence Dashboard**: Visualizes progress, skill coverage, role-readiness, and biggest gaps using radar charts and metrics.
- **Adaptive Skill Check & Re-personalization**: Difficulty-adaptive questions that test proficiency, allowing the system to restructure the roadmap with one click if the user demonstrates mastery of foundational concepts.
- **Gamification**: Badges for completion, consistency, and providing feedback.
- **Offline-Safe**: Fully functional even when internet or API keys are temporarily unavailable.

## 5. Challenges Faced and Addressed

- **Hallucination of Curricula**: Overcome by strictly enforcing the `PrerequisiteDAGSolver` layer. The LLM handles human interaction, but the deterministic algorithm builds the path.
- **Performance at Scale**: Overcome by normalizing catalog embeddings into an in-memory ANN (Approximate Nearest Neighbors) FAISS index, eliminating full catalog scans on every query.
- **Cold-Start Problem**: Addressed by relying on semantic alignment and explicit user onboarding rather than requiring millions of historical user interactions to generate the initial path.
