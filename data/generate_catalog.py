# NOTE: The checked-in courses.csv uses curated first-party/educational resource URLs for the hackathon demo.
# Regenerating this file may restore generated URLs; prefer editing courses.csv for submission demos.
"""
Generates data/courses.csv — the learning resource catalog.
Each row is a learning item (course / project / assessment) with explicit
prerequisites so the path generator can build a real dependency graph.
"""
import csv
import os

items = [
    # course_id, title, domain, skill_tag, level, prereq_ids(;), duration_hrs, type, description, resource_url

    # ---------------- DATA SCIENCE / AI-ML DOMAIN ----------------
    ("py101", "Python Programming Fundamentals", "Data Science", "python", "beginner", "", 20, "course",
     "Variables, control flow, functions, data structures in Python.", "https://example.com/py101"),
    ("py102", "Python for Data Analysis (Pandas & NumPy)", "Data Science", "data_wrangling", "beginner", "py101", 15, "course",
     "Hands-on data manipulation using pandas and numpy.", "https://example.com/py102"),
    ("stats101", "Statistics & Probability Foundations", "Data Science", "statistics", "beginner", "", 18, "course",
     "Descriptive stats, distributions, hypothesis testing.", "https://example.com/stats101"),
    ("viz101", "Data Visualization with Matplotlib/Seaborn", "Data Science", "data_viz", "beginner", "py102", 10, "course",
     "Communicate insights visually from tabular data.", "https://example.com/viz101"),
    ("proj_eda", "Project: Exploratory Data Analysis on a Real Dataset", "Data Science", "data_wrangling", "beginner", "viz101;stats101", 8, "project",
     "Apply EDA techniques end-to-end on a public dataset and present findings.", "https://example.com/proj_eda"),
    ("ml101", "Machine Learning Foundations", "Data Science", "machine_learning", "intermediate", "stats101;py102", 25, "course",
     "Supervised/unsupervised learning, regression, classification, evaluation metrics.", "https://example.com/ml101"),
    ("assess_ml101", "Assessment: ML Foundations Quiz", "Data Science", "machine_learning", "intermediate", "ml101", 1, "assessment",
     "Test your understanding of core ML algorithms and metrics.", "https://example.com/assess_ml101"),
    ("proj_ml_pred", "Project: Build a Predictive Model End-to-End", "Data Science", "machine_learning", "intermediate", "assess_ml101", 12, "project",
     "Train, evaluate and deploy a classification/regression model on real data.", "https://example.com/proj_ml_pred"),
    ("dl101", "Deep Learning Foundations (Neural Networks)", "Data Science", "deep_learning", "advanced", "ml101", 30, "course",
     "Feedforward nets, backprop, CNNs, RNNs using PyTorch/TensorFlow.", "https://example.com/dl101"),
    ("nlp101", "Natural Language Processing Basics", "Data Science", "nlp", "advanced", "dl101", 20, "course",
     "Text preprocessing, embeddings, transformers, and LLM basics.", "https://example.com/nlp101"),
    ("proj_capstone_ds", "Capstone Project: End-to-End AI Application", "Data Science", "machine_learning", "advanced", "nlp101;proj_ml_pred", 25, "project",
     "Design, build and deploy a full AI-powered application.", "https://example.com/proj_capstone_ds"),

    # ---------------- WEB DEVELOPMENT DOMAIN ----------------
    ("html101", "HTML & CSS Fundamentals", "Web Development", "frontend_basics", "beginner", "", 12, "course",
     "Structure and style web pages from scratch.", "https://example.com/html101"),
    ("js101", "JavaScript Essentials", "Web Development", "javascript", "beginner", "html101", 18, "course",
     "Core JS syntax, DOM manipulation, events, async programming.", "https://example.com/js101"),
    ("react101", "React.js for Building UIs", "Web Development", "frontend_framework", "intermediate", "js101", 22, "course",
     "Components, hooks, state management, routing.", "https://example.com/react101"),
    ("node101", "Node.js & Express Backend Development", "Web Development", "backend", "intermediate", "js101", 20, "course",
     "Build REST APIs with Node, Express, and middleware.", "https://example.com/node101"),
    ("db101", "Databases: SQL & NoSQL Fundamentals", "Web Development", "databases", "intermediate", "", 14, "course",
     "Relational modelling with SQL and document modelling with MongoDB.", "https://example.com/db101"),
    ("proj_fullstack", "Project: Full-Stack Web Application", "Web Development", "backend", "intermediate", "react101;node101;db101", 20, "project",
     "Build and deploy a complete full-stack CRUD application.", "https://example.com/proj_fullstack"),
    ("assess_web_int", "Assessment: Full-Stack Fundamentals", "Web Development", "backend", "intermediate", "proj_fullstack", 1, "assessment",
     "Validate your full-stack development skills.", "https://example.com/assess_web_int"),
    ("cloud_deploy101", "Deploying Web Apps to the Cloud", "Web Development", "devops", "advanced", "assess_web_int", 10, "course",
     "CI/CD basics and deployment to cloud platforms.", "https://example.com/cloud_deploy101"),
    ("proj_capstone_web", "Capstone Project: Production-Ready Web Platform", "Web Development", "backend", "advanced", "cloud_deploy101", 25, "project",
     "Ship a scalable, deployed, production-grade web platform.", "https://example.com/proj_capstone_web"),

    # ---------------- CLOUD / DEVOPS DOMAIN ----------------
    ("cloud101", "Cloud Computing Fundamentals (AWS/Azure/GCP)", "Cloud & DevOps", "cloud_basics", "beginner", "", 15, "course",
     "Core cloud concepts: compute, storage, networking, IAM.", "https://example.com/cloud101"),
    ("linux101", "Linux & Shell Scripting Essentials", "Cloud & DevOps", "linux", "beginner", "", 12, "course",
     "Command line proficiency and automation with shell scripts.", "https://example.com/linux101"),
    ("docker101", "Containers with Docker", "Cloud & DevOps", "containers", "intermediate", "linux101", 14, "course",
     "Build, ship and run containerized applications.", "https://example.com/docker101"),
    ("k8s101", "Container Orchestration with Kubernetes", "Cloud & DevOps", "orchestration", "advanced", "docker101;cloud101", 20, "course",
     "Deploy and scale containerized workloads with Kubernetes.", "https://example.com/k8s101"),
    ("cicd101", "CI/CD Pipelines", "Cloud & DevOps", "devops", "advanced", "docker101", 12, "course",
     "Automate build/test/deploy pipelines with GitHub Actions/Jenkins.", "https://example.com/cicd101"),
    ("proj_capstone_cloud", "Capstone Project: Automated Cloud Deployment Pipeline", "Cloud & DevOps", "devops", "advanced", "k8s101;cicd101", 20, "project",
     "Design a full CI/CD pipeline deploying to Kubernetes on the cloud.", "https://example.com/proj_capstone_cloud"),

    # ---------------- CYBERSECURITY DOMAIN ----------------
    ("sec101", "Cybersecurity Fundamentals", "Cybersecurity", "security_basics", "beginner", "", 15, "course",
     "CIA triad, threat landscape, common attack vectors.", "https://example.com/sec101"),
    ("netsec101", "Networking & Network Security Basics", "Cybersecurity", "network_security", "beginner", "sec101", 16, "course",
     "TCP/IP, firewalls, VPNs, and network defense fundamentals.", "https://example.com/netsec101"),
    ("pentest101", "Ethical Hacking & Penetration Testing Basics", "Cybersecurity", "pentesting", "intermediate", "netsec101", 22, "course",
     "Reconnaissance, vulnerability scanning, exploitation basics in a lab.", "https://example.com/pentest101"),
    ("proj_ctf", "Project: Capture-The-Flag Challenge Set", "Cybersecurity", "pentesting", "intermediate", "pentest101", 10, "project",
     "Solve a set of CTF-style security challenges.", "https://example.com/proj_ctf"),
    ("proj_capstone_sec", "Capstone Project: Security Audit of a Sample Application", "Cybersecurity", "pentesting", "advanced", "proj_ctf", 18, "project",
     "Perform and document a full security audit with remediation plan.", "https://example.com/proj_capstone_sec"),

    # ---------------- PRODUCT MANAGEMENT DOMAIN ----------------
    ("pm101", "Product Management Fundamentals", "Product Management", "pm_basics", "beginner", "", 12, "course",
     "Product lifecycle, discovery, roadmapping fundamentals.", "https://example.com/pm101"),
    ("ux101", "UX Research & Design Basics", "Product Management", "ux", "beginner", "", 14, "course",
     "User research, personas, wireframing, usability testing.", "https://example.com/ux101"),
    ("analytics101", "Product Analytics & Metrics", "Product Management", "analytics", "intermediate", "pm101", 12, "course",
     "Define and track north-star metrics, funnels, and experiments.", "https://example.com/analytics101"),
    ("proj_prd", "Project: Write a Product Requirements Document", "Product Management", "pm_basics", "intermediate", "pm101;ux101", 8, "project",
     "Produce a complete PRD for a new feature or product.", "https://example.com/proj_prd"),
    ("proj_capstone_pm", "Capstone Project: Product Launch Case Study", "Product Management", "pm_basics", "advanced", "proj_prd;analytics101", 15, "project",
     "Plan and simulate a full product launch with metrics strategy.", "https://example.com/proj_capstone_pm"),

    # ---------------- DIGITAL MARKETING DOMAIN ----------------
    ("mkt101", "Digital Marketing Fundamentals", "Digital Marketing", "marketing_basics", "beginner", "", 10, "course",
     "Channels, funnels, positioning, and campaign basics.", "https://example.com/mkt101"),
    ("seo101", "SEO & Content Marketing", "Digital Marketing", "seo", "beginner", "mkt101", 10, "course",
     "Keyword research, on-page SEO, and content strategy.", "https://example.com/seo101"),
    ("ads101", "Performance Marketing & Paid Ads", "Digital Marketing", "paid_ads", "intermediate", "mkt101", 12, "course",
     "Google/Meta ads, targeting, budgeting, and optimization.", "https://example.com/ads101"),
    ("proj_campaign", "Project: Design an Integrated Marketing Campaign", "Digital Marketing", "marketing_basics", "intermediate", "seo101;ads101", 10, "project",
     "Plan a multi-channel campaign with budget and KPIs.", "https://example.com/proj_campaign"),
    ("proj_capstone_mkt", "Capstone Project: Full Go-To-Market Strategy", "Digital Marketing", "marketing_basics", "advanced", "proj_campaign", 15, "project",
     "Build a complete GTM strategy for a product launch.", "https://example.com/proj_capstone_mkt"),
]

header = ["course_id", "title", "domain", "skill_tag", "level", "prerequisites",
          "duration_hours", "type", "description", "resource_url"]

def write_catalog():
    out = os.path.join(os.path.dirname(__file__), "courses.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(items)
    print(f"Wrote {len(items)} catalog items to {out}.")
    return out

# ---- Build-time semantic index -------------------------------------------------
def build_semantic_index():
    """Pre-serialize catalog vectors + ANN index so Streamlit has no catalog cold-start."""
    import hashlib, json, os
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
    try:
        import faiss
    except ImportError:
        faiss = None
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "courses.csv")
    df = pd.read_csv(csv_path).fillna("")
    aliases = {
        "Data Science": "data scientist machine learning artificial intelligence analytics data analysis statistics predictive modeling",
        "Web Development": "web developer frontend backend full stack javascript react websites APIs UI",
        "Cloud & DevOps": "cloud engineer devops aws azure gcp docker kubernetes infrastructure deployment ci cd",
        "Cybersecurity": "cyber security ethical hacking penetration testing network defense vulnerability SOC",
        "Product Management": "product manager product owner roadmap PRD UX user research product analytics",
        "Digital Marketing": "digital marketing SEO content social media growth paid advertising campaigns",
    }
    def text(row):
        return f"{row['title']}. {row['description']} Domain: {row['domain']}. Skills: {row['skill_tag']}. Related terms: {aliases.get(str(row['domain']), '')}."
    texts=[text(r) for _,r in df.iterrows()]
    # Offline-safe deterministic embedding fallback. If OPENAI_API_KEY exists, use true embeddings.
    vectors=None; backend="tfidf"
    key=os.environ.get("OPENAI_API_KEY")
    if key:
        try:
            from openai import OpenAI
            resp=OpenAI(api_key=key).embeddings.create(model="text-embedding-3-small", input=texts)
            vectors=np.asarray([x.embedding for x in resp.data],dtype="float32")
            backend="faiss" if faiss is not None else "lsh"
        except Exception:
            vectors=None
    if vectors is None:
        vec=TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)
        vectors=vec.fit_transform(texts).toarray().astype("float32")
        norms=np.linalg.norm(vectors,axis=1,keepdims=True); vectors/=np.maximum(norms,1e-12)
        joblib.dump(vec, os.path.join(base,"catalog_tfidf_vectorizer.joblib"))
        backend="faiss" if faiss is not None else "tfidf"
    vectors=np.asarray(vectors,dtype="float32")
    np.savez_compressed(os.path.join(base,"catalog_embeddings.npz"), vectors=vectors)
    bits=18
    if faiss is not None:
        faiss.normalize_L2(vectors)
        idx=faiss.IndexHNSWFlat(vectors.shape[1],32,faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction=80; idx.hnsw.efSearch=48; idx.add(vectors)
        faiss.write_index(idx, os.path.join(base,"catalog_index.faiss"))
        backend="faiss"
    else:
        # Serialized dependency-free ANN index for environments without faiss-cpu.
        rng=np.random.default_rng(42)
        hp=rng.standard_normal((vectors.shape[1],bits),dtype=np.float32)
        norms=np.linalg.norm(vectors,axis=1,keepdims=True); normed=vectors/np.maximum(norms,1e-12)
        signs=(normed @ hp)>=0
        packed=np.packbits(signs,axis=1,bitorder="little")
        signatures=np.array([int.from_bytes(r.tobytes(),"little") for r in packed],dtype=np.uint32)
        np.savez_compressed(os.path.join(base,"catalog_ann_lsh.npz"), hyperplanes=hp, signatures=signatures)
        backend="lsh"
    ids=[str(x) for x in df.course_id.tolist()]
    fingerprint=hashlib.sha1("|".join(f"{r.course_id}:{hashlib.sha1(text(r).encode()).hexdigest()}" for _,r in df.iterrows()).encode()).hexdigest()
    with open(os.path.join(base,"catalog_index_meta.json"),"w",encoding="utf-8") as f:
        json.dump({"version":3,"backend":backend,"model":"text-embedding-3-small" if key else "tfidf-fallback","catalog_fingerprint":fingerprint,"ids":ids,"dimension":int(vectors.shape[1]),"bits":bits},f,indent=2)
    print(f"Pre-serialized semantic index: {backend}, {len(ids)} courses, dim={vectors.shape[1]}")

if __name__ == "__main__":
    write_catalog()
    build_semantic_index()
