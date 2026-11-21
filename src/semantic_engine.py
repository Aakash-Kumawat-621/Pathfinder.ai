"""Semantic retrieval for PathFinder with OpenAI-embedding and offline TF-IDF backends."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    import faiss
    _faiss_ok = True
except ImportError:
    faiss = None
    _faiss_ok = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

try:
    from openai import OpenAI
    _openai_import_ok = True
except ImportError:
    _openai_import_ok = False

EMBED_MODEL = "text-embedding-3-small"
_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
_CACHE_PATH = os.path.join(_DATA_DIR, "embeddings_cache.json")
_PRECOMPUTED_META = os.path.join(_DATA_DIR, "catalog_index_meta.json")
_PRECOMPUTED_EMB = os.path.join(_DATA_DIR, "catalog_embeddings.npz")
_PRECOMPUTED_FAISS = os.path.join(_DATA_DIR, "catalog_index.faiss")
_PRECOMPUTED_VECTORIZER = os.path.join(_DATA_DIR, "catalog_tfidf_vectorizer.joblib")
_PRECOMPUTED_LSH = os.path.join(_DATA_DIR, "catalog_ann_lsh.npz")

DOMAIN_ALIASES = {
    "Data Science": "data scientist machine learning artificial intelligence analytics data analysis statistics predictive modeling",
    "Web Development": "web developer frontend backend full stack javascript react websites APIs UI",
    "Cloud & DevOps": "cloud engineer devops aws azure gcp docker kubernetes infrastructure deployment ci cd",
    "Cybersecurity": "cyber security ethical hacking penetration testing network defense vulnerability SOC",
    "Product Management": "product manager product owner roadmap PRD UX user research product analytics",
    "Digital Marketing": "digital marketing SEO content social media growth paid advertising campaigns",
}


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


def _course_text(row):
    alias = DOMAIN_ALIASES.get(str(row["domain"]), "")
    return f"{row['title']}. {row['description']} Domain: {row['domain']}. Skills: {row['skill_tag']}. Related terms: {alias}."


class _TfidfIndex:
    def __init__(self, df):
        self.ids = list(df["course_id"])
        texts = [_course_text(row) for _, row in df.iterrows()]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)

    def similarity_to(self, query_text):
        q_vec = self.vectorizer.transform([str(query_text)])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        return dict(zip(self.ids, sims))


class FaissANNIndex:
    """In-memory FAISS HNSW inner-product index over normalized catalog vectors."""
    def __init__(self, ids, vectors, m=32):
        if not _faiss_ok:
            raise RuntimeError("faiss-cpu is not installed")
        self.ids = list(ids)
        self.matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(self.matrix)
        self.index = faiss.IndexHNSWFlat(self.matrix.shape[1], int(m), faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 80
        self.index.hnsw.efSearch = 48
        self.index.add(self.matrix)
    def search(self, query_vector, top_k=10):
        q=np.asarray(query_vector,dtype="float32").reshape(1,-1)
        faiss.normalize_L2(q)
        scores, indices=self.index.search(q,min(int(top_k),len(self.ids)))
        return [(self.ids[int(i)],float(s)) for s,i in zip(scores[0],indices[0]) if int(i)>=0]


class InMemoryANNIndex:
    """Dependency-free approximate cosine index using random-hyperplane LSH.

    Catalog vectors are normalized and bucketed at startup. Querying probes the
    exact bucket plus nearby Hamming buckets, then ranks only those candidates.
    This removes the normal full catalog scan from the hot recommendation path.
    """
    def __init__(self, ids, vectors, bits=18, seed=42):
        self.ids = list(ids)
        self.matrix = np.asarray(vectors, dtype=np.float32)
        if self.matrix.ndim != 2 or not len(self.ids):
            raise ValueError("ANN index requires a non-empty 2D vector matrix")
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.maximum(norms, 1e-12)
        self.bits = min(int(bits), 24)
        rng = np.random.default_rng(seed)
        self.hyperplanes = rng.standard_normal((self.matrix.shape[1], self.bits), dtype=np.float32)
        signatures = self._signatures(self.matrix)
        self.buckets = {}
        for i, sig in enumerate(signatures):
            self.buckets.setdefault(int(sig), []).append(i)

    def _signatures(self, vectors):
        signs = np.asarray(vectors) @ self.hyperplanes >= 0
        packed = np.packbits(signs, axis=1, bitorder="little")
        return np.array([int.from_bytes(row.tobytes(), "little") for row in packed], dtype=np.uint32)

    def _candidate_indices(self, signature, probes=2):
        candidates = set(self.buckets.get(int(signature), []))
        # Probe Hamming-distance-1 buckets. With a small catalog this remains cheap,
        # while avoiding an O(N·D) scan in the normal case.
        for bit in range(min(self.bits, 16)):
            candidates.update(self.buckets.get(int(signature) ^ (1 << bit), []))
        if len(candidates) < probes and len(candidates) < len(self.ids):
            # Safety fallback for sparse LSH buckets; still bounded by a small top-K sample.
            candidates.update(range(min(len(self.ids), max(16, probes))))
        return sorted(candidates)

    def search(self, query_vector, top_k=None):
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        sig = self._signatures(q)[0]
        idx = self._candidate_indices(sig, top_k or 10)
        if not idx:
            return []
        sims = (self.matrix[idx] @ q[0]).astype(float)
        order = np.argsort(-sims)
        k = len(order) if top_k is None else min(int(top_k), len(order))
        return [(self.ids[idx[i]], float(sims[i])) for i in order[:k]]


def _catalog_fingerprint(df):
    payload = "|".join(f"{r['course_id']}:{hashlib.sha1(_course_text(r).encode()).hexdigest()}" for _, r in df.iterrows())
    return hashlib.sha1(payload.encode()).hexdigest()


def _load_precomputed_index(df):
    """Load the build-time catalog ANN artifact; never regenerate catalog vectors at app startup."""
    try:
        with open(_PRECOMPUTED_META, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("catalog_fingerprint") != _catalog_fingerprint(df):
            return None
        ids = [str(x) for x in df["course_id"].tolist()]
        if meta.get("ids") != ids or not os.path.exists(_PRECOMPUTED_EMB):
            return None
        vectors = np.load(_PRECOMPUTED_EMB, allow_pickle=False)["vectors"]
        if vectors.shape[0] != len(ids):
            return None
        backend = meta.get("backend", "tfidf")
        if backend == "faiss" and _faiss_ok and os.path.exists(_PRECOMPUTED_FAISS):
            idx = faiss.read_index(_PRECOMPUTED_FAISS)
            obj = type("SerializedFAISSIndex", (), {})()
            obj.ids, obj.matrix, obj.index, obj.backend = ids, vectors.astype("float32"), idx, "faiss"
            obj.search = lambda query_vector, top_k=10: _faiss_search(obj, query_vector, top_k)
            return obj
        if backend == "lsh":
            if not os.path.exists(_PRECOMPUTED_LSH):
                return None
            l=np.load(_PRECOMPUTED_LSH, allow_pickle=False)
            return SerializedLSHIndex(ids, vectors, l["hyperplanes"], l["signatures"], int(meta.get("bits",18)))
        # TF-IDF serialized vector space is the offline-safe build artifact.
        if os.path.exists(_PRECOMPUTED_VECTORIZER):
            vec = joblib.load(_PRECOMPUTED_VECTORIZER)
            obj = _TfidfIndex.__new__(_TfidfIndex)
            obj.ids, obj.vectorizer, obj.matrix, obj.backend = ids, vec, vectors, "tfidf_precomputed"
            return obj
    except Exception:
        return None
    return None


def _faiss_search(obj, query_vector, top_k=10):
    q=np.asarray(query_vector,dtype="float32").reshape(1,-1)
    faiss.normalize_L2(q)
    scores, indices=obj.index.search(q,min(int(top_k),len(obj.ids)))
    return [(obj.ids[int(i)],float(s)) for s,i in zip(scores[0],indices[0]) if int(i)>=0]


class SerializedLSHIndex:
    """Disk-loaded random-hyperplane ANN index; no catalog rebuild at application startup."""
    def __init__(self, ids, matrix, hyperplanes, signatures, bits=18):
        self.ids = list(ids)
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.hyperplanes = np.asarray(hyperplanes, dtype=np.float32)
        self.bits = int(bits)
        self.buckets = {}
        for i, sig in enumerate(np.asarray(signatures).tolist()):
            self.buckets.setdefault(int(sig), []).append(i)
        self.backend = "lsh"

    def _signature(self, q):
        signs = (np.asarray(q, dtype=np.float32).reshape(1,-1) @ self.hyperplanes) >= 0
        packed = np.packbits(signs, axis=1, bitorder="little")
        return int.from_bytes(packed[0].tobytes(), "little")

    def search(self, query_vector, top_k=10):
        q=np.asarray(query_vector,dtype=np.float32).reshape(-1)
        q/=max(float(np.linalg.norm(q)),1e-12)
        sig=self._signature(q)
        cand=set(self.buckets.get(sig, []))
        for bit in range(min(self.bits,16)):
            cand.update(self.buckets.get(sig ^ (1<<bit), []))
        if not cand:
            cand=set(range(min(len(self.ids), max(16,int(top_k)))))
        idx=np.fromiter(cand,dtype=np.int64)
        sims=self.matrix[idx] @ q
        order=np.argsort(-sims)[:min(int(top_k),len(sims))]
        return [(self.ids[int(idx[j])], float(sims[j])) for j in order]


_ann_cache = {}


_tfidf_cache = {}


def _catalog_key(df):
    return hashlib.sha1("|".join(map(str, df["course_id"].tolist())).encode()).hexdigest()


def _get_tfidf_index(df):
    key = _catalog_key(df)
    if key not in _tfidf_cache:
        _tfidf_cache[key] = _TfidfIndex(df)
    return _tfidf_cache[key]


def _load_embedding_cache():
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_embedding_cache(cache):
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    tmp = _CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, _CACHE_PATH)


def _embed_texts(client, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _get_catalog_embeddings(client, df):
    cache = _load_embedding_cache()
    catalog = {}
    changed = False
    for _, row in df.iterrows():
        cid = str(row["course_id"])
        text = _course_text(row)
        fingerprint = hashlib.sha1(text.encode()).hexdigest()
        entry = cache.get(cid)
        if isinstance(entry, dict) and entry.get("fingerprint") == fingerprint and entry.get("model") == EMBED_MODEL:
            catalog[cid] = entry["embedding"]
        else:
            catalog[cid] = None
    missing = [cid for cid, emb in catalog.items() if emb is None]
    if missing:
        rows = {str(row["course_id"]): row for _, row in df.iterrows()}
        texts = [_course_text(rows[cid]) for cid in missing]
        embeddings = _embed_texts(client, texts)
        for cid, emb in zip(missing, embeddings):
            text = _course_text(rows[cid])
            cache[cid] = {
                "model": EMBED_MODEL,
                "fingerprint": hashlib.sha1(text.encode()).hexdigest(),
                "embedding": emb,
            }
            catalog[cid] = emb
            changed = True
    if changed:
        _save_embedding_cache(cache)
    return catalog



def _ann_key(df):
    return _catalog_key(df)


def _get_ann_index(df, client):
    """Load the build-time ANN artifact; build only when the artifact is absent/stale."""
    key = _catalog_fingerprint(df)
    if key in _ann_cache:
        return _ann_cache[key]
    index = _load_precomputed_index(df)
    if index is not None:
        _ann_cache[key] = index
        return index
    if client is None:
        raise RuntimeError("Precomputed semantic index is missing or stale")
    catalog = _get_catalog_embeddings(client, df)
    ids = [str(x) for x in df["course_id"].tolist()]
    vectors = [catalog[cid] for cid in ids]
    index = FaissANNIndex(ids, vectors) if _faiss_ok else InMemoryANNIndex(ids, vectors)
    _ann_cache[key] = index
    return index


def warm_catalog_index(df=None):
    """Precompute and retain the catalog ANN index at application startup.

    Returns the index when OpenAI embeddings are available, otherwise ``None``.
    """
    df = load_catalog_for_index(df)
    client = _get_client()
    try:
        return _get_ann_index(df, client)
    except Exception:
        return None


def load_catalog_for_index(df=None):
    if df is not None:
        return df
    import pandas as pd
    path = os.path.join(os.path.dirname(__file__), "..", "data", "courses.csv")
    return pd.read_csv(path).fillna("")

def backend_name():
    # Report the actual build-time artifact when present.
    try:
        import pandas as pd
        idx = _load_precomputed_index(pd.read_csv(os.path.join(_DATA_DIR, "courses.csv")).fillna(""))
        if idx is not None:
            return "openai_embeddings" if getattr(idx, "backend", "") == "faiss" else "tfidf_vector_space"
    except Exception:
        pass
    return ("openai_faiss_hnsw" if _faiss_ok else "openai_lsh_ann") if _get_client() is not None else "tfidf_vector_space"


def semantic_similarity_scores(query_text, df):
    client = _get_client()
    pre = _load_precomputed_index(df)
    # A build-time TF-IDF artifact must not be queried with a 1536-d OpenAI vector.
    if pre is not None and hasattr(pre, "vectorizer"):
        return pre.similarity_to(query_text)
    if client is not None:
        try:
            index = _get_ann_index(df, client)
            q_emb = _embed_texts(client, [str(query_text)])[0]
            hits = index.search(q_emb, top_k=min(16, len(index.ids)))
            scores = {cid: float(score) for cid, score in hits}
            return scores
        except Exception:
            pass
    return _get_tfidf_index(df).similarity_to(query_text)


def semantic_domain_scores(goal_text, df):
    scores = semantic_similarity_scores(goal_text, df)
    by_domain = {}
    for domain in df["domain"].unique():
        vals = sorted((scores.get(cid, 0.0) for cid in df.loc[df["domain"] == domain, "course_id"]), reverse=True)[:3]
        by_domain[domain] = float(np.mean(vals)) if vals else 0.0
    return by_domain


def best_matching_domain(goal_text, df):
    scores = semantic_domain_scores(goal_text, df)
    return max(scores, key=scores.get), scores


def related_courses(goal_text, df, exclude_ids=None, top_k=4, min_score=0.05):
    exclude = set(exclude_ids or [])
    scores = semantic_similarity_scores(goal_text, df)
    domain_scores = semantic_domain_scores(goal_text, df)
    if not domain_scores:
        return []
    best = max(domain_scores.values()) or 1e-9
    eligible = {d for d, s in domain_scores.items() if s >= 0.30 * best}
    ranked = []
    for _, row in df.iterrows():
        cid = row["course_id"]
        score = float(scores.get(cid, 0.0))
        if cid not in exclude and row["domain"] in eligible and score >= min_score:
            ranked.append((cid, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
