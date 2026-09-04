"""
Case database: load cases from JSON, build and query the FAISS vector index.
"""

import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.schema import Case, OutcomeMetrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = _ROOT / "data" / "cases.json"
INDEX_PATH = _ROOT / "data" / "faiss.index"
META_PATH = _ROOT / "data" / "cases.pkl"  # pickled Case list (with embeddings)

# ---------------------------------------------------------------------------
# Model (lazy-loaded singleton)
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# Load raw JSON → Case objects
# ---------------------------------------------------------------------------
def load_cases_from_json(path: str | Path = DATA_PATH) -> list[Case]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cases: list[Case] = []
    for entry in raw:
        metrics = OutcomeMetrics(**entry.pop("outcome_metrics"))
        entry["outcome_metrics"] = metrics
        cases.append(Case(**entry))
    return cases


# ---------------------------------------------------------------------------
# Embed all cases
# ---------------------------------------------------------------------------
def embed_cases(cases: list[Case], batch_size: int = 32) -> np.ndarray:
    """Return (N, D) float32 matrix of case embeddings and attach them to
    each Case object."""
    model = get_model()
    texts = [c.text_for_embedding for c in cases]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)
    for case, emb in zip(cases, embeddings):
        case.embedding = emb.tolist()
    return embeddings


# ---------------------------------------------------------------------------
# Build / save / load FAISS index
# ---------------------------------------------------------------------------
def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Inner-product index (vectors are L2-normalised → IP = cosine sim)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, cases: list[Case],
               index_path: Path = INDEX_PATH, meta_path: Path = META_PATH):
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(cases, f)


def load_index(index_path: Path = INDEX_PATH,
               meta_path: Path = META_PATH) -> tuple[faiss.IndexFlatIP, list[Case]]:
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        cases = pickle.load(f)
    return index, cases


# ---------------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, index: faiss.IndexFlatIP, cases: list[Case],
             top_k: int = 5) -> list[tuple[Case, float]]:
    """Embed *query*, search the FAISS index, return top-k (Case, score) pairs."""
    model = get_model()
    q_emb = model.encode([query], normalize_embeddings=True,
                         convert_to_numpy=True).astype(np.float32)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((cases[idx], float(score)))
    return results


# ---------------------------------------------------------------------------
# One-shot: build everything from scratch
# ---------------------------------------------------------------------------
def build_all():
    """Load cases → embed → build index → persist."""
    print("Loading cases …")
    cases = load_cases_from_json()
    print(f"  {len(cases)} cases loaded.")

    print("Embedding cases …")
    embeddings = embed_cases(cases)
    print(f"  Embedding matrix shape: {embeddings.shape}")

    print("Building FAISS index …")
    index = build_index(embeddings)

    print("Saving index and metadata …")
    save_index(index, cases)
    print("Done ✓")
    return index, cases


if __name__ == "__main__":
    build_all()
