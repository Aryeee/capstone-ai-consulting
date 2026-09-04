"""
Pre-compute case embeddings and export everything as a single JSON file
that the Next.js frontend can import statically.

Run once:  python scripts/precompute_embeddings.py
Output:    vercel-app/public/data/cases_with_embeddings.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer
import numpy as np

DATA_IN = ROOT / "data" / "cases.json"
DATA_OUT = ROOT / "vercel-app" / "public" / "data" / "cases_with_embeddings.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print(f"Loading model '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading cases from {DATA_IN} …")
    with open(DATA_IN, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Build the same concatenated text used in the Streamlit version
    texts = []
    for c in cases:
        challenges = ", ".join(c["key_challenges"])
        text = (
            f"{c['title']}. "
            f"Industry: {c['industry']}. "
            f"Problem: {c['problem_description']} "
            f"Key challenges: {challenges}. "
            f"Actions taken: {c['actions_taken']}"
        )
        texts.append(text)

    print(f"Encoding {len(texts)} cases …")
    embeddings = model.encode(texts, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True)

    # Attach embeddings to each case
    for case, emb in zip(cases, embeddings):
        case["embedding"] = [round(float(x), 6) for x in emb]

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_OUT, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print(f"Done. Wrote {len(cases)} cases with {embeddings.shape[1]}-dim embeddings to {DATA_OUT}")


if __name__ == "__main__":
    main()
