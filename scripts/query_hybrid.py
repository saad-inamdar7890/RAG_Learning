import argparse
import json
import os
import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "faiss is required. Install with: python -m pip install faiss-cpu"
    ) from exc

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_DIR = os.path.join(BASE_DIR, "data", "artifacts", "index")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def load_metadata(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_bm25(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    return data


def bm25_scores(bm25_data, query_tokens: List[str]) -> np.ndarray:
    # Recompute BM25 scores from stored idf/avgdl and corpus tokens
    idf = bm25_data["idf"]
    avgdl = bm25_data["avgdl"]
    corpus_tokens = bm25_data["corpus_tokens"]

    k1 = 1.5
    b = 0.75

    scores = []
    for doc in corpus_tokens:
        doc_len = len(doc)
        tf = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for q in query_tokens:
            if q not in tf:
                continue
            numerator = tf[q] * (k1 + 1)
            denominator = tf[q] + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf.get(q, 0.0) * (numerator / denominator)
        scores.append(score)
    return np.array(scores, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid BM25 + FAISS search.")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--topk", type=int, default=5, help="Number of results")
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for vector score")
    args = parser.parse_args()

    faiss_path = os.path.join(INDEX_DIR, "faiss.index")
    faiss_meta = os.path.join(INDEX_DIR, "chunks.jsonl")
    bm25_path = os.path.join(INDEX_DIR, "bm25.json")
    bm25_meta = os.path.join(INDEX_DIR, "bm25_chunks.jsonl")
    info_path = os.path.join(INDEX_DIR, "index_info.json")

    if not os.path.exists(faiss_path) or not os.path.exists(faiss_meta):
        print("FAISS index not found. Run build_faiss_index.py first.")
        return 1
    if not os.path.exists(bm25_path) or not os.path.exists(bm25_meta):
        print("BM25 index not found. Run build_bm25_index.py first.")
        return 1

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.loads(f.read().strip())
        model_name = info.get("model", model_name)

    embedder = SentenceTransformer(model_name)
    query_vec = embedder.encode([args.query], normalize_embeddings=True)

    index = faiss.read_index(faiss_path)
    vec_scores, vec_indices = index.search(
        np.asarray(query_vec, dtype=np.float32), 50
    )

    bm25_data = load_bm25(bm25_path)
    bm25_sc = bm25_scores(bm25_data, tokenize(args.query))

    # Normalize scores to 0..1 range
    vec_sc = vec_scores[0]
    if vec_sc.max() > vec_sc.min():
        vec_sc = (vec_sc - vec_sc.min()) / (vec_sc.max() - vec_sc.min())
    bm25_sc = bm25_sc
    if bm25_sc.max() > bm25_sc.min():
        bm25_sc = (bm25_sc - bm25_sc.min()) / (bm25_sc.max() - bm25_sc.min())

    # Build combined scores for all docs using vector candidates + bm25
    combined = {}
    for rank, idx in enumerate(vec_indices[0]):
        if idx < 0:
            continue
        combined[idx] = max(combined.get(idx, 0.0), args.alpha * vec_sc[rank])

    for idx, score in enumerate(bm25_sc):
        combined[idx] = combined.get(idx, 0.0) + (1 - args.alpha) * score

    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[: args.topk]

    metadata = load_metadata(faiss_meta)

    print(f"Query: {args.query}")
    print("Results:")
    for rank, (idx, score) in enumerate(top, start=1):
        row = metadata[idx]
        snippet = (row.get("text") or "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        print(
            f"{rank}. score={score:.4f} doc={row.get('doc_id')} "
            f"page={row.get('page_number')} chunk={row.get('chunk_id')}"
        )
        print(f"   {snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
