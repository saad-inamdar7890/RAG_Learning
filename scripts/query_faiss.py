import argparse
import json
import os
from typing import List, Tuple

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


def load_metadata(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Query FAISS index with a text prompt.")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--topk", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    index_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "chunks.jsonl")
    info_path = os.path.join(INDEX_DIR, "index_info.json")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print("Index or metadata not found. Run build_faiss_index.py first.")
        return 1

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.loads(f.read().strip())
        model_name = info.get("model", model_name)

    model = SentenceTransformer(model_name)
    query_vec = model.encode([args.query], normalize_embeddings=True)

    index = faiss.read_index(index_path)
    scores, indices = index.search(np.asarray(query_vec, dtype=np.float32), args.topk)

    metadata = load_metadata(meta_path)
    print(f"Model: {model_name}")
    print(f"Query: {args.query}")
    print("Results:")

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
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
