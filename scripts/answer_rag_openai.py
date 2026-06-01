import argparse
import json
import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "faiss is required. Install with: python -m pip install faiss-cpu"
    ) from exc

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "openai is required. Install with: python -m pip install openai"
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


def build_context(chunks: List[dict]) -> str:
    lines = []
    for i, row in enumerate(chunks, start=1):
        doc_id = row.get("doc_id")
        page = row.get("page_number")
        text = (row.get("text") or "").strip()
        lines.append(f"[{i}] doc={doc_id} page={page}\n{text}")
    return "\n\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG answerer using OpenAI + FAISS.")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--topk", type=int, default=5, help="Top-k chunks")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
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

    embedder = SentenceTransformer(model_name)
    query_vec = embedder.encode([args.query], normalize_embeddings=True)

    index = faiss.read_index(index_path)
    scores, indices = index.search(np.asarray(query_vec, dtype=np.float32), args.topk)

    metadata = load_metadata(meta_path)
    retrieved = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        retrieved.append(metadata[idx])

    context = build_context(retrieved)

    system_msg = (
        "You are an assistant that answers questions using ONLY the provided context. "
        "Cite sources using bracketed numbers like [1], [2]. If the answer is not "
        "in the context, say you don't know."
    )
    user_msg = (
        f"Question: {args.query}\n\nContext:\n{context}\n\n"
        "Answer with citations."
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )

    print(response.choices[0].message.content)

    print("\nSources:")
    for i, row in enumerate(retrieved, start=1):
        print(
            f"[{i}] doc={row.get('doc_id')} page={row.get('page_number')} "
            f"chunk={row.get('chunk_id')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
