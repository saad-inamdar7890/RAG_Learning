import argparse
import json
import os
import re
from typing import List

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    import faiss
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "faiss is required. Install with: python -m pip install faiss-cpu"
    ) from exc

try:
    import requests
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "requests is required. Install with: python -m pip install requests"
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


def build_context(chunks: List[dict]) -> str:
    lines = []
    for i, row in enumerate(chunks, start=1):
        doc_id = row.get("doc_id")
        page = row.get("page_number")
        text = (row.get("text") or "").strip()
        lines.append(f"[{i}] doc={doc_id} page={page}\n{text}")
    return "\n\n".join(lines)


def call_ollama(prompt: str, model: str, host: str) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RAG answerer using Ollama + hybrid retrieval + reranker."
    )
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--topk", type=int, default=5, help="Top-k chunks")
    parser.add_argument("--alpha", type=float, default=0.6, help="Hybrid weight")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Candidate size for hybrid retrieval",
    )
    parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    parser.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name",
    )
    parser.add_argument(
        "--rerank-topk",
        type=int,
        default=10,
        help="Top-k for reranking candidates",
    )
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model")
    parser.add_argument(
        "--host", default="http://localhost:11434", help="Ollama host URL"
    )
    args = parser.parse_args()

    index_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "chunks.jsonl")
    bm25_path = os.path.join(INDEX_DIR, "bm25.json")
    bm25_meta = os.path.join(INDEX_DIR, "bm25_chunks.jsonl")
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
    vec_scores, vec_indices = index.search(
        np.asarray(query_vec, dtype=np.float32), args.candidate_k
    )

    metadata = load_metadata(meta_path)

    use_hybrid = os.path.exists(bm25_path) and os.path.exists(bm25_meta)
    if use_hybrid:
        bm25_data = load_bm25(bm25_path)
        bm25_sc = bm25_scores(bm25_data, tokenize(args.query))

        vec_sc = vec_scores[0]
        if vec_sc.max() > vec_sc.min():
            vec_sc = (vec_sc - vec_sc.min()) / (vec_sc.max() - vec_sc.min())
        if bm25_sc.max() > bm25_sc.min():
            bm25_sc = (bm25_sc - bm25_sc.min()) / (bm25_sc.max() - bm25_sc.min())

        combined = {}
        for rank, idx in enumerate(vec_indices[0]):
            if idx < 0:
                continue
            combined[idx] = max(combined.get(idx, 0.0), args.alpha * vec_sc[rank])

        for idx, score in enumerate(bm25_sc):
            combined[idx] = combined.get(idx, 0.0) + (1 - args.alpha) * score

        top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[: args.topk]
        retrieved = [metadata[idx] for idx, _ in top]
    else:
        retrieved = []
        for idx in vec_indices[0][: args.topk]:
            if idx < 0 or idx >= len(metadata):
                continue
            retrieved.append(metadata[idx])

    if args.rerank and retrieved:
        pairs = [[args.query, r.get("text", "")] for r in retrieved[: args.rerank_topk]]
        reranker = CrossEncoder(args.rerank_model)
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(retrieved[: args.rerank_topk], scores), key=lambda x: x[1], reverse=True
        )
        retrieved = [r for r, _ in ranked] + retrieved[args.rerank_topk :]
        retrieved = retrieved[: args.topk]

    context = build_context(retrieved)

    prompt = (
        "You are an assistant that answers questions using ONLY the provided context. "
        "Cite sources using bracketed numbers like [1], [2]. If the answer is not "
        "in the context, say you don't know.\n\n"
        f"Question: {args.query}\n\nContext:\n{context}\n\n"
        "Answer with citations."
    )

    answer = call_ollama(prompt, args.model, args.host)
    print(answer)

    print("\nSources:")
    for i, row in enumerate(retrieved, start=1):
        print(
            f"[{i}] doc={row.get('doc_id')} page={row.get('page_number')} "
            f"chunk={row.get('chunk_id')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
