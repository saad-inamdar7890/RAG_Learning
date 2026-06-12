import json
import os
import re
import time
from typing import List, Dict

import numpy as np
import faiss
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer
from src import metrics

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_DIR = os.path.join(BASE_DIR, "data", "artifacts", "index")

def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)

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

class RAGPipeline:
    def __init__(self, ollama_host="http://localhost:11434", ollama_model="llama3.1:8b"):
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        
        index_path = os.path.join(INDEX_DIR, "faiss.index")
        meta_path = os.path.join(INDEX_DIR, "chunks.jsonl")
        bm25_path = os.path.join(INDEX_DIR, "bm25.json")
        bm25_meta = os.path.join(INDEX_DIR, "bm25_chunks.jsonl")
        info_path = os.path.join(INDEX_DIR, "index_info.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise RuntimeError("Index or metadata not found. Run build_faiss_index.py first.")

        # Load Metadata
        self.metadata = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        
        # Load embedder
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.loads(f.read().strip())
            model_name = info.get("model", model_name)
        
        print("Loading embedder...")
        self.embedder = SentenceTransformer(model_name)
        
        print("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        
        print("Loading BM25 index...")
        self.use_hybrid = os.path.exists(bm25_path) and os.path.exists(bm25_meta)
        if self.use_hybrid:
            with open(bm25_path, "r", encoding="utf-8") as f:
                self.bm25_data = json.loads(f.read())
        
        print("Loading Cross-Encoder reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Pipeline initialized!")

    def retrieve_and_rerank(self, query: str, top_k: int = 5, alpha: float = 0.6) -> Dict:
        """Returns dict with 'chunks' and 'step_times' for tracing."""
        step_times: Dict[str, float] = {}

        # Step 1: Embed
        t0 = time.perf_counter()
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        step_times["embed_s"] = round(time.perf_counter() - t0, 3)

        # Step 2: Vector search
        t0 = time.perf_counter()
        vec_scores, vec_indices = self.index.search(np.asarray(query_vec, dtype=np.float32), 50)
        step_times["faiss_s"] = round(time.perf_counter() - t0, 3)

        # Step 3: BM25 + hybrid merge
        t0 = time.perf_counter()
        if self.use_hybrid:
            bm25_sc = bm25_scores(self.bm25_data, tokenize(query))
            vec_sc = vec_scores[0]
            if vec_sc.max() > vec_sc.min():
                vec_sc = (vec_sc - vec_sc.min()) / (vec_sc.max() - vec_sc.min())
            if bm25_sc.max() > bm25_sc.min():
                bm25_sc = (bm25_sc - bm25_sc.min()) / (bm25_sc.max() - bm25_sc.min())
            combined = {}
            for rank, idx in enumerate(vec_indices[0]):
                if idx < 0: continue
                combined[idx] = max(combined.get(idx, 0.0), alpha * vec_sc[rank])
            for idx, score in enumerate(bm25_sc):
                combined[idx] = combined.get(idx, 0.0) + (1 - alpha) * score
            top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:50]
            retrieved = [self.metadata[idx] for idx, _ in top]
        else:
            retrieved = []
            for idx in vec_indices[0][:50]:
                if 0 <= idx < len(self.metadata):
                    retrieved.append(self.metadata[idx])
        step_times["bm25_hybrid_s"] = round(time.perf_counter() - t0, 3)

        # Deduplicate by parent text
        deduped = {}
        for r in retrieved:
            parent_text = r.get("parent_text", r.get("text", ""))
            if parent_text not in deduped:
                new_r = r.copy()
                new_r["text"] = parent_text
                deduped[parent_text] = new_r
        retrieved_deduped = list(deduped.values())

        # Step 4: Cross-encoder reranking
        t0 = time.perf_counter()
        if retrieved_deduped:
            pairs = [[query, r.get("text", "")] for r in retrieved_deduped[:15]]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(retrieved_deduped[:15], scores), key=lambda x: x[1], reverse=True)
            retrieved_deduped = [r for r, _ in ranked] + retrieved_deduped[15:]
        step_times["rerank_s"] = round(time.perf_counter() - t0, 3)

        return {"chunks": retrieved_deduped[:top_k], "step_times": step_times}

    def generate_answer(self, query: str, context_chunks: List[dict]) -> Dict:
        """Returns dict with 'answer', 'tokens', and 'generate_s'."""
        context = build_context(context_chunks)
        prompt = (
            "You are a strict enterprise AI assistant. You MUST answer the user's question using ONLY the provided context.\n"
            "CRITICAL: You MUST cite your sources for every claim using bracketed numbers like [1] or [2].\n"
            "If the answer is not in the context, you must reply: 'I cannot answer this based on the provided documents.'\n\n"
            f"Question: {query}\n\nContext:\n{context}\n\n"
            "Answer with mandatory citations:"
        )
        url = self.ollama_host.rstrip("/") + "/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        t0 = time.perf_counter()
        response = requests.post(url, json=payload, timeout=120)
        generate_s = round(time.perf_counter() - t0, 3)

        if response.status_code != 200:
            raise RuntimeError(f"Ollama Error: {response.text}")
        data = response.json()
        tokens = data.get("eval_count", 0)  # tokens generated by Ollama
        return {
            "answer": data.get("response", "").strip(),
            "tokens": tokens,
            "generate_s": generate_s,
        }

    def ask(self, query: str) -> Dict:
        t_total = time.perf_counter()

        retrieval = self.retrieve_and_rerank(query)
        chunks = retrieval["chunks"]
        step_times = retrieval["step_times"]

        generation = self.generate_answer(query, chunks)
        step_times["generate_s"] = generation["generate_s"]

        total_s = round(time.perf_counter() - t_total, 3)

        # Record to metrics store
        metrics.record_request(
            total_s=total_s,
            steps=step_times,
            tokens=generation["tokens"],
        )

        sources = []
        for i, chunk in enumerate(chunks, start=1):
            sources.append({
                "citation_number": i,
                "doc_id": chunk.get("doc_id"),
                "page_number": chunk.get("page_number"),
                "text": chunk.get("text", "")[:200] + "...",
            })

        return {
            "answer": generation["answer"],
            "sources": sources,
            "trace": {
                "total_s": total_s,
                "steps": step_times,
                "tokens": generation["tokens"],
            },
        }
