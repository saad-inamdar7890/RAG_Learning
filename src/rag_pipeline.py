import json
import os
import re
from typing import List, Dict

import numpy as np
import faiss
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

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

    def retrieve_and_rerank(self, query: str, top_k: int = 5, alpha: float = 0.6) -> List[dict]:
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        vec_scores, vec_indices = self.index.search(np.asarray(query_vec, dtype=np.float32), 50)
        
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
                
            top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
            retrieved = [self.metadata[idx] for idx, _ in top]
        else:
            retrieved = []
            for idx in vec_indices[0][:top_k]:
                if 0 <= idx < len(self.metadata):
                    retrieved.append(self.metadata[idx])
        
        # Reranking
        if retrieved:
            pairs = [[query, r.get("text", "")] for r in retrieved[:10]]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(retrieved[:10], scores), key=lambda x: x[1], reverse=True)
            retrieved = [r for r, _ in ranked] + retrieved[10:]
            retrieved = retrieved[:top_k]
            
        return retrieved

    def generate_answer(self, query: str, context_chunks: List[dict]) -> str:
        context = build_context(context_chunks)
        prompt = (
            "You are an assistant that answers questions using ONLY the provided context. "
            "Cite sources using bracketed numbers like [1], [2]. If the answer is not "
            "in the context, say you don't know.\n\n"
            f"Question: {query}\n\nContext:\n{context}\n\n"
            "Answer with citations."
        )
        
        url = self.ollama_host.rstrip("/") + "/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama Error: {response.text}")
        response.raise_for_status()
        
        data = response.json()
        return data.get("response", "").strip()

    def ask(self, query: str) -> Dict:
        chunks = self.retrieve_and_rerank(query)
        answer = self.generate_answer(query, chunks)
        
        sources = []
        for i, chunk in enumerate(chunks, start=1):
            sources.append({
                "citation_number": i,
                "doc_id": chunk.get("doc_id"),
                "page_number": chunk.get("page_number"),
                "text": chunk.get("text", "")[:200] + "..." # snippet
            })
            
        return {
            "answer": answer,
            "sources": sources
        }
