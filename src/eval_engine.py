"""
Evaluation engine that re-chunks documents using a chosen strategy,
rebuilds the vector indexes, and runs the benchmark query set.
"""
import glob
import json
import os
import re
from typing import List, Dict
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import faiss
except Exception as exc:
    raise SystemExit("faiss is required: pip install faiss-cpu") from exc

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"

# ── Benchmark query set ──────────────────────────────────────────────────────
EVAL_QUERIES = [
    {"query": "What is covered under the standard fire policy?",
     "expected_topic": "fire damage coverage"},
    {"query": "Are floods covered by a standard homeowner's policy?",
     "expected_topic": "flood exclusions"},
    {"query": "What is the standard deductible for collision coverage?",
     "expected_topic": "collision deductible"},
    {"query": "Is windstorm or hail damage covered?",
     "expected_topic": "wind and hail coverage"},
    {"query": "What is 'loss of use' coverage?",
     "expected_topic": "additional living expenses or loss of use"},
    {"query": "What are the limits for personal property coverage?",
     "expected_topic": "personal property limits"},
    {"query": "How is roof damage depreciation calculated?",
     "expected_topic": "roof depreciation or actual cash value"},
    {"query": "How can I insure expensive jewelry or art?",
     "expected_topic": "scheduled personal property or high-value items"},
    {"query": "Does this cover medical payments if someone gets hurt on my property?",
     "expected_topic": "medical payments to others"},
    {"query": "What does an umbrella policy add?",
     "expected_topic": "umbrella limits or excess liability"},
    {"query": "Is vandalism covered if the home is vacant?",
     "expected_topic": "vandalism and vacancy clauses"},
    {"query": "What happens if I am hit by an uninsured driver?",
     "expected_topic": "uninsured motorist coverage"},
    {"query": "Will the policy pay for a rental car while mine is being repaired?",
     "expected_topic": "rental reimbursement"},
    {"query": "Are there limitations on mold and fungus damage?",
     "expected_topic": "mold and fungus limits"},
    {"query": "What is the definition of insured in a homeowners policy?",
     "expected_topic": "definition of insured persons"},
    {"query": "What exclusions apply to business property at home?",
     "expected_topic": "business property exclusions"},
    {"query": "How does the claims process work?",
     "expected_topic": "claims filing process"},
    {"query": "What is subrogation in insurance?",
     "expected_topic": "subrogation rights"},
    {"query": "Does my policy cover damage caused by animals or pests?",
     "expected_topic": "animal or pest damage"},
    {"query": "What is replacement cost value vs actual cash value?",
     "expected_topic": "replacement cost vs actual cash value"},
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _load_pages() -> List[dict]:
    pages = []
    for path in sorted(glob.glob(os.path.join(CLEAN_DIR, "*__pages.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pages.append(json.loads(line))
    return pages


# ── Chunking strategies ───────────────────────────────────────────────────────

def chunk_normal(pages: List[dict]) -> List[dict]:
    """Fixed-size character chunks with overlap (800 / 150)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks, idx = [], 0
    for page in pages:
        text = " ".join(page.get("text", "").split())
        doc_id = page.get("doc_id", "UNKNOWN")
        page_no = page.get("page_number", 0)
        for ch in splitter.split_text(text):
            chunks.append({
                "chunk_id": f"{doc_id}::p{page_no}::c{idx:05d}",
                "doc_id": doc_id,
                "page_number": page_no,
                "text": ch,
                "chunk_index": idx,
            })
            idx += 1
    return chunks


def chunk_semantic(pages: List[dict]) -> List[dict]:
    """
    Larger semantic windows (1200 / 200) — keeps more context together
    so the embedder can capture full arguments/clauses.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks, idx = [], 0
    for page in pages:
        text = " ".join(page.get("text", "").split())
        doc_id = page.get("doc_id", "UNKNOWN")
        page_no = page.get("page_number", 0)
        for ch in splitter.split_text(text):
            chunks.append({
                "chunk_id": f"{doc_id}::p{page_no}::c{idx:05d}",
                "doc_id": doc_id,
                "page_number": page_no,
                "text": ch,
                "chunk_index": idx,
            })
            idx += 1
    return chunks


def chunk_parent_child(pages: List[dict]) -> List[dict]:
    """
    Parent (1500 / 150) → Child (300 / 50) hierarchy.
    Embeddings use child text; retrieval context returns parent text.
    """
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks, idx = [], 0
    for page in pages:
        text = " ".join(page.get("text", "").split())
        doc_id = page.get("doc_id", "UNKNOWN")
        page_no = page.get("page_number", 0)
        for parent_ch in parent_splitter.split_text(text):
            for child_ch in child_splitter.split_text(parent_ch):
                chunks.append({
                    "chunk_id": f"{doc_id}::p{page_no}::c{idx:05d}",
                    "doc_id": doc_id,
                    "page_number": page_no,
                    "text": child_ch,          # embed with child
                    "parent_text": parent_ch,  # retrieve with parent
                    "chunk_index": idx,
                })
                idx += 1
    return chunks


# ── Index building ────────────────────────────────────────────────────────────

def build_indexes(chunks: List[dict], index_dir: str) -> None:
    """Embed chunks with FAISS and build BM25 in index_dir."""
    os.makedirs(index_dir, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    texts = [c.get("text", "") for c in chunks]

    embeddings = model.encode(texts, batch_size=64,
                              show_progress_bar=False,
                              normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(embeddings, dtype=np.float32))
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    with open(os.path.join(index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            row = {k: c.get(k) for k in
                   ("chunk_id", "doc_id", "page_number", "text", "parent_text")}
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    with open(os.path.join(index_dir, "index_info.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "chunks": len(chunks), "dim": dim}, f, indent=2)

    # BM25
    token_corpus = [tokenize(t) for t in texts]
    avg_dl = sum(len(t) for t in token_corpus) / max(len(token_corpus), 1)
    df: Dict[str, int] = {}
    for doc in token_corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    N = len(token_corpus)
    idf = {term: float(np.log((N - cnt + 0.5) / (cnt + 0.5) + 1))
           for term, cnt in df.items()}
    bm25_payload = {"corpus_tokens": token_corpus, "avgdl": avg_dl, "idf": idf}

    with open(os.path.join(index_dir, "bm25.json"), "w", encoding="utf-8") as f:
        json.dump(bm25_payload, f, ensure_ascii=True)

    with open(os.path.join(index_dir, "bm25_chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            row = {k: c.get(k) for k in
                   ("chunk_id", "doc_id", "page_number", "text")}
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


# ── Retrieve + Rerank ────────────────────────────────────────────────────────

def _retrieve(query: str, index_dir: str, top_k: int = 5) -> List[dict]:
    embedder = SentenceTransformer(MODEL_NAME)
    qv = embedder.encode([query], normalize_embeddings=True)

    faiss_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    meta = []
    with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))

    vec_scores, vec_idx = faiss_index.search(
        np.asarray(qv, dtype=np.float32), min(50, len(meta)))

    with open(os.path.join(index_dir, "bm25.json"), encoding="utf-8") as f:
        bm25_data = json.load(f)
    qtoken = tokenize(query)
    idf = bm25_data["idf"]
    avgdl = bm25_data["avgdl"]
    corpus_tokens = bm25_data["corpus_tokens"]
    k1, b = 1.5, 0.75
    bm25_sc = []
    for doc in corpus_tokens:
        dl = len(doc)
        tf_map: Dict[str, int] = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1
        sc = 0.0
        for q in qtoken:
            if q in tf_map:
                tf = tf_map[q]
                sc += idf.get(q, 0) * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * (dl / avgdl)))
        bm25_sc.append(sc)
    bm25_arr = np.array(bm25_sc, dtype=np.float32)

    vs = vec_scores[0]
    if vs.max() > vs.min():
        vs = (vs - vs.min()) / (vs.max() - vs.min())
    if bm25_arr.max() > bm25_arr.min():
        bm25_arr = (bm25_arr - bm25_arr.min()) / (bm25_arr.max() - bm25_arr.min())

    alpha = 0.6
    combined: Dict[int, float] = {}
    for rank, idx in enumerate(vec_idx[0]):
        if idx < 0:
            continue
        combined[int(idx)] = max(combined.get(int(idx), 0.0), alpha * float(vs[rank]))
    for idx, sc in enumerate(bm25_arr):
        combined[idx] = combined.get(idx, 0.0) + (1 - alpha) * float(sc)

    top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]
    retrieved = [meta[idx] for idx, _ in top if idx < len(meta)]

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, r.get("parent_text") or r.get("text", "")] for r in retrieved[:15]]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(retrieved[:15], scores), key=lambda x: x[1], reverse=True)
    return [r for r, _ in ranked][:top_k]


LATENCY_THRESHOLD_S = 30.0  # seconds — answers must arrive within this

def _check_citations(answer: str) -> bool:
    """Criterion 2: Answer must contain at least one [N] bracketed citation."""
    import re
    return bool(re.search(r'\[\d+\]', answer))


def _combined_judge(query: str, answer: str, expected_topic: str, context_chunks: List[dict]) -> dict:
    """
    Combines faithfulness and topicality judges into a single LLM call.
    Returns: {"faithful": bool, "topical": bool}
    """
    import requests as req
    context_texts = "\n".join(
        f"[{i+1}] {(c.get('parent_text') or c.get('text',''))[:400]}"
        for i, c in enumerate(context_chunks)
    )
    prompt = (
        "You are an impartial judge. Evaluate the AI's ANSWER based on two criteria:\n"
        "1. Faithfulness: Does the ANSWER contain ONLY facts found in the CONTEXT? (True/False)\n"
        f"2. Topicality: Does the ANSWER cover the topic of: '{expected_topic}'? (True/False)\n\n"
        f"QUESTION: {query}\n\n"
        f"CONTEXT:\n{context_texts}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Reply with a raw JSON object (no markdown, no extra text) with exact keys 'faithful' and 'topical' mapping to boolean values:\n"
        "{\"faithful\": true, \"topical\": true}"
    )
    resp = req.post(
        OLLAMA_HOST.rstrip("/") + "/api/generate",
        json={
            "model": OLLAMA_MODEL, 
            "prompt": prompt,
            "stream": False, 
            "format": "json",
            "options": {"temperature": 0.0}
        },
        timeout=60,
    )
    resp.raise_for_status()
    try:
        res = json.loads(resp.json().get("response", "{}"))
        return {
            "faithful": bool(res.get("faithful", False)),
            "topical": bool(res.get("topical", False)),
        }
    except Exception:
        return {"faithful": False, "topical": False}


def _generate(query: str, chunks: List[dict]) -> str:
    import requests as req
    lines = []
    for i, c in enumerate(chunks, 1):
        text = (c.get("parent_text") or c.get("text", "")).strip()
        lines.append(f"[{i}] doc={c.get('doc_id')} page={c.get('page_number')}\n{text}")
    context = "\n\n".join(lines)
    prompt = (
        "You are a strict enterprise AI assistant. "
        "Answer using ONLY the context below.\n"
        "CRITICAL: cite every claim with [1], [2], etc.\n"
        "If the answer is not in the context reply: "
        "'I cannot answer this based on the provided documents.'\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    )
    resp = req.post(
        OLLAMA_HOST.rstrip("/") + "/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt,
              "stream": False, "options": {"temperature": 0.2}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


# ── Public interface ─────────────────────────────────────────────────────────

def run_evaluation(strategy: str) -> List[dict]:
    """
    Full pipeline:
      1. Load pages
      2. Chunk with `strategy` (normal | semantic | parent_child)
      3. Build indexes
      4. For each eval query, evaluate on 4 strict criteria:
           - Latency      : end-to-end < LATENCY_THRESHOLD_S
           - Citation     : answer contains [N] bracketed references
           - Faithfulness : answer grounded only in retrieved context
           - Topical      : answer addresses the expected topic
      5. A query PASSES only if ALL 4 criteria pass
      6. Return list of result dicts with per-criterion breakdown
    """
    import time

    pages = _load_pages()
    if not pages:
        raise RuntimeError("No cleaned page files found in data/clean.")

    chunkers = {
        "normal": chunk_normal,
        "semantic": chunk_semantic,
        "parent_child": chunk_parent_child,
    }
    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: {list(chunkers)}")

    chunks = chunkers[strategy](pages)
    build_indexes(chunks)

    results = []
    for item in EVAL_QUERIES:
        query = item["query"]
        expected = item["expected_topic"]

        # ── Criterion 1: Latency ──────────────────────────────────────────
        t_start = time.time()
        try:
            retrieved = _retrieve(query)
            answer = _generate(query, retrieved)
        except Exception as exc:
            answer = f"ERROR: {exc}"
            retrieved = []

        latency_s = round(time.time() - t_start, 2)
        latency_ok = latency_s <= LATENCY_THRESHOLD_S

        # ── Criterion 2: Citation format ──────────────────────────────────
        citation_ok = _check_citations(answer)

        # ── Criterion 3: Faithfulness ─────────────────────────────────────
        try:
            faithful_ok = _faithfulness_judge(query, answer, retrieved) if retrieved else False
        except Exception:
            faithful_ok = False

        # ── Criterion 4: Topical correctness ──────────────────────────────
        try:
            topical_ok = _topical_judge(query, answer, expected) if answer and not answer.startswith("ERROR") else False
        except Exception:
            topical_ok = False

        # ── Overall pass: ALL 4 must pass ─────────────────────────────────
        all_passed = latency_ok and citation_ok and faithful_ok and topical_ok

        results.append({
            "query": query,
            "expected_topic": expected,
            "answer": answer,
            "passed": all_passed,
            "latency_s": latency_s,
            "criteria": {
                "latency":     {"passed": latency_ok,  "detail": f"{latency_s}s (threshold: {LATENCY_THRESHOLD_S}s)"},
                "citation":    {"passed": citation_ok,  "detail": "Contains [N] citations" if citation_ok else "Missing citations"},
                "faithfulness":{"passed": faithful_ok,  "detail": "Grounded in context" if faithful_ok else "Introduces outside facts"},
                "topical":     {"passed": topical_ok,   "detail": f"Covers: {expected}"},
            },
            "sources": [
                {"doc_id": c.get("doc_id"), "page_number": c.get("page_number")}
                for c in retrieved
            ],
        })

    return results


def generate_single_eval(strategy: str, query: str) -> dict:
    """
    Ensure index exists for strategy, retrieve chunks, and generate answer.
    """
    import time
    
    pages = _load_pages()
    if not pages:
        raise RuntimeError("No cleaned page files found in data/clean.")

    chunkers = {
        "normal": chunk_normal,
        "semantic": chunk_semantic,
        "parent_child": chunk_parent_child,
    }
    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy '{strategy}'")

    index_dir = os.path.join(ARTIFACTS_DIR, f"index_{strategy}")
    if not os.path.exists(os.path.join(index_dir, "faiss.index")):
        chunks = chunkers[strategy](pages)
        build_indexes(chunks, index_dir)

    t_start = time.time()
    try:
        retrieved = _retrieve(query, index_dir)
        answer = _generate(query, retrieved)
    except Exception as exc:
        answer = f"ERROR: {exc}"
        retrieved = []
    
    latency_s = round(time.time() - t_start, 2)
    
    return {
        "answer": answer,
        "latency_s": latency_s,
        "retrieved_context": retrieved
    }

def judge_single_eval(query: str, answer: str, expected_topic: str, retrieved_context: List[dict], generate_latency_s: float) -> dict:
    """
    Run the strict 4 criteria on a single answer.
    """
    latency_ok = generate_latency_s <= LATENCY_THRESHOLD_S
    citation_ok = _check_citations(answer)
    
    faithful_ok = False
    topical_ok = False
    if answer and not answer.startswith("ERROR") and retrieved_context:
        judge_res = _combined_judge(query, answer, expected_topic, retrieved_context)
        faithful_ok = judge_res.get("faithful", False)
        topical_ok = judge_res.get("topical", False)

    all_passed = latency_ok and citation_ok and faithful_ok and topical_ok
    
    return {
        "passed": all_passed,
        "criteria": {
            "latency":      {"passed": latency_ok,  "detail": f"{generate_latency_s}s (limit {LATENCY_THRESHOLD_S}s)"},
            "citation":     {"passed": citation_ok,  "detail": "Has [N] citations" if citation_ok else "Missing citations"},
            "faithfulness": {"passed": faithful_ok,  "detail": "Grounded" if faithful_ok else "Outside facts"},
            "topical":      {"passed": topical_ok,   "detail": f"Covers: {expected_topic}"},
        }
    }
