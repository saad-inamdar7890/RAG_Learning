import glob
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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts")
INDEX_DIR = os.path.join(ARTIFACTS_DIR, "index")

MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH", "64"))


def load_chunks() -> List[dict]:
    rows: List[dict] = []
    files = sorted(glob.glob(os.path.join(ARTIFACTS_DIR, "*__chunks.jsonl")))
    if not files:
        print("No chunk files found in data/artifacts.")
        return rows

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def main() -> int:
    rows = load_chunks()
    if not rows:
        return 0

    os.makedirs(INDEX_DIR, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)
    texts = [r.get("text", "") for r in rows]
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(embeddings, dtype=np.float32))

    index_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "chunks.jsonl")
    info_path = os.path.join(INDEX_DIR, "index_info.json")

    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for r in rows:
            meta = {
                "chunk_id": r.get("chunk_id"),
                "doc_id": r.get("doc_id"),
                "page_number": r.get("page_number"),
                "start_char": r.get("start_char"),
                "end_char": r.get("end_char"),
                "text": r.get("text"),
            }
            f.write(json.dumps(meta, ensure_ascii=True) + "\n")

    with open(info_path, "w", encoding="utf-8") as f:
        info = {
            "model": MODEL_NAME,
            "chunks": len(rows),
            "dim": dim,
        }
        f.write(json.dumps(info, ensure_ascii=True, indent=2) + "\n")

    print(f"Wrote index: {index_path}")
    print(f"Wrote metadata: {meta_path}")
    print(f"Wrote info: {info_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
