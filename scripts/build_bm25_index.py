import glob
import json
import os
import re
from typing import List

from rank_bm25 import BM25Okapi

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts")
INDEX_DIR = os.path.join(ARTIFACTS_DIR, "index")


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


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def main() -> int:
    rows = load_chunks()
    if not rows:
        return 0

    os.makedirs(INDEX_DIR, exist_ok=True)

    tokens = [tokenize(r.get("text", "")) for r in rows]
    bm25 = BM25Okapi(tokens)

    bm25_path = os.path.join(INDEX_DIR, "bm25.json")
    meta_path = os.path.join(INDEX_DIR, "bm25_chunks.jsonl")

    with open(bm25_path, "w", encoding="utf-8") as f:
        payload = {
            "corpus_tokens": tokens,
            "avgdl": bm25.avgdl,
            "idf": bm25.idf,
        }
        f.write(json.dumps(payload, ensure_ascii=True))

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

    print(f"Wrote BM25 index: {bm25_path}")
    print(f"Wrote BM25 metadata: {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
