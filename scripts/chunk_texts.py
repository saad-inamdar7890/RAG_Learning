import argparse
import glob
import json
import os
from typing import List, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts")


def load_pages(path: str) -> List[dict]:
    pages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pages.append(json.loads(line))
    return pages


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def chunk_text(text: str, size: int, overlap: int) -> List[Tuple[int, int, str]]:
    chunks = []
    if size <= 0:
        return chunks

    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def write_chunks(doc_id: str, chunks: List[dict]) -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_path = os.path.join(ARTIFACTS_DIR, f"{doc_id}__chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk parsed page-wise JSONL files.")
    parser.add_argument("--size", type=int, default=800, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="Chunk overlap in characters")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of docs (0 = all)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(CLEAN_DIR, "*__pages.jsonl")))
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        print("No parsed JSONL files found in data/clean.")
        return 0

    for path in files:
        pages = load_pages(path)
        if not pages:
            print(f"No pages in {os.path.basename(path)}")
            continue

        doc_id = pages[0].get("doc_id") or os.path.basename(path).split("__", 1)[0]
        chunk_rows = []
        chunk_index = 0
        for page in pages:
            text = normalize_whitespace(page.get("text", ""))
            if not text:
                continue
            page_number = page.get("page_number")
            for start_char, end_char, chunk in chunk_text(text, args.size, args.overlap):
                row = {
                    "chunk_id": f"{doc_id}::p{page_number}::c{chunk_index:05d}",
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "page_number": page_number,
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": chunk,
                }
                chunk_rows.append(row)
                chunk_index += 1

        out_path = write_chunks(doc_id, chunk_rows)
        print(f"Wrote {out_path} ({len(chunk_rows)} chunks)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
