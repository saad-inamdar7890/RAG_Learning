import argparse
import json
import os
from typing import List

from PyPDF2 import PdfReader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")


def list_pdfs(raw_dir: str) -> List[str]:
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")]
    files.sort()
    return [os.path.join(raw_dir, f) for f in files]


def doc_id_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    if "__" in base:
        return base.split("__", 1)[0]
    return os.path.splitext(base)[0]


def parse_pdf(pdf_path: str) -> List[dict]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page_number": i, "text": text})
    return pages


def write_jsonl(doc_id: str, pdf_path: str, pages: List[dict]) -> str:
    os.makedirs(CLEAN_DIR, exist_ok=True)
    out_path = os.path.join(CLEAN_DIR, f"{doc_id}__pages.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in pages:
            row = {
                "doc_id": doc_id,
                "source_pdf": os.path.basename(pdf_path),
                "page_number": item["page_number"],
                "text": item["text"],
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse PDFs into page-wise JSONL.")
    parser.add_argument("--limit", type=int, default=0, help="Number of PDFs to parse (0 = all)")
    args = parser.parse_args()

    if not os.path.isdir(RAW_DIR):
        print(f"Raw dir not found: {RAW_DIR}")
        return 1

    pdfs = list_pdfs(RAW_DIR)
    if not pdfs:
        print("No PDFs found in data/raw.")
        return 0

    target = pdfs if args.limit == 0 else pdfs[: args.limit]

    for pdf_path in target:
        doc_id = doc_id_from_filename(pdf_path)
        print(f"Parsing {doc_id}: {os.path.basename(pdf_path)}")
        pages = parse_pdf(pdf_path)
        out_path = write_jsonl(doc_id, pdf_path, pages)
        print(f"Wrote {out_path} ({len(pages)} pages)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
