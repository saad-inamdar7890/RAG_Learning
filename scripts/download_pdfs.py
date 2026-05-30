import csv
import os
import sys
import urllib.request
from urllib.parse import urlparse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "data", "lists", "corpus_status.csv")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def safe_filename(doc_id: str, url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    if not name:
        name = f"{doc_id}.pdf"
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return f"{doc_id}__{name}"


def download(url: str, dest_path: str) -> None:
    with urllib.request.urlopen(url) as response:
        data = response.read()
    with open(dest_path, "wb") as f:
        f.write(data)


def main() -> int:
    os.makedirs(RAW_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return 1

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    to_download = [r for r in rows if r.get("status", "").strip().lower() == "new"]
    if not to_download:
        print("No rows with status=new to download.")
        return 0

    for row in to_download:
        doc_id = (row.get("doc_id") or "").strip()
        url = (row.get("source_url") or "").strip()
        if not doc_id or not url:
            print(f"Skipping row with missing doc_id/url: {row}")
            continue

        filename = safe_filename(doc_id, url)
        dest_path = os.path.join(RAW_DIR, filename)
        if os.path.exists(dest_path):
            print(f"Exists, skipping: {dest_path}")
            continue

        try:
            print(f"Downloading {doc_id}: {url}")
            download(url, dest_path)
            print(f"Saved to {dest_path}")
        except Exception as exc:
            print(f"Failed {doc_id}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
