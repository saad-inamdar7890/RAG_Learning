import argparse
import glob
import json
import os
from statistics import mean

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def analyze_file(path: str):
    pages = list(load_jsonl(path))
    if not pages:
        return {
            "file": os.path.basename(path),
            "pages": 0,
            "empty_pages": 0,
            "empty_pct": 0.0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
        }

    lengths = [len(p.get("text", "")) for p in pages]
    empty_pages = sum(1 for n in lengths if n == 0)
    return {
        "file": os.path.basename(path),
        "pages": len(pages),
        "empty_pages": empty_pages,
        "empty_pct": (empty_pages / len(pages)) * 100.0,
        "avg_chars": int(mean(lengths)),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check parsing quality for page-wise JSONL files.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to scan (0 = all)")
    args = parser.parse_args()

    if not os.path.isdir(CLEAN_DIR):
        print(f"Clean dir not found: {CLEAN_DIR}")
        return 1

    files = sorted(glob.glob(os.path.join(CLEAN_DIR, "*__pages.jsonl")))
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        print("No parsed JSONL files found in data/clean.")
        return 0

    for path in files:
        stats = analyze_file(path)
        print(
            f"{stats['file']}: pages={stats['pages']}, empty={stats['empty_pages']} "
            f"({stats['empty_pct']:.1f}%), avg_chars={stats['avg_chars']}, "
            f"min={stats['min_chars']}, max={stats['max_chars']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
