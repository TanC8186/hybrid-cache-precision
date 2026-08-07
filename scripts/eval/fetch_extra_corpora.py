"""Fetch small C4/PG19 text slices for the extra PPL corpora (canonical protocol).

Uses streaming HF datasets via hf-mirror; saves deterministic slices as plain
text files (data/c4_slice.txt, data/pg19_slice.txt) plus a provenance JSON with
dataset/revision/sample ids. No GPU required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--c4-examples", type=int, default=30)
    ap.add_argument("--pg19-books", type=int, default=3)
    ap.add_argument("--pg19-chars-per-book", type=int, default=200_000)
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset  # noqa: PLC0415

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict = {}

    c4 = load_dataset("allenai/c4", "en", split="test", streaming=True)
    c4_lines = []
    c4_ids = []
    for index, row in enumerate(c4):
        if index >= args.c4_examples:
            break
        c4_lines.append(str(row["text"]).strip())
        c4_ids.append(index)
    c4_path = out_dir / "c4_slice.txt"
    c4_path.write_text("\n\n".join(c4_lines) + "\n", encoding="utf-8")
    provenance["c4"] = {
        "dataset": "allenai/c4",
        "config": "en",
        "split": "test",
        "example_indices": c4_ids,
        "chars": c4_path.stat().st_size,
        "sha256": sha256_file(c4_path),
    }

    pg19 = load_dataset("pg19", split="test", streaming=True)
    pg19_parts = []
    pg19_ids = []
    for index, row in enumerate(pg19):
        if index >= args.pg19_books:
            break
        pg19_parts.append(str(row["text"])[: args.pg19_chars_per_book])
        pg19_ids.append(index)
    pg19_path = out_dir / "pg19_slice.txt"
    pg19_path.write_text("\n\n".join(pg19_parts) + "\n", encoding="utf-8")
    provenance["pg19"] = {
        "dataset": "pg19",
        "split": "test",
        "book_indices": pg19_ids,
        "chars_per_book": args.pg19_chars_per_book,
        "chars": pg19_path.stat().st_size,
        "sha256": sha256_file(pg19_path),
    }

    provenance_path = out_dir / "extra_corpora_manifest.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
