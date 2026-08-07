"""Fetch deterministic C4/PG19 text slices for the extra PPL corpora.

The `datasets` streaming path repeatedly failed on the C4 file tree (12k
shards), so this version downloads raw artifacts directly:

- C4: allenai/c4 "en" validation shard 00000 (gzip-json) from hf-mirror at a
  fixed revision; first 30 non-empty documents become data/c4_slice.txt.
- PG19: test books 10146/10321/10356 from the canonical Google Cloud Storage
  bucket (deepmind-gutenberg); first 200k chars of each book become
  data/pg19_slice.txt.

Writes data/extra_corpora_manifest.json with sha256 + provenance. No GPU
required; raw artifacts are cached under data/{c4_raw,pg19_raw} and reused.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import urllib.request
from pathlib import Path


C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_SHARD_URL = (
    "https://hf-mirror.com/datasets/allenai/c4/resolve/"
    f"{C4_REVISION}/en/c4-validation.00000-of-00008.json.gz"
)
PG19_BASE_URL = "https://storage.googleapis.com/deepmind-gutenberg"
PG19_BOOKS = ["10146", "10321", "10356"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, target.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    print(f"saved {target} ({target.stat().st_size} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--c4-examples", type=int, default=30)
    ap.add_argument("--pg19-chars-per-book", type=int, default=200_000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_c4 = out_dir / "c4_raw" / "c4-validation.00000-of-00008.json.gz"
    provenance: dict = {}

    # C4 slice
    download(C4_SHARD_URL, raw_c4)
    c4_lines: list[str] = []
    c4_ids: list[int] = []
    with gzip.open(raw_c4, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            c4_lines.append(text)
            c4_ids.append(index)
            if len(c4_lines) >= args.c4_examples:
                break
    c4_path = out_dir / "c4_slice.txt"
    c4_path.write_text("\n\n".join(c4_lines) + "\n", encoding="utf-8")
    provenance["c4"] = {
        "dataset": "allenai/c4",
        "config": "en",
        "split": "validation",
        "revision": C4_REVISION,
        "shard": "c4-validation.00000-of-00008.json.gz",
        "line_indices": c4_ids,
        "chars": c4_path.stat().st_size,
        "sha256": sha256_file(c4_path),
    }
    print(json.dumps(provenance["c4"], ensure_ascii=False, indent=2))

    # PG19 slice
    pg19_parts: list[str] = []
    pg19_ids: list[str] = []
    for book in PG19_BOOKS:
        raw = out_dir / "pg19_raw" / f"test_{book}.txt"
        download(f"{PG19_BASE_URL}/test/{book}.txt", raw)
        pg19_parts.append(raw.read_text(encoding="utf-8")[: args.pg19_chars_per_book])
        pg19_ids.append(book)
    pg19_path = out_dir / "pg19_slice.txt"
    pg19_path.write_text("\n\n".join(pg19_parts) + "\n", encoding="utf-8")
    provenance["pg19"] = {
        "dataset": "deepmind/pg19",
        "split": "test",
        "source": PG19_BASE_URL,
        "book_ids": pg19_ids,
        "chars_per_book": args.pg19_chars_per_book,
        "chars": pg19_path.stat().st_size,
        "sha256": sha256_file(pg19_path),
    }
    print(json.dumps(provenance["pg19"], ensure_ascii=False, indent=2))

    provenance_path = out_dir / "extra_corpora_manifest.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest → {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
