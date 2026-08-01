"""下载 Wikitext-2-raw-v1 test 集，保存为纯文本文件供 PPL 评测。

经 hf-mirror（huggingface.co 被墙）。用 hf_hub_download（重试逻辑好）下载 parquet，
再用 pyarrow 解析。

用法: python scripts/fetch_wikitext.py --max-docs 800
输出: data/wikitext2_test.txt + data/MANIFEST.yaml 哈希
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO = "wikitext"
FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=800)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    local = hf_hub_download(REPO, FILE, repo_type="dataset", cache_dir="data/hf_cache")
    table = pq.read_table(local)
    text_col = table.column("text").to_pylist()
    docs = [d for d in text_col if d and d.strip()][: args.max_docs]

    out = Path("data/wikitext2_test.txt")
    out.parent.mkdir(exist_ok=True)
    text = "\n\n".join(docs)
    out.write_text(text, encoding="utf-8")

    sha = hashlib.sha256(text.encode()).hexdigest()
    print(f"文档数: {len(docs)} | 字符: {len(text):,} | 文件: {out}")
    print(f"sha256: {sha}")


if __name__ == "__main__":
    main()
