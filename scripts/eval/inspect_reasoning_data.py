"""Inspect downloaded reasoning dataset schemas (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path("/root/autodl-tmp/caches/datasets")

    import pandas as pd

    gsm = pd.read_parquet(root / "gsm8k" / "main" / "test-00000-of-00001.parquet")
    print("gsm8k test rows:", len(gsm))
    print("gsm8k columns:", list(gsm.columns))
    print("gsm8k sample:", json.dumps(gsm.iloc[0].to_dict(), ensure_ascii=False)[:500])

    mmlu = pd.read_parquet(root / "mmlu" / "all" / "test-00000-of-00001.parquet")
    print("mmlu all/test rows:", len(mmlu))
    print("mmlu columns:", list(mmlu.columns))
    print("mmlu sample:", json.dumps(mmlu.iloc[0].to_dict(), ensure_ascii=False)[:500])

    for split in ("aime2025-I", "aime2025-II"):
        with (root / "aime2025" / f"{split}.jsonl").open(encoding="utf-8") as handle:
            line = handle.readline()
        print(f"{split} sample:", line[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
