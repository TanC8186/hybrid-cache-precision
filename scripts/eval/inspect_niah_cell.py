"""Inspect one NIAH rerun cell: max_tokens, hit/hit_think/hit_final diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/quality/niah-fixed/niah-fixed-20260807/fp16__seed7__d25__l2048.json")
    rec = json.loads(path.read_text(encoding="utf-8"))
    print(
        "file:", path.name,
        "| max_tokens:", rec.get("max_tokens"),
        "| acc:", rec.get("accuracy"),
        "| status:", rec.get("status"),
    )
    for case in rec["cases"]:
        print(
            "out_tokens:", case["output_tokens"],
            "| hit:", case["hit"],
            "| hit_think:", case["hit_think"],
            "| hit_final:", case["hit_final"],
            "| answer:", repr(case["answer"][:160]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
