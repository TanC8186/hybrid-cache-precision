"""Compare RULER-subset fp16 (fp32 state) vs fp16_statebf16 (bf16 state)."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TASKS = [
    "ruler_niah_single",
    "ruler_niah_multikey",
    "ruler_niah_multivalue",
    "ruler_niah_multiquery",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
]
LENGTHS = [4096, 8192]


def load_cell(ruler_dir: Path, attempt: str, task: str, length: int, seed: int) -> float:
    path = ruler_dir / attempt / f"{task}__L{length}__fp16_statebf16__s{seed}.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("status") != "completed_validated":
        raise SystemExit(f"cell not completed: {path}")
    return float(rec["accuracy"])


def load_cell_fp16(ruler_dir: Path, attempt: str, task: str, length: int, seed: int) -> float:
    path = ruler_dir / attempt / f"{task}__L{length}__fp16__s{seed}.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("status") != "completed_validated":
        raise SystemExit(f"cell not completed: {path}")
    return float(rec["accuracy"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler-dir", default="results/quality/ruler-subset")
    ap.add_argument("--attempt-bf16", default="ruler-subset-20260808-statebf16")
    ap.add_argument("--attempt-fp16", default="ruler-subset-20260807-v2-256")
    ap.add_argument("--out", default="results/quality/ruler-statebf16-analysis-20260808.json")
    args = ap.parse_args()
    ruler_dir = Path(args.ruler_dir)
    seed = 7

    rows = []
    for task in TASKS:
        for length in LENGTHS:
            fp16 = load_cell_fp16(ruler_dir, args.attempt_fp16, task, length, seed)
            bf16 = load_cell(ruler_dir, args.attempt_bf16, task, length, seed)
            rows.append(
                {
                    "task": task,
                    "length": length,
                    "fp16_state_fp32_acc": fp16,
                    "fp16_state_bf16_acc": bf16,
                    "delta_bf16_minus_fp32": round(bf16 - fp16, 2),
                }
            )

    fp16_vals = [r["fp16_state_fp32_acc"] for r in rows]
    bf16_vals = [r["fp16_state_bf16_acc"] for r in rows]
    result = {
        "schema_version": 1,
        "metric": "string_match_all (official RULER)",
        "protocol": (
            "ruler_quality.py --max-tokens 256 --thinking default, seed 7; "
            "fp16_statebf16 adds mamba_ssm_cache_dtype=bfloat16 to the fp16 engine args"
        ),
        "rows": rows,
        "overall": {
            "n_cells": len(rows),
            "fp32_state_mean": round(statistics.mean(fp16_vals), 2),
            "bf16_state_mean": round(statistics.mean(bf16_vals), 2),
            "delta_mean": round(statistics.mean([b - f for f, b in zip(fp16_vals, bf16_vals)]), 2),
            "delta_min": round(min(b - f for f, b in zip(fp16_vals, bf16_vals)), 2),
            "delta_max": round(max(b - f for f, b in zip(fp16_vals, bf16_vals)), 2),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
