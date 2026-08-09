"""RULER non-zero cells across 3 dataset seeds (42 + 11 + 23).

For each cell (task x length) report fp16/fp16_statebf16 mean over dataset
seeds and paired delta with 95% t-CI, plus think-truncation diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


CELLS = [
    ("ruler_fwe", 4096, "2b"),
    ("ruler_fwe", 8192, "2b"),
    ("ruler_niah_multiquery", 4096, "9b"),
    ("ruler_niah_multiquery", 8192, "9b"),
    ("ruler_fwe", 8192, "9b"),
]
DATASET_SEEDS = [42, 11, 23]


def t_half(n: int, sd: float) -> float:
    df = n - 1
    table = {1: 12.706, 2: 4.303}
    return table.get(df, 1.96) * sd / math.sqrt(n)


def legacy_attempt(model: str, alloc: str) -> str:
    if model == "2b":
        return "ruler-subset-20260808-statebf16" if alloc == "fp16_statebf16" else "ruler-subset-20260807-v2-256"
    return "ruler-subset-20260808-9b"


def load_cell(ruler_dir: Path, attempt_2b: str, attempt_9b: str, task: str, length: int, dseed: int, alloc: str, model: str) -> dict:
    if dseed == 42:
        attempt = legacy_attempt(model, alloc)
        name = f"{task}__L{length}__{alloc}__s7.json"
    else:
        attempt = attempt_2b if model == "2b" else attempt_9b
        name = f"{task}__L{length}__{alloc}__s7__d{dseed}.json"
    path = ruler_dir / attempt / name
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("status") != "completed_validated":
        raise SystemExit(f"cell not completed: {path}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler-dir", default="results/quality/ruler-subset")
    ap.add_argument("--attempt-2b", default="ruler-subset-20260809-multiseed-2b")
    ap.add_argument("--attempt-9b", default="ruler-subset-20260809-multiseed-9b")
    ap.add_argument("--out", default="results/quality/ruler-statebf16-multiseed-analysis-20260809.json")
    args = ap.parse_args()
    ruler_dir = Path(args.ruler_dir)

    rows = []
    for task, length, model in CELLS:
        fp16_vals = []
        bf16_vals = []
        per_seed = {}
        for dseed in DATASET_SEEDS:
            fp16 = load_cell(ruler_dir, args.attempt_2b, args.attempt_9b, task, length, dseed, "fp16", model)
            bf16 = load_cell(ruler_dir, args.attempt_2b, args.attempt_9b, task, length, dseed, "fp16_statebf16", model)
            fp16_vals.append(float(fp16["accuracy"]))
            bf16_vals.append(float(bf16["accuracy"]))
            per_seed[str(dseed)] = {
                "fp16": float(fp16["accuracy"]),
                "fp16_statebf16": float(bf16["accuracy"]),
            }
        diffs = [b - f for b, f in zip(bf16_vals, fp16_vals)]
        mean_d = statistics.mean(diffs)
        half = t_half(len(diffs), statistics.stdev(diffs)) if len(diffs) > 1 else 0.0
        rows.append(
            {
                "task": task,
                "length": length,
                "model": model,
                "fp16_mean": round(statistics.mean(fp16_vals), 2),
                "bf16_mean": round(statistics.mean(bf16_vals), 2),
                "delta_mean": round(mean_d, 2),
                "ci95_delta": [round(mean_d - half, 2), round(mean_d + half, 2)],
                "per_dataset_seed": per_seed,
            }
        )

    result = {
        "schema_version": 1,
        "dataset_seeds": DATASET_SEEDS,
        "engine_seed": 7,
        "protocol": "ruler_quality.py --max-tokens 256 --thinking default; dataset generated per random_seed",
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
