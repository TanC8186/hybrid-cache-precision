"""Compare GSM8K 3-seed fp16 (fp32 state) vs fp16_statebf16 (bf16 state)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def t_half(n: int, sd: float) -> float:
    df = n - 1
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    }
    return table.get(df, 1.96) * sd / math.sqrt(n)


def load(reasoning_dir: Path, attempt: str, allocation: str, seed: int) -> float:
    path = reasoning_dir / attempt / f"gsm8k__{allocation}__s{seed}.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("status") != "completed_validated":
        raise SystemExit(f"cell not completed: {path}")
    return float(rec["accuracy"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning-dir", default="results/quality/reasoning")
    ap.add_argument("--attempt-bf16", default="reasoning-gsm8k-3seed-statebf16-20260808")
    ap.add_argument("--attempt-fp16", default="reasoning-gsm8k-3seed-20260808")
    ap.add_argument("--out", default="results/quality/gsm8k-statebf16-analysis-20260808.json")
    args = ap.parse_args()
    reasoning_dir = Path(args.reasoning_dir)
    seeds = [7, 42, 2026]

    rows = []
    for seed in seeds:
        fp16 = load(reasoning_dir, args.attempt_fp16, "fp16", seed)
        bf16 = load(reasoning_dir, args.attempt_bf16, "fp16_statebf16", seed)
        rows.append(
            {
                "seed": seed,
                "fp16_state_fp32_acc": fp16,
                "fp16_state_bf16_acc": bf16,
                "delta_bf16_minus_fp32": round(bf16 - fp16, 4),
            }
        )

    diffs = [r["delta_bf16_minus_fp32"] for r in rows]
    mean_d = statistics.mean(diffs)
    half = t_half(len(diffs), statistics.stdev(diffs)) if len(diffs) > 1 else 0.0
    result = {
        "schema_version": 1,
        "bench": "gsm8k",
        "protocol": (
            "reasoning_bench.py --disable-thinking, 200 samples, greedy; "
            "fp16_statebf16 adds mamba_ssm_cache_dtype=bfloat16 to the fp16 engine args"
        ),
        "rows": rows,
        "paired_delta": {
            "mean": round(mean_d, 4),
            "ci95": [round(mean_d - half, 4), round(mean_d + half, 4)],
            "n_seeds": len(diffs),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
