"""GSM8K 3-seed analysis: per-allocation mean, paired 95% t-CI vs fp16."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ALLOCATIONS = ["fp16", "uniform_int4", "packed_per_layer", "turboquant_k8v4", "turboquant_4bit_nc"]


def t_half(n: int, sd: float) -> float:
    df = n - 1
    table = {1: 12.706, 2: 4.303, 3: 3.182}
    return table.get(df, 1.96) * sd / math.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/reasoning")
    ap.add_argument("--attempt", default="reasoning-gsm8k-3seed-20260808")
    ap.add_argument("--out", default="results/quality/reasoning-gsm8k-3seed-analysis.json")
    args = ap.parse_args()

    cells: dict[tuple[str, int], dict] = {}
    for path in (Path(args.dir) / args.attempt).glob("gsm8k__*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        cells[(rec["allocation"], rec["seed"])] = rec

    seeds = [7, 42, 2026]
    missing = [(a, s) for a in ALLOCATIONS for s in seeds if (a, s) not in cells]
    if missing:
        raise SystemExit(f"incomplete: {missing}")

    fp16 = {s: cells[("fp16", s)]["accuracy"] for s in seeds}
    rows = []
    for alloc in ALLOCATIONS:
        scores = [cells[(alloc, s)]["accuracy"] for s in seeds]
        row = {
            "allocation": alloc,
            "n": cells[(alloc, seeds[0])]["num_samples"],
            "per_seed": {str(s): round(v, 4) for s, v in zip(seeds, scores)},
            "mean": round(statistics.mean(scores), 4),
            "std": round(statistics.stdev(scores), 4),
        }
        if alloc != "fp16":
            diffs = [v - fp16[s] for s, v in zip(seeds, scores)]
            mean_d = statistics.mean(diffs)
            half = t_half(len(diffs), statistics.stdev(diffs))
            row["delta_vs_fp16"] = round(mean_d, 4)
            row["ci95_vs_fp16"] = [round(mean_d - half, 4), round(mean_d + half, 4)]
        rows.append(row)

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "bench": "gsm8k",
        "protocol": "no-think, greedy, max_tokens=1024, 200 samples, seeds 7/42/2026",
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
