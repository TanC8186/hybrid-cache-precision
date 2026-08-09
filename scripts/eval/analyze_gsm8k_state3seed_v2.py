"""GSM8K state-direction 3-seed v2 analysis (seed-subsampled protocol).

Allocations: fp16 (fp32 state), fp16_statebf16, uniform_int4 (fp32 state),
uniform_int4_statebf16. Paired 95% t-CI is over the three per-seed deltas;
the same seed maps to the same 200-question subset in every allocation.
Fail-closed: every cell must record seed_semantics and 200 sampled_indices.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ALLOCATIONS = ["fp16", "fp16_statebf16", "uniform_int4", "uniform_int4_statebf16"]
SEEDS = [7, 42, 2026]


def t_half(n: int, sd: float) -> float:
    df = n - 1
    table = {1: 12.706, 2: 4.303, 3: 3.182}
    return table.get(df, 1.96) * sd / math.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/reasoning")
    ap.add_argument("--attempt", default="reasoning-gsm8k-state3seed-v2-20260809")
    ap.add_argument("--out", default="results/quality/gsm8k-state3seed-v2-analysis-20260809.json")
    ap.add_argument("--allocations", default=",".join(ALLOCATIONS))
    args = ap.parse_args()
    allocations = [a.strip() for a in args.allocations.split(",") if a.strip()]
    for a in allocations:
        if a not in ALLOCATIONS:
            raise SystemExit(f"unknown allocation: {a}")

    cells: dict[tuple[str, int], dict] = {}
    for path in (Path(args.dir) / args.attempt).glob("gsm8k__*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        cells[(rec["allocation"], rec["seed"])] = rec

    missing = [(a, s) for a in allocations for s in SEEDS if (a, s) not in cells]
    if missing:
        raise SystemExit(f"incomplete cells: {missing}")

    for (a, s), rec in cells.items():
        if rec.get("seed_semantics") is None:
            raise SystemExit(f"cell missing seed_semantics: {a} seed={s}")
        indices = rec.get("sampled_indices") or []
        if rec["bench"] == "gsm8k" and len(indices) != 200:
            raise SystemExit(f"cell missing 200 sampled_indices: {a} seed={s} ({len(indices)})")

    fp16 = {s: cells[("fp16", s)]["accuracy"] for s in SEEDS}
    rows = []
    for alloc in allocations:
        scores = [cells[(alloc, s)]["accuracy"] for s in SEEDS]
        row = {
            "allocation": alloc,
            "n": cells[(alloc, SEEDS[0])]["num_samples"],
            "per_seed": {str(s): round(v, 4) for s, v in zip(SEEDS, scores)},
            "mean": round(statistics.mean(scores), 4),
            "std": round(statistics.stdev(scores), 4),
        }
        if alloc != "fp16":
            diffs = [v - fp16[s] for s, v in zip(SEEDS, scores)]
            mean_d = statistics.mean(diffs)
            half = t_half(len(diffs), statistics.stdev(diffs))
            row["delta_vs_fp16"] = round(mean_d, 4)
            row["ci95_vs_fp16"] = [round(mean_d - half, 4), round(mean_d + half, 4)]
        rows.append(row)

    stacking_marginal = None
    if "uniform_int4" in allocations and "uniform_int4_statebf16" in allocations:
        int4 = {s: cells[("uniform_int4", s)]["accuracy"] for s in SEEDS}
        stack = {s: cells[("uniform_int4_statebf16", s)]["accuracy"] for s in SEEDS}
        stack_diffs = [stack[s] - int4[s] for s in SEEDS]
        stack_mean = statistics.mean(stack_diffs)
        stack_half = t_half(len(stack_diffs), statistics.stdev(stack_diffs))
        stacking_marginal = {
            "compare": "uniform_int4_statebf16 vs uniform_int4",
            "per_seed_delta": {str(s): round(stack[s] - int4[s], 4) for s in SEEDS},
            "mean": round(stack_mean, 4),
            "ci95": [round(stack_mean - stack_half, 4), round(stack_mean + stack_half, 4)],
        }

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "bench": "gsm8k",
        "protocol": (
            "no-think, greedy, max_tokens=1024, 200 rows sampled per seed "
            "(random_state=seed, no replacement); same seed -> same subset in "
            "every allocation; engine seed kept for provenance only"
        ),
        "rows": rows,
        "stacking_marginal": stacking_marginal,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
