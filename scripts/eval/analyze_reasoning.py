"""Analyze reasoning benchmark cells: per (bench, allocation) mean over 3
seeds and paired deltas vs fp16."""

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/reasoning")
    ap.add_argument("--attempt", default="reasoning-20260807")
    ap.add_argument("--out", default="results/quality/reasoning-analysis.json")
    args = ap.parse_args()

    allocs = ["fp16", "uniform_int4", "packed_per_layer", "turboquant_k8v4", "turboquant_4bit_nc"]
    benches = ["gsm8k", "mmlu", "aime25"]
    seeds = [7]
    by_key: dict[tuple, dict] = {}
    for path in sorted((Path(args.dir) / args.attempt).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        by_key[(rec["bench"], rec["allocation"], rec["seed"])] = rec

    missing = [
        (b, a, s) for b in benches for a in allocs for s in seeds if (b, a, s) not in by_key
    ]
    if missing:
        raise SystemExit(f"incomplete cells: {len(missing)} missing, e.g. {missing[:5]}")

    rows = []
    for b in benches:
        row = {"bench": b, "num_samples": by_key[(b, "fp16", seeds[0])]["num_samples"]}
        for a in allocs:
            scores = [by_key[(b, a, s)]["accuracy"] for s in seeds]
            row[a] = round(statistics.mean(scores), 4)
            row[f"std_{a}"] = round(statistics.stdev(scores), 4)
        for a in allocs:
            if a == "fp16":
                continue
            diffs = [
                by_key[(b, a, s)]["accuracy"] - by_key[(b, "fp16", s)]["accuracy"] for s in seeds
            ]
            m = statistics.mean(diffs)
            sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
            half = t_half(len(diffs), sd) if len(diffs) > 1 else 0.0
            row[f"delta_{a}"] = round(m, 4)
            row[f"ci95_{a}"] = (
                [round(m - half, 4), round(m + half, 4)] if len(diffs) > 1 else None
            )
        rows.append(row)

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "rows": rows,
        "extraction_note": "gsm8k=last number; mmlu=last A-D; aime25=last integer",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
