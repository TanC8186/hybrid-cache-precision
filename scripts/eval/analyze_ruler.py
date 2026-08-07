"""Analyze the RULER-subset quality matrix.

Per (task, length, allocation): mean over 3 seeds of the official
string_match_all score, plus overall and paired deltas vs fp16.
"""

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
    ap.add_argument("--ruler-dir", default="results/quality/ruler-subset")
    ap.add_argument("--attempt", default="ruler-subset-20260807")
    ap.add_argument("--out", default="results/quality/ruler-subset-analysis.json")
    args = ap.parse_args()

    allocs = ["fp16", "uniform_int4", "packed_per_layer", "turboquant_k8v4", "turboquant_4bit_nc"]
    tasks = [
        "ruler_niah_single",
        "ruler_niah_multikey",
        "ruler_niah_multivalue",
        "ruler_niah_multiquery",
        "ruler_vt",
        "ruler_cwe",
        "ruler_fwe",
    ]
    lengths = [4096, 8192]
    seeds = [7, 42, 2026]

    by_key: dict[tuple, dict] = {}
    for path in sorted((Path(args.ruler_dir) / args.attempt).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        key = (rec["task"], rec["length"], rec["allocation"], rec["seed"])
        by_key[key] = rec

    expected = {(t, l, a, s) for t in tasks for l in lengths for a in allocs for s in seeds}
    missing = sorted(expected - set(by_key))
    if missing:
        raise SystemExit(f"incomplete cells: {len(missing)} missing, e.g. {missing[:5]}")

    table = []
    for t in tasks:
        for l in lengths:
            row = {"task": t, "length": l}
            for a in allocs:
                scores = [by_key[(t, l, a, s)]["accuracy"] for s in seeds]
                row[a] = round(statistics.mean(scores), 2)
            for a in allocs:
                if a == "fp16":
                    continue
                diffs = [
                    by_key[(t, l, a, s)]["accuracy"] - by_key[(t, l, "fp16", s)]["accuracy"]
                    for s in seeds
                ]
                m = statistics.mean(diffs)
                sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
                half = t_half(len(diffs), sd)
                row[f"delta_{a}"] = round(m, 2)
                row[f"ci95_{a}"] = [round(m - half, 2), round(m + half, 2)]
            table.append(row)

    overall = {}
    for a in allocs:
        scores = [by_key[(t, l, a, s)]["accuracy"] for t in tasks for l in lengths for s in seeds]
        overall[a] = {
            "n_cells": len(scores),
            "mean": round(statistics.mean(scores), 2),
            "sd": round(statistics.stdev(scores), 2),
        }

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "metric": "string_match_all (official RULER)",
        "tasks": tasks,
        "lengths": lengths,
        "seeds": seeds,
        "rows": table,
        "overall": overall,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
