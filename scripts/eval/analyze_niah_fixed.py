"""Analyze the NIAH rerun (max_tokens>=128) across all 5 allocations.

Reports per-allocation cell-mean accuracy, overall needle accuracy,
hit_final (answer after </think>) diagnostics, and paired t-CI deltas
vs fp16 (18 matched cells, df=17).
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
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    crit = table.get(df, 1.96)
    return crit * sd / math.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--niah-dir", default="results/quality/niah-fixed")
    ap.add_argument("--attempt", default="niah-fixed-20260807")
    ap.add_argument("--out", default="results/quality/niah-fixed-analysis.json")
    args = ap.parse_args()

    allocs = ["fp16", "uniform_int4", "packed_per_layer", "turboquant_k8v4", "turboquant_4bit_nc"]
    by_alloc: dict[str, list[dict]] = {a: [] for a in allocs}
    for path in sorted((Path(args.niah_dir) / args.attempt).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        if rec.get("max_tokens", 0) < 128:
            raise SystemExit(f"refusing stale protocol (max_tokens={rec.get('max_tokens')}): {path}")
        by_alloc.setdefault(rec["allocation"], []).append(rec)

    missing = [a for a in allocs if len(by_alloc[a]) != 18]
    if missing:
        raise SystemExit(f"incomplete cells: { {a: len(by_alloc[a]) for a in allocs} }")

    def cell_key(rec: dict) -> tuple:
        return (rec["seed"], rec["depth_pct"], rec["max_len"])

    cells = {a: {cell_key(r): r for r in recs} for a, recs in by_alloc.items()}
    common = sorted(set(cells["fp16"]) & set(cells["uniform_int4"]) & set(cells["packed_per_layer"])
                    & set(cells["turboquant_k8v4"]) & set(cells["turboquant_4bit_nc"]))
    if len(common) != 18:
        raise SystemExit(f"matched cells incomplete: {len(common)}/18")

    def needle_stats(rec: dict) -> tuple[int, int, int, int]:
        hits = sum(1 for c in rec["cases"] if c["hit"])
        hit_final = sum(1 for c in rec["cases"] if c["hit_final"])
        think_tag = sum(1 for c in rec["cases"] if "<think>" in c["answer"].lower())
        return hits, hit_final, think_tag, len(rec["cases"])

    rows = []
    for a in allocs:
        total = sum(needle_stats(rec)[3] for rec in cells[a].values())
        hits = sum(needle_stats(rec)[0] for rec in cells[a].values())
        hit_final = sum(needle_stats(rec)[1] for rec in cells[a].values())
        think_tag = sum(needle_stats(rec)[2] for rec in cells[a].values())
        rows.append(
            {
                "allocation": a,
                "n_cells": 18,
                "n_needles": total,
                "accuracy_mean_cell": round(statistics.mean(rec["accuracy"] for rec in cells[a].values()), 4),
                "accuracy_std_cell": round(statistics.stdev(rec["accuracy"] for rec in cells[a].values()), 4),
                "accuracy_overall_needles": round(hits / total, 4),
                "hit_final_overall": round(hit_final / total, 4),
                "think_tag_needles": think_tag,
            }
        )

    def paired(a: str, b: str) -> dict:
        diffs = [cells[a][c]["accuracy"] - cells[b][c]["accuracy"] for c in common]
        m = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        half = t_half(len(common), sd)
        return {
            "vs": b,
            "n_cells": len(common),
            "mean_diff": round(m, 4),
            "sd_diff": round(sd, 4),
            "ci95": [round(m - half, 4), round(m + half, 4)],
        }

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "protocol_note": "max_tokens>=128; hit = code anywhere in answer (same as R4/R5); hit_final = code after </think>",
        "rows": rows,
        "paired_vs_fp16": {
            a: paired(a, "fp16") for a in allocs if a != "fp16"
        },
        "cells": [{"seed": c[0], "depth_pct": c[1], "max_len": c[2]} for c in common],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
