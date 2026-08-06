"""Analyze R4 quality samples: PPL paired CIs + NIAH accuracy paired diffs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def t_half(n: int, sd: float) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
    df = n - 1
    if df not in table:
        raise ValueError(f"unsupported df={df}")
    return table[df] * sd / math.sqrt(n)


def load_ppl_seeds(base: Path, allocs: list[str]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for alloc in allocs:
        path = base / f"{alloc}.csv.seeds.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = {}
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[int(r["seed"])] = float(r["ppl"])
        out[alloc] = rows
    return out


def load_niah(base: Path, attempt: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted((base / attempt).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        out.setdefault(rec["allocation"], []).append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppl-dir", default="results/quality/r4-ppl")
    ap.add_argument("--niah-dir", default="results/quality/r4-niah")
    ap.add_argument("--attempt", default="r4-20260806")
    ap.add_argument("--out", default="results/quality/r4-analysis.json")
    args = ap.parse_args()

    allocs = ["fp16", "uniform_int4", "packed_per_layer"]
    ppl = load_ppl_seeds(Path(args.ppl_dir), allocs)
    niah = load_niah(Path(args.niah_dir), f"{args.attempt}-niah")

    missing_ppl = [a for a in allocs if len(ppl.get(a, {})) < 3]
    missing_niah = [a for a in allocs if len(niah.get(a, [])) < 3]
    if missing_ppl or missing_niah:
        print(f"incomplete: ppl={missing_ppl} niah={missing_niah}")
        raise SystemExit(1)

    seeds = sorted(ppl["fp16"])

    def paired_ppl(a: str, b: str) -> dict:
        diffs = [ppl[a][s] - ppl[b][s] for s in seeds]
        m = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        half = t_half(len(seeds), sd)
        rel = m / statistics.mean(ppl[b].values()) * 100
        return {
            "vs": b,
            "mean_diff": round(m, 4),
            "sd_diff": round(sd, 4),
            "ci95": [round(m - half, 4), round(m + half, 4)],
            "rel_pct": round(rel, 2),
        }

    def cell_key(rec: dict) -> tuple:
        return (rec["seed"], rec["depth_pct"], rec["max_len"])

    niah_cells = {a: {cell_key(r): r["accuracy"] for r in recs} for a, recs in niah.items()}
    cells = sorted(set(niah_cells["fp16"]) & set(niah_cells["uniform_int4"]) & set(niah_cells["packed_per_layer"]))

    def paired_niah(a: str, b: str) -> dict:
        diffs = [niah_cells[a][c] - niah_cells[b][c] for c in cells]
        m = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        half = t_half(len(cells), sd) if len(cells) > 1 else 0.0
        return {"vs": b, "n_cells": len(cells), "mean_diff": round(m, 4), "ci95": [round(m - half, 4), round(m + half, 4)]}

    result = {
        "schema_version": 1,
        "ppl": {
            "rows": [
                {
                    "allocation": a,
                    "mean_ppl": round(statistics.mean(ppl[a].values()), 4),
                    "std_ppl": round(statistics.stdev(ppl[a].values()), 4),
                    "seeds": seeds,
                }
                for a in allocs
            ],
            "paired": {
                "uniform_int4_vs_fp16": paired_ppl("uniform_int4", "fp16"),
                "packed_per_layer_vs_fp16": paired_ppl("packed_per_layer", "fp16"),
                "packed_per_layer_vs_uniform_int4": paired_ppl("packed_per_layer", "uniform_int4"),
            },
        },
        "niah": {
            "cells": [{"seed": c[0], "depth_pct": c[1], "max_len": c[2]} for c in cells],
            "rows": [
                {
                    "allocation": a,
                    "mean_accuracy": round(statistics.mean([niah_cells[a][c] for c in cells]), 4),
                    "n_cells": len(cells),
                }
                for a in allocs
            ],
            "paired": {
                "uniform_int4_vs_fp16": paired_niah("uniform_int4", "fp16"),
                "packed_per_layer_vs_fp16": paired_niah("packed_per_layer", "fp16"),
                "packed_per_layer_vs_uniform_int4": paired_niah("packed_per_layer", "uniform_int4"),
            },
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
