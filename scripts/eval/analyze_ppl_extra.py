"""Aggregate C4/PG19 PPL cells (canonical 3-seed protocol) into tables."""

from __future__ import annotations

import argparse
import csv
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
    ap.add_argument("--dir", default="results/quality/ppl-extra")
    ap.add_argument("--attempt", default="ppl-extra-20260807")
    ap.add_argument("--out", default="results/quality/ppl-extra-analysis-20260807.json")
    args = ap.parse_args()

    cells: dict[tuple[str, str, str], list[float]] = {}
    for path in sorted(Path(args.dir).glob(f"{args.attempt}__*")):
        if not path.name.endswith(".csv.seeds.csv"):
            continue
        core = path.name[len(args.attempt) + 2 : -len(".csv.seeds.csv")]
        parts = core.split("__")
        if len(parts) != 3:
            raise SystemExit(f"unexpected filename: {path.name}")
        corpus, alloc, model = parts[0], parts[1], parts[2]
        if model not in ("2b", "9b"):
            continue  # legacy master-named artifacts from the naming bug
        requested_bits = {"fp16": "16", "uniform": "4", "packed": "4"}[alloc]
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        # hybrid_premise always emits an extra FP16 baseline row; for bits=16 runs
        # this duplicates the requested row. Deduplicate by (bits, seed).
        seen: dict[tuple[str, str], float] = {}
        for r in rows:
            key = (r["bits"], r["seed"])
            seen.setdefault(key, float(r["ppl"]))
        values = [ppl for (bits, _), ppl in seen.items() if bits == requested_bits]
        if len(values) != 3:
            raise SystemExit(f"unexpected seed rows for {path.name}: {values}")
        cells[(corpus, alloc, model)] = values

    corpora = ["c4", "pg19"]
    allocs = ["fp16", "uniform", "packed"]
    model_names = sorted({m for (_, _, m) in cells if m in ("2b", "9b")})
    missing = [(c, a, m) for c in corpora for a in allocs for m in model_names if (c, a, m) not in cells]
    if missing:
        raise SystemExit(f"incomplete cells: {missing}")
    if len(cells) != len(corpora) * len(allocs) * len(model_names):
        raise SystemExit(f"unexpected cell count: {len(cells)}")

    tables: dict[str, list[dict]] = {}
    for m in model_names:
        rows = []
        for corpus in corpora:
            fp16 = cells[(corpus, "fp16", m)]
            row = {"corpus": corpus, "fp16_mean": round(statistics.mean(fp16), 4)}
            for alloc in ("uniform", "packed"):
                values = cells[(corpus, alloc, m)]
                row[f"{alloc}_mean"] = round(statistics.mean(values), 4)
                diffs = [v - f for v, f in zip(values, fp16)]
                mean_d = statistics.mean(diffs)
                half = t_half(len(diffs), statistics.stdev(diffs)) if len(diffs) > 1 else 0.0
                row[f"delta_{alloc}_vs_fp16"] = round(mean_d, 4)
                row[f"ci95_{alloc}_vs_fp16"] = [round(mean_d - half, 4), round(mean_d + half, 4)]
            rows.append(row)
        tables[m] = rows

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "protocol": "hybrid_premise.py --seeds 7,42,2026 --num-seqs 5 --max-len 2048 --chunk 128",
        "dedupe_note": (
            "hybrid_premise.py always appends an FP16 baseline row; bits=16 cells therefore "
            "contain 6 rows in *.seeds.csv. Analysis deduplicates by (bits, seed) and uses "
            "the 3 unique requested-bit rows."
        ),
        "allocations": {"fp16": "--bits 16", "uniform": "--bits 4", "packed": "--bits 4 --layer-bits '{\"23\":16}'"},
        "tables": tables,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
