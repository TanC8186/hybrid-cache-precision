"""Aggregate Q-stacking PPL cells (int4 KV x {fp32,bf16} state) into paired
tables and compare the stacking cost against the fp16-KV state matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def t_half(n: int, sd: float) -> float:
    df = n - 1
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
    return table.get(df, 1.96) * sd / math.sqrt(n)


def load_cell(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen: dict[tuple[str, str], float] = {}
    for r in rows:
        seen.setdefault((r["bits"], r["seed"]), float(r["ppl"]))
    values = [ppl for (bits, _), ppl in seen.items() if bits == "4"]
    if len(values) != 3:
        raise SystemExit(f"unexpected seed rows for {path.name}: {values}")
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/ppl-stacking")
    ap.add_argument("--attempt", default="ppl-stacking-20260809")
    ap.add_argument("--fp16-state-analysis", default="results/quality/ppl-state-dtype-analysis-20260808.json")
    ap.add_argument("--out", default="results/quality/ppl-stacking-analysis-20260809.json")
    args = ap.parse_args()

    cells: dict[tuple[str, str], list[float]] = {}
    for path in sorted(Path(args.dir).glob(f"{args.attempt}__*")):
        if not path.name.endswith(".csv.seeds.csv"):
            continue
        core = path.name[len(args.attempt) + 2 : -len(".csv.seeds.csv")]
        parts = core.split("__")
        if len(parts) != 3:
            raise SystemExit(f"unexpected filename: {path.name}")
        corpus, state, model = parts
        if model != "2b" or state not in ("statefp32", "statebf16"):
            continue
        cells[(corpus, state)] = load_cell(path)

    missing = [(c, s) for c in ("c4", "pg19") for s in ("statefp32", "statebf16") if (c, s) not in cells]
    if missing:
        raise SystemExit(f"incomplete cells: {missing}")

    tables = {}
    for corpus in ("c4", "pg19"):
        fp32 = cells[(corpus, "statefp32")]
        bf16 = cells[(corpus, "statebf16")]
        diffs = [b - f for b, f in zip(bf16, fp32)]
        mean_d = statistics.mean(diffs)
        half = t_half(len(diffs), statistics.stdev(diffs))
        tables[corpus] = {
            "fp32_state_mean": round(statistics.mean(fp32), 4),
            "bf16_state_mean": round(statistics.mean(bf16), 4),
            "delta_bf16_vs_fp32": round(mean_d, 4),
            "ci95_delta": [round(mean_d - half, 4), round(mean_d + half, 4)],
        }

    fp16_state = {}
    fp16_path = Path(args.fp16_state_analysis)
    if fp16_path.exists():
        fp16_state = json.loads(fp16_path.read_text(encoding="utf-8")).get("tables", {}).get("2b", [])
        fp16_state = {r["corpus"]: r for r in fp16_state}

    stacking_cost = {}
    for corpus in ("c4", "pg19"):
        base = fp16_state.get(corpus, {})
        stacking_cost[corpus] = {
            "fp16kv_state_delta": base.get("delta_bf16_vs_fp32"),
            "int4kv_state_delta": tables[corpus]["delta_bf16_vs_fp32"],
            "note": (
                "state-bf16 marginal cost under int4 KV vs under fp16 KV; "
                "if CI overlap, stacking adds no measurable state-precision cost"
            ),
        }

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "protocol": (
            "hybrid_premise.py --bits 4 --seeds 7,42,2026 --num-seqs 5 "
            "--max-len 2048 --chunk 128; statefp32=--state-dtype auto, "
            "statebf16=--state-dtype bfloat16 (write-boundary rounding)"
        ),
        "tables": tables,
        "stacking_cost_vs_fp16_kv": stacking_cost,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
