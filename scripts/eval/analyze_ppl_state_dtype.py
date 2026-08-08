"""Aggregate C4/PG19 PPL state-dtype cells (fp32 vs bf16) into paired tables."""

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


def load_cell(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen: dict[tuple[str, str], float] = {}
    for r in rows:
        key = (r["bits"], r["seed"])
        seen.setdefault(key, float(r["ppl"]))
    values = [ppl for (bits, _), ppl in seen.items() if bits == "16"]
    if len(values) != 3:
        raise SystemExit(f"unexpected seed rows for {path.name}: {values}")
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/ppl-state-dtype")
    ap.add_argument("--attempt", default="ppl-state-20260808")
    ap.add_argument("--out", default="results/quality/ppl-state-dtype-analysis-20260808.json")
    args = ap.parse_args()

    cells: dict[tuple[str, str, str], list[float]] = {}
    for path in sorted(Path(args.dir).glob(f"{args.attempt}__*")):
        if not path.name.endswith(".csv.seeds.csv"):
            continue
        core = path.name[len(args.attempt) + 2 : -len(".csv.seeds.csv")]
        parts = core.split("__")
        if len(parts) != 3:
            raise SystemExit(f"unexpected filename: {path.name}")
        corpus, state, model = parts
        if model not in ("2b", "9b") or state not in ("statefp32", "statebf16"):
            continue
        cells[(corpus, state, model)] = load_cell(path)

    corpora = ["c4", "pg19"]
    states = ["statefp32", "statebf16"]
    models = sorted({m for (_, _, m) in cells})
    missing = [(c, s, m) for c in corpora for s in states for m in models if (c, s, m) not in cells]
    if missing:
        raise SystemExit(f"incomplete cells: {missing}")
    if len(cells) != len(corpora) * len(states) * len(models):
        raise SystemExit(f"unexpected cell count: {len(cells)}")

    tables: dict[str, list[dict]] = {}
    for model in models:
        rows = []
        for corpus in corpora:
            fp32 = cells[(corpus, "statefp32", model)]
            bf16 = cells[(corpus, "statebf16", model)]
            diffs = [b - f for b, f in zip(bf16, fp32)]
            mean_d = statistics.mean(diffs)
            half = t_half(len(diffs), statistics.stdev(diffs)) if len(diffs) > 1 else 0.0
            pct_diffs = [(b / f - 1.0) * 100.0 for b, f in zip(bf16, fp32)]
            rows.append(
                {
                    "corpus": corpus,
                    "fp32_mean": round(statistics.mean(fp32), 4),
                    "bf16_mean": round(statistics.mean(bf16), 4),
                    "delta_bf16_vs_fp32": round(mean_d, 4),
                    "ci95_delta": [round(mean_d - half, 4), round(mean_d + half, 4)],
                    "delta_pct_mean": round(statistics.mean(pct_diffs), 4),
                }
            )
        tables[model] = rows

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "protocol": (
            "hybrid_premise.py --bits 16 --seeds 7,42,2026 --num-seqs 5 "
            "--max-len 2048 --chunk 128; statefp32=--state-dtype auto (no patch), "
            "statebf16=--state-dtype bfloat16 (cast at recurrent-state write boundary)"
        ),
        "dedupe_note": (
            "hybrid_premise.py always appends an FP16 baseline row; bits=16 cells "
            "contain 6 rows in *.seeds.csv. Analysis deduplicates by (bits, seed)."
        ),
        "tables": tables,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
