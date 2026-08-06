"""Analyze R5 TurboQuant NIAH results vs the R4 fp16 baseline (same 18-cell grid)."""

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
    return table.get(df, 1.96) * sd / math.sqrt(n)


def load_niah(dir_path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(dir_path.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        out.setdefault(rec["allocation"], []).append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="results/quality/r4-niah/r4-20260806-niah")
    ap.add_argument("--turbo-dir", default="results/quality/r5-turboquant/r5-turboquant-20260806")
    ap.add_argument("--out", default="results/quality/r5-analysis.json")
    args = ap.parse_args()

    base = load_niah(Path(args.base_dir))
    turbo = load_niah(Path(args.turbo_dir))
    fp16 = {f"{r['seed']}:{r['depth_pct']}:{r['max_len']}": r["accuracy"] for r in base.get("fp16", [])}
    if len(fp16) < 18:
        raise SystemExit(f"fp16 baseline incomplete: {len(fp16)}/18")

    rows = []
    paired = {}
    for alloc in ("turboquant_k8v4", "turboquant_4bit_nc"):
        recs = turbo.get(alloc, [])
        cells = {f"{r['seed']}:{r['depth_pct']}:{r['max_len']}": r["accuracy"] for r in recs}
        if len(cells) < 18:
            raise SystemExit(f"{alloc} incomplete: {len(cells)}/18")
        accs = list(cells.values())
        diffs = [cells[k] - fp16[k] for k in cells]
        m = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        half = t_half(len(diffs), sd)
        rows.append(
            {
                "allocation": alloc,
                "mean_accuracy": round(statistics.mean(accs), 4),
                "n_cells": len(cells),
            }
        )
        paired[alloc] = {
            "vs": "fp16",
            "n_cells": len(diffs),
            "mean_diff": round(m, 4),
            "sd_diff": round(sd, 4),
            "ci95": [round(m - half, 4), round(m + half, 4)],
        }

    result = {
        "schema_version": 1,
        "fp16_baseline": {
            "mean_accuracy": round(statistics.mean(fp16.values()), 4),
            "n_cells": len(fp16),
        },
        "rows": rows,
        "paired": paired,
        "source": {"base_dir": args.base_dir, "turbo_dir": args.turbo_dir},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
