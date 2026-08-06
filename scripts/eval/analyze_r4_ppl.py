"""Analyze R4 PPL samples: per-seed PPL, paired differences vs fp16, t-CIs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def t_half(n: int, sd: float) -> float:
    """95% t critical value for small n (df=n-1)."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
    df = n - 1
    if df not in table:
        raise ValueError(f"unsupported df={df}")
    return table[df] * sd / math.sqrt(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/r4-ppl/r4-ppl-20260806")
    ap.add_argument("--out", default="results/quality/r4-ppl-analysis.json")
    args = ap.parse_args()

    samples: dict[str, dict[int, float]] = {}
    for path in sorted(Path(args.dir).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        samples.setdefault(rec["allocation"], {})[rec["seed"]] = rec["ppl"]

    allocs = ["fp16", "uniform_int4", "packed_per_layer"]
    missing = [a for a in allocs if a not in samples or len(samples[a]) < 3]
    if missing:
        print(f"incomplete samples for: {missing}")
        raise SystemExit(1)

    seeds = sorted(samples["fp16"])
    rows = []
    for alloc in allocs:
        ppls = [samples[alloc][s] for s in seeds]
        mean = statistics.mean(ppls)
        sd = statistics.stdev(ppls)
        rows.append(
            {
                "allocation": alloc,
                "mean_ppl": round(mean, 4),
                "std_ppl": round(sd, 4),
                "seeds": seeds,
            }
        )

    def paired(a: str, b: str) -> dict:
        diffs = [samples[a][s] - samples[b][s] for s in seeds]
        m = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        half = t_half(len(seeds), sd)
        rel = m / statistics.mean(samples[b].values()) * 100
        return {
            "vs": b,
            "mean_diff": round(m, 4),
            "sd_diff": round(sd, 4),
            "ci95": [round(m - half, 4), round(m + half, 4)],
            "rel_pct": round(rel, 2),
        }

    result = {
        "schema_version": 1,
        "rows": rows,
        "paired": {
            "uniform_int4_vs_fp16": paired("uniform_int4", "fp16"),
            "packed_per_layer_vs_fp16": paired("packed_per_layer", "fp16"),
            "packed_per_layer_vs_uniform_int4": paired("packed_per_layer", "uniform_int4"),
        },
        "source_dir": args.dir,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
