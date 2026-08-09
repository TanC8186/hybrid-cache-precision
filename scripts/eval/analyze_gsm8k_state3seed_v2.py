"""GSM8K state-direction seed-subsampled analysis (ARS R1 + S3 power).

Allocations: fp16 (fp32 state), fp16_statebf16, uniform_int4 (fp32 state),
uniform_int4_statebf16. Paired 95% t-CI is over the per-seed deltas; the same
seed maps to the same 200-question subset in every allocation. Fail-closed:
every cell must record seed_semantics and 200 sampled_indices.

Power/MDE reporting (nature-statistics): for each non-fp16 allocation we report
the exact two-sided paired-t p value, the 80%-power minimum detectable effect
(MDE = (t_{1-alpha/2,df} + t_{0.8,df}) * sd/sqrt(n)), and power at the observed
effect. Pre-registered decision rule: significance only if the paired CI
excludes 0; a CI containing 0 is reported as point estimate with wide
uncertainty, never reframed as a trend.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from scipy import stats as _stats
except ImportError:  # pragma: no cover
    _stats = None


ALLOCATIONS = ["fp16", "fp16_statebf16", "uniform_int4", "uniform_int4_statebf16"]


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
    ap.add_argument("--seeds", default="7,42,2026")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power-target", type=float, default=0.80)
    args = ap.parse_args()
    allocations = [a.strip() for a in args.allocations.split(",") if a.strip()]
    for a in allocations:
        if a not in ALLOCATIONS:
            raise SystemExit(f"unknown allocation: {a}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        raise SystemExit("need at least 2 seeds")

    def power_metrics(diffs: list[float]) -> dict:
        n = len(diffs)
        df = n - 1
        mean_d = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if n > 1 else 0.0
        se = sd / math.sqrt(n)
        half = t_half(n, sd)
        if _stats is not None and se > 0:
            t_stat = mean_d / se
            p = float(2.0 * _stats.t.sf(abs(t_stat), df))
            t_crit = float(_stats.t.ppf(1 - args.alpha / 2, df))
            t_power = float(_stats.t.ppf(args.power_target, df))
            mde = (t_crit + t_power) * se
            ncp = abs(mean_d) / se
            power = float(_stats.nct.sf(t_crit, df, ncp) + _stats.nct.cdf(-t_crit, df, ncp))
        else:
            p = mde = power = None
        return {
            "n_seeds": n,
            "mean": round(mean_d, 4),
            "sd": round(sd, 4),
            "se": round(se, 4),
            "ci95": [round(mean_d - half, 4), round(mean_d + half, 4)],
            "p_value": round(p, 5) if p is not None and p == p else None,
            "ci_excludes_zero": (mean_d - half > 0) or (mean_d + half < 0),
            "mde_80_power": round(mde, 4) if mde is not None else None,
            "power_at_observed": round(power, 4) if power is not None and power == power else None,
        }

    cells: dict[tuple[str, int], dict] = {}
    for path in (Path(args.dir) / args.attempt).glob("gsm8k__*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        cells[(rec["allocation"], rec["seed"])] = rec

    missing = [(a, s) for a in allocations for s in seeds if (a, s) not in cells]
    if missing:
        raise SystemExit(f"incomplete cells: {missing}")

    for (a, s), rec in cells.items():
        if rec.get("seed_semantics") is None:
            raise SystemExit(f"cell missing seed_semantics: {a} seed={s}")
        indices = rec.get("sampled_indices") or []
        if rec["bench"] == "gsm8k" and len(indices) != 200:
            raise SystemExit(f"cell missing 200 sampled_indices: {a} seed={s} ({len(indices)})")

    fp16 = {s: cells[("fp16", s)]["accuracy"] for s in seeds}
    rows = []
    for alloc in allocations:
        scores = [cells[(alloc, s)]["accuracy"] for s in seeds]
        row = {
            "allocation": alloc,
            "n": cells[(alloc, seeds[0])]["num_samples"],
            "n_seeds": len(seeds),
            "per_seed": {str(s): round(v, 4) for s, v in zip(seeds, scores)},
            "mean": round(statistics.mean(scores), 4),
            "std": round(statistics.stdev(scores), 4),
        }
        if alloc != "fp16":
            diffs = [v - fp16[s] for s, v in zip(seeds, scores)]
            pm = power_metrics(diffs)
            row["delta_vs_fp16"] = pm["mean"]
            row["ci95_vs_fp16"] = pm["ci95"]
            row["paired_t"] = {
                "p_value": pm["p_value"],
                "ci_excludes_zero": pm["ci_excludes_zero"],
                "mde_80_power": pm["mde_80_power"],
                "power_at_observed": pm["power_at_observed"],
            }
        rows.append(row)

    stacking_marginal = None
    if "uniform_int4" in allocations and "uniform_int4_statebf16" in allocations:
        int4 = {s: cells[("uniform_int4", s)]["accuracy"] for s in seeds}
        stack = {s: cells[("uniform_int4_statebf16", s)]["accuracy"] for s in seeds}
        stack_diffs = [stack[s] - int4[s] for s in seeds]
        pm = power_metrics(stack_diffs)
        stacking_marginal = {
            "compare": "uniform_int4_statebf16 vs uniform_int4",
            "per_seed_delta": {str(s): round(stack[s] - int4[s], 4) for s in seeds},
            "mean": pm["mean"],
            "ci95": pm["ci95"],
            "paired_t": {
                "p_value": pm["p_value"],
                "ci_excludes_zero": pm["ci_excludes_zero"],
                "mde_80_power": pm["mde_80_power"],
                "power_at_observed": pm["power_at_observed"],
            },
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
        "power_plan": {
            "alpha": args.alpha,
            "power_target": args.power_target,
            "n_seeds": len(seeds),
            "mde_formula": "MDE = (t_{1-alpha/2,df} + t_{power,df}) * sd / sqrt(n)",
            "decision_rule": "significance only if paired CI excludes 0; otherwise point estimate with uncertainty",
        },
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
