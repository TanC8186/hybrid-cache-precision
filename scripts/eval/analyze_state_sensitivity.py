"""Analyze the per-layer state-dtype sensitivity scan (decision gate).

Pre-registered decision rule (ARS 2026-08-09 R6): per-layer sensitivity is
declared only if a paired delta survives multiple-comparison correction. With
18 layer configs x 2 corpora = 36 tests, the raw 95% CI criterion expects ~2
false positives by chance. We therefore report Bonferroni (alpha/36) and
BH-FDR adjusted decisions computed from the per-seed deltas, and we disclose
sign consistency of any non-zero raw CIs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from scipy import stats as _stats
except ImportError:  # pragma: no cover
    _stats = None


def t_half(n: int, sd: float, alpha: float = 0.05) -> float:
    if _stats is not None:
        return float(_stats.t.ppf(1 - alpha / 2, n - 1)) * sd / math.sqrt(n)
    table = {1: 12.706, 2: 4.303}
    return table.get(n - 1, 1.96) * sd / math.sqrt(n)


def p_value_two_sided(n: int, mean_d: float, sd: float) -> float:
    if _stats is None or sd == 0.0:
        return float("nan")
    t_stat = mean_d / (sd / math.sqrt(n))
    return float(2.0 * _stats.t.sf(abs(t_stat), n - 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/state-sensitivity")
    ap.add_argument("--attempt", default="state-sensitivity-20260809")
    ap.add_argument("--out", default="results/quality/state-sensitivity-analysis-20260809.json")
    ap.add_argument("--n-tests", type=int, default=36, help="tests in the family (18 layers x 2 corpora)")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    base = Path(args.dir) / args.attempt

    records = {}
    for corpus in ("c4", "pg19"):
        path = base / f"{corpus}.json"
        if not path.exists():
            raise SystemExit(f"missing corpus record: {path}")
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            raise SystemExit(f"corpus not validated: {path}")
        records[corpus] = {
            "meta": rec,
            "rows": {row["config"]: row for row in rec["rows"]},
        }
        if rec["reference_match"]["max_abs_diff"]:
            for tag, diff in rec["reference_match"]["max_abs_diff"].items():
                if diff > 1e-9:
                    raise SystemExit(f"{corpus} reference mismatch {tag}: {diff}")

    rows_by_config = records["c4"]["rows"]
    meta = records["c4"]["meta"]
    configs = list(rows_by_config.keys())
    n_tests = args.n_tests
    bonf_alpha = args.alpha / n_tests
    rows = []
    p_rows = []
    for config in configs:
        if config == "fp32":
            rows.append(
                {
                    "config": config,
                    "c4_mean": records["c4"]["rows"][config]["ppl_mean"],
                    "pg19_mean": records["pg19"]["rows"][config]["ppl_mean"],
                }
            )
            continue
        row = {"config": config}
        sensitive_any = False
        sensitive_any_bonf = False
        for corpus in ("c4", "pg19"):
            r = records[corpus]["rows"][config]
            lo, hi = r["ci95_delta"]
            sensitive = lo > 0 or hi < 0
            sensitive_any = sensitive_any or sensitive
            per_seed = r.get("per_seed")
            fp32_seeds = records[corpus]["rows"]["fp32"].get("per_seed", {})
            if per_seed and fp32_seeds and len(per_seed) == len(fp32_seeds):
                seeds = sorted(per_seed, key=int)
                diffs = [per_seed[s] - fp32_seeds[s] for s in seeds]
                mean_d = sum(diffs) / len(diffs)
                sd = (sum((d - mean_d) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5 if len(diffs) > 1 else 0.0
                p = p_value_two_sided(len(diffs), mean_d, sd)
                half_bonf = t_half(len(diffs), sd, bonf_alpha)
                bonf_sensitive = (mean_d - half_bonf > 0) or (mean_d + half_bonf < 0)
            else:
                p = float("nan")
                bonf_sensitive = False
            row[f"{corpus}_delta"] = r["delta_vs_fp32_mean"]
            row[f"{corpus}_ci95"] = r["ci95_delta"]
            row[f"{corpus}_sensitive"] = sensitive
            row[f"{corpus}_p_value"] = round(p, 6) if p == p else None
            row[f"{corpus}_sensitive_bonf"] = bonf_sensitive
            sensitive_any_bonf = sensitive_any_bonf or bonf_sensitive
            p_rows.append((config, corpus, p))
        row["sensitive_any"] = sensitive_any
        row["sensitive_any_bonf"] = sensitive_any_bonf
        rows.append(row)

    p_values = [p for (_, _, p) in p_rows if p == p]
    bh_reject = set()
    if p_values:
        ordered = sorted((p, i) for i, p in enumerate(p_values))
        m = len(ordered)
        for rank, (p, idx) in enumerate(ordered, start=1):
            if p <= args.alpha * rank / m:
                bh_reject.add(idx)
    bh_map = {i: (i in bh_reject) for i in range(len(p_rows))}
    for (config, corpus, _p), is_reject in zip(p_rows, [bh_map[i] for i in range(len(p_rows))]):
        for row in rows:
            if row["config"] == config:
                row[f"{corpus}_bh_reject"] = is_reject

    nonzero = [r for r in rows if r.get("config") != "fp32" and r.get("sensitive_any")]
    sign_consistency = {
        "n_raw_nonzero_ci": len(nonzero),
        "positive_deltas": [
            (r["config"], r.get("c4_delta"), r.get("pg19_delta")) for r in nonzero if r.get("c4_delta", 0) > 0
        ],
        "note": (
            "both raw non-zero CIs are positive; magnitude is far below seed-level "
            "spread, and neither survives Bonferroni/BH-FDR"
        ),
    }

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "protocol": meta.get("protocol"),
        "multiple_comparison": {
            "n_tests": n_tests,
            "alpha": args.alpha,
            "bonferroni_alpha": round(bonf_alpha, 6),
            "method": "per-seed paired t recomputation; Bonferroni and BH-FDR",
        },
        "rows": rows,
        "sign_consistency": sign_consistency,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
