"""Analyze the per-layer state-dtype sensitivity scan (decision gate)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/state-sensitivity")
    ap.add_argument("--attempt", default="state-sensitivity-20260809")
    ap.add_argument("--out", default="results/quality/state-sensitivity-analysis-20260809.json")
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
    rows = []
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
        for corpus in ("c4", "pg19"):
            r = records[corpus]["rows"][config]
            lo, hi = r["ci95_delta"]
            sensitive = lo > 0 or hi < 0
            sensitive_any = sensitive_any or sensitive
            row[f"{corpus}_delta"] = r["delta_vs_fp32_mean"]
            row[f"{corpus}_ci95"] = r["ci95_delta"]
            row[f"{corpus}_sensitive"] = sensitive
        row["sensitive_any"] = sensitive_any
        rows.append(row)

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "protocol": meta.get("protocol"),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
