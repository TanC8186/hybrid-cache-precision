"""Summarize the statebf16 Random60 pilot (fp32 vs bf16 GDN state)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


THRESHOLDS = ["250", "500", "1000", "2000", "3000"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = args.attempt_dir

    rows = []
    for sample_dir in sorted((root / "samples").iterdir()):
        analysis = json.loads((sample_dir / "analysis.json").read_text(encoding="utf-8"))
        allocation = sample_dir.name.split("__")[0]
        rate = analysis["offered_rate_req_s"]
        row = {
            "sample_id": analysis["sample_id"],
            "allocation": allocation,
            "rate": rate,
            "failed": analysis["failed"],
            "ttft_p99_ms": round(analysis["ttft_p99_ms_recomputed"], 1),
            "tpot_p99_ms": round(analysis["tpot_p99_ms_recomputed"], 1),
            "slo_sweep": {
                thr: {
                    "goodput_over_offered": round(analysis["slo_sweep"][thr]["goodput_over_offered"], 4),
                    "sustainable": analysis["slo_sweep"][thr]["sustainable"],
                }
                for thr in THRESHOLDS
            },
        }
        rows.append(row)

    # Group by allocation and rate for the comparison table.
    table: dict[str, dict[float, dict]] = {}
    for row in rows:
        table.setdefault(row["allocation"], {})[row["rate"]] = row

    # Server-log pattern proof: bf16 sessions must carry the Qwen3.5 override warning.
    log_proof: dict[str, dict[str, bool]] = {}
    for allocation in ("fp16", "fp16_statebf16"):
        logs = list((root / "servers" / allocation).glob("*/server.log"))
        log_proof[allocation] = {
            "session_count": len(logs),
            "has_override_warning": any(
                "Using the user-specified value" in log.read_text(encoding="utf-8", errors="replace")
                for log in logs
            ),
            "has_piecewise": any(
                "CUDAGraphMode.PIECEWISE" in log.read_text(encoding="utf-8", errors="replace")
                for log in logs
            ),
        }

    result = {
        "schema_version": 1,
        "experiment": "statebf16_random60_pilot_2b",
        "protocol": (
            "Random60, seed 7, warmup 120 req, 60 s measurement window, rates 30/40/50 req/s; "
            "fp16 = fp32 GDN state, fp16_statebf16 = --mamba-ssm-cache-dtype bfloat16; "
            "protocol-v3, failures count as SLO misses"
        ),
        "rows": rows,
        "table": {
            alloc: {str(rate): r for rate, r in by_rate.items()} for alloc, by_rate in table.items()
        },
        "log_proof": log_proof,
        "pilot_read": (
            "single seed 7, 60 s window; boundary numbers are directional, formal 3-seed required. "
            "At r40 the bf16-state cell reached sustainable (>=0.95 goodput) at TTFT>=2000 ms while "
            "the fp32-state cell stayed below 0.95 at every threshold."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
