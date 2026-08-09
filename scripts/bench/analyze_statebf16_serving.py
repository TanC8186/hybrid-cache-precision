"""Analyze statebf16 int4-KV protocol-v3 serving attempts (ARS 2026-08-09 R4).

Produces: sustainable boundary tables (allocation x workload x TTFT threshold),
paired goodput deltas (int4_statebf16 minus int4) per rate/threshold with 3-seed
95% t-CIs, and per-allocation server-log proof (int4_per_token_head /
Using the user-specified value / CUDAGraphMode.PIECEWISE).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


THRESHOLDS = ["250", "500", "1000", "2000", "3000"]


def t_half(n: int, sd: float) -> float:
    df = n - 1
    table = {1: 12.706, 2: 4.303, 3: 3.182}
    return table.get(df, 1.96) * sd / math.sqrt(n)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_attempt(raw_dir: Path, attempt_id: str) -> dict:
    attempt_dir = raw_dir / attempt_id
    if not attempt_dir.is_dir():
        raise SystemExit(f"missing attempt: {attempt_dir}")
    contract = load_json(attempt_dir / "attempt_contract.json")
    plan = {str(item["sample_id"]): item for item in contract["plan"]}
    samples = []
    for sample_dir in sorted((attempt_dir / "samples").glob("*")):
        if not sample_dir.is_dir():
            continue
        analysis = load_json(sample_dir / "analysis.json")
        status = load_json(sample_dir / "status.json")
        if sha256_file(sample_dir / "analysis.json") != status.get("analysis_sha256"):
            raise SystemExit(f"{sample_dir.name}: status.analysis_sha256 mismatch")
        if status["status"] != "completed_validated":
            continue
        expected = int(plan[sample_dir.name]["num_prompts"])
        if int(analysis["completed"]) + int(analysis["failed"]) != expected:
            raise SystemExit(
                f"{attempt_id}/{sample_dir.name}: denominator mismatch "
                f"(completed+failed != plan num_prompts {expected})"
            )
        samples.append(
            {
                "sample_id": sample_dir.name,
                "allocation": sample_dir.name.split("__")[0],
                "workload": sample_dir.name.split("__")[1],
                "offered_rate": float(analysis["offered_rate_req_s"]),
                "seed": int(sample_dir.name.split("__")[-1][1:]),
                "failed": int(analysis["failed"]),
                "ttft_p99_ms": round(float(analysis["ttft_p99_ms_recomputed"]), 1),
                "tpot_p99_ms": round(float(analysis["tpot_p99_ms_recomputed"]), 1),
                "goodput_over_offered": {
                    thr: float(analysis["slo_sweep"][thr]["goodput_over_offered"])
                    for thr in THRESHOLDS
                },
            }
        )
    expected_samples = len(contract["plan"])
    if len(samples) != expected_samples:
        raise SystemExit(
            f"{attempt_id}: completed {len(samples)}/{expected_samples} samples"
        )
    return {
        "attempt_id": attempt_id,
        "git_commit": contract.get("git_commit"),
        "vllm_source_commit": contract.get("vllm_source_commit"),
        "config_sha256": contract.get("config_sha256"),
        "samples": samples,
    }


def boundaries(samples: list[dict]) -> dict:
    keys = sorted({(s["allocation"], s["workload"]) for s in samples})
    out = {}
    for allocation, workload in keys:
        cell = [s for s in samples if s["allocation"] == allocation and s["workload"] == workload]
        seeds = {s["seed"] for s in cell}
        rates = sorted({s["offered_rate"] for s in cell})
        row = {}
        for threshold in THRESHOLDS:
            sustainable = []
            for rate in rates:
                rate_samples = [s for s in cell if s["offered_rate"] == rate]
                if len(rate_samples) != len(seeds):
                    raise SystemExit(f"missing seed sample: {allocation}/{workload} r={rate}")
                if all(s["goodput_over_offered"][threshold] >= 0.95 for s in rate_samples):
                    sustainable.append(rate)
            row[threshold] = max(sustainable) if sustainable else None
        out[f"{allocation}__{workload}"] = row
    return out


def paired_deltas(samples: list[dict]) -> list[dict]:
    workloads = sorted({s["workload"] for s in samples})
    rows = []
    for workload in workloads:
        fp32 = {s["seed"]: s for s in samples if s["allocation"] == "int4" and s["workload"] == workload}
        bf16 = {s["seed"]: s for s in samples if s["allocation"] == "int4_statebf16" and s["workload"] == workload}
        rates = sorted({s["offered_rate"] for s in fp32.values()})
        seeds = sorted(fp32)
        for rate in rates:
            for threshold in THRESHOLDS:
                diffs = [
                    bf16[s]["goodput_over_offered"][threshold]
                    - fp32[s]["goodput_over_offered"][threshold]
                    for s in seeds
                ]
                mean_d = statistics.mean(diffs)
                sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
                half = t_half(len(diffs), sd)
                rows.append(
                    {
                        "workload": workload,
                        "rate": rate,
                        "threshold_ms": int(threshold),
                        "per_seed_delta": {str(s): round(d, 4) for s, d in zip(seeds, diffs)},
                        "mean_delta_goodput": round(mean_d, 4),
                        "ci95": [round(mean_d - half, 4), round(mean_d + half, 4)],
                        "fp32_all_sustainable": all(
                            fp32[s]["goodput_over_offered"][threshold] >= 0.95 for s in seeds
                        ),
                        "bf16_all_sustainable": all(
                            bf16[s]["goodput_over_offered"][threshold] >= 0.95 for s in seeds
                        ),
                    }
                )
    return rows


def log_proof(raw_dir: Path) -> dict:
    proof = {}
    for allocation in ("int4", "int4_statebf16"):
        logs = list(raw_dir.glob(f"*/servers/{allocation}/*/server.log"))
        texts = [log.read_text(encoding="utf-8", errors="replace") for log in logs]
        proof[allocation] = {
            "session_count": len(texts),
            "has_int4": any("int4_per_token_head" in t for t in texts),
            "has_override_warning": any("Using the user-specified value" in t for t in texts),
            "has_piecewise": any("CUDAGraphMode.PIECEWISE" in t for t in texts),
        }
    return proof


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--attempt-ids", required=True)
    ap.add_argument("--out", default="results/verified/2026-08-09/statebf16-serving-formal-analysis.json")
    args = ap.parse_args()

    attempts = [
        audit_attempt(args.raw_dir.resolve(), attempt_id.strip())
        for attempt_id in args.attempt_ids.split(",")
        if attempt_id.strip()
    ]
    all_samples = [sample for attempt in attempts for sample in attempt["samples"]]
    result = {
        "schema_version": 1,
        "attempts": [
            {
                "attempt_id": a["attempt_id"],
                "git_commit": a["git_commit"],
                "vllm_source_commit": a["vllm_source_commit"],
                "config_sha256": a["config_sha256"],
                "num_samples": len(a["samples"]),
            }
            for a in attempts
        ],
        "boundaries": boundaries(all_samples),
        "paired_deltas": paired_deltas(all_samples),
        "log_proof": log_proof(args.raw_dir.resolve()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["boundaries"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
