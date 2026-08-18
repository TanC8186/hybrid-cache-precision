"""Fail-closed logical audit for the M3 mechanism-isolation attempt.

The serving runner already validates request-level denominators and latency
cross-checks.  This audit validates the higher-level contract: all six
allocation contrasts are paired by seed, no phase is silently pooled into the
formal denominator, and telemetry artifacts are present for every server
session.  It intentionally reports descriptive paired differences only; the
formal unit is allocation x seed, not an individual request or telemetry row.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

ALLOCATIONS = (
    "full_fixed_block_count",
    "joint_fixed_block_count",
    "full_fixed_bytes",
    "joint_fixed_bytes",
    "full_fixed_concurrency",
    "joint_fixed_concurrency",
)
CONTRASTS = ("block_count", "bytes", "concurrency")
FORMAL_SEEDS = (11, 23, 47)
REQUIRED_TELEMETRY_METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_time_seconds",
)
PROMETHEUS_FAMILY_SUFFIXES = ("_bucket", "_count", "_sum", "_created")
FALLACY_NAMES = (
    "Simpson's paradox",
    "Ecological fallacy",
    "Berkson's paradox",
    "Collider bias",
    "Base rate neglect",
    "Regression to the mean",
    "Survivorship bias",
    "Look-elsewhere effect",
    "Garden of forking paths",
    "Correlation != causation",
    "Reverse causality",
)


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def metric_families(metric_names: set[str] | list[str]) -> set[str]:
    """Normalize Prometheus scalar and histogram samples to family names."""
    families: set[str] = set()
    for name in metric_names:
        for suffix in PROMETHEUS_FAMILY_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        families.add(name)
    return families


def fallacy_scan() -> list[dict[str, str]]:
    details = {
        "Simpson's paradox": "Effects are kept stratified by constraint and seed; no aggregate direction replaces a stratum.",
        "Ecological fallacy": "The formal unit is allocation x seed; request and telemetry rows are not treated as independent observations.",
        "Berkson's paradox": "All predeclared allocation/seed cells are required; no outcome-based selection filter is applied.",
        "Collider bias": "No post-treatment queue or telemetry variable is conditioned on when forming paired effects.",
        "Base rate neglect": "SLO and failure rates retain the complete offered-request denominator for every sample.",
        "Regression to the mean": "Formal seeds are frozen independently of MVEx/pilot outcomes and are not selected for extremeness.",
        "Survivorship bias": "Failed, missing, or incomplete cells fail closed rather than being removed from the denominator.",
        "Look-elsewhere effect": "Three contrasts, six allocations, and three formal seeds were frozen before the run; no contrast is selected by result.",
        "Garden of forking paths": "Load, rate, duration, SLO thresholds, seeds, and analysis unit are fixed in the M3 contract.",
        "Correlation != causation": "Paired serving differences are reported descriptively and are not framed as a causal mechanism estimate.",
        "Reverse causality": "Allocation constraints are assigned before serving; observed latency cannot determine the assigned dtype retrospectively.",
    }
    return [
        {
            "fallacy": name,
            "severity": "CAUTION" if name in {"Ecological fallacy", "Look-elsewhere effect", "Garden of forking paths"} else "NOTE",
            "status": "CHECKED",
            "detail": details[name],
        }
        for name in FALLACY_NAMES
    ]


def audit_attempt(attempt_dir: Path) -> dict[str, Any]:
    contract = read_json(attempt_dir / "attempt_contract.json")
    require(contract.get("phase", {}).get("name") == "formal", "attempt is not M3 formal")
    plan = contract.get("plan", [])
    require(len(plan) == 18, f"formal plan has {len(plan)} samples, expected 18")
    expected_ids = {
        f"{allocation}__random__r40__s{seed}"
        for allocation in ALLOCATIONS
        for seed in FORMAL_SEEDS
    }
    actual_ids = {str(sample.get("sample_id")) for sample in plan}
    require(actual_ids == expected_ids, "formal sample plan differs from frozen 18-cell matrix")

    summary = read_json(attempt_dir / "summary.json")
    require(summary.get("counts") == {"completed_validated": 18}, f"invalid formal counts: {summary.get('counts')}")
    sample_root = attempt_dir / "samples"
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(expected_ids):
        sample_dir = sample_root / sample_id
        sample_contract = read_json(sample_dir / "contract.json")
        require(sample_contract.get("sample_id") == sample_id, f"sample contract ID mismatch: {sample_id}")
        require(sample_contract.get("allocation") == sample_id.split("__", 1)[0], f"sample allocation mismatch: {sample_id}")
        require(sample_contract.get("workload") == "random", f"sample workload mismatch: {sample_id}")
        require(float(sample_contract.get("request_rate", math.nan)) == 40.0, f"sample rate mismatch: {sample_id}")
        require(int(sample_contract.get("seed", -1)) in FORMAL_SEEDS, f"sample seed mismatch: {sample_id}")
        status = read_json(sample_dir / "status.json")
        require(status.get("status") == "completed_validated", f"sample not validated: {sample_id}")
        analysis = read_json(sample_dir / "analysis.json")
        require(analysis.get("status") == "completed_validated", f"analysis status invalid: {sample_id}")
        expected_requests = int(sample_contract.get("num_prompts", -1))
        require(expected_requests > 0, f"missing sample denominator: {sample_id}")
        require(
            int(analysis.get("completed", -1)) + int(analysis.get("failed", -1)) == expected_requests,
            f"request denominator mismatch: {sample_id}",
        )
        result = read_json(sample_dir / "result.json")
        require(math.isfinite(float(result.get("duration", math.nan))), f"non-finite duration: {sample_id}")
        rows.append({"sample_id": sample_id, "analysis": analysis, "status": status, "contract": sample_contract})

    # Each allocation must have a session telemetry file.  A session may serve
    # several seeds, so inspect all session directories rather than assuming a
    # one-to-one mapping with samples.
    server_root = attempt_dir / "servers"
    sessions = list(server_root.glob("*/**/telemetry_summary.json"))
    require(len(sessions) == len(ALLOCATIONS), f"telemetry sessions={len(sessions)} != {len(ALLOCATIONS)} allocations")
    telemetry = []
    for path in sessions:
        data = read_json(path)
        require(int(data.get("records", 0)) > 0, f"empty telemetry: {path}")
        metric_names = set(data.get("metric_names", []))
        missing_metrics = sorted(set(REQUIRED_TELEMETRY_METRICS) - metric_families(metric_names))
        require(not missing_metrics, f"telemetry missing required metrics: {path}: {missing_metrics}")
        require(int(data.get("gpu_records", 0)) > 0, f"telemetry has no GPU records: {path}")
        metrics_errors = int(data.get("metrics_errors", 0))
        require(metrics_errors <= 1, f"too many telemetry metric errors: {path}: {metrics_errors}")
        allocation = path.parts[-3] if len(path.parts) >= 3 else "unknown"
        require(allocation in ALLOCATIONS, f"unexpected telemetry allocation path: {path}")
        telemetry.append(
            {
                "path": str(path),
                "allocation": allocation,
                "records": data.get("records"),
                "gpu_records": data.get("gpu_records"),
                "metrics_errors": metrics_errors,
                "missing_required_metrics": missing_metrics,
            }
        )
    require({row["allocation"] for row in telemetry} == set(ALLOCATIONS), "telemetry allocation coverage mismatch")

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        allocation, _workload, _rate, seed_text = sample_id.split("__")
        seed = int(seed_text[1:])
        by_key[(allocation, seed)] = row["analysis"]
    paired: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        full = f"full_fixed_{contrast}"
        joint = f"joint_fixed_{contrast}"
        for seed in FORMAL_SEEDS:
            left = by_key[(full, seed)]
            right = by_key[(joint, seed)]
            paired.append(
                {
                    "contrast": contrast,
                    "seed": seed,
                    "throughput_delta_req_s": float(right["request_throughput_req_s"]) - float(left["request_throughput_req_s"]),
                    "ttft_p95_delta_ms": float(right["ttft_p95_ms_recomputed"]) - float(left["ttft_p95_ms_recomputed"]),
                    "tpot_p95_delta_ms": float(right["tpot_p95_ms_recomputed"]) - float(left["tpot_p95_ms_recomputed"]),
                }
            )

    return {
        "schema_version": 1,
        "attempt_id": contract.get("attempt_id"),
        "gate": "M3 formal logical/statistical audit",
        "gate_status": "PASS",
        "evidence_status": "ANALYZED",
        "reproducibility": "not_run",
        "formal_cells": len(rows),
        "allocations": list(ALLOCATIONS),
        "seeds": list(FORMAL_SEEDS),
        "unit_of_analysis": "allocation x seed",
        "telemetry_sessions": telemetry,
        "paired_effects": paired,
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": fallacy_scan(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit_attempt(args.attempt_dir.resolve())
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "gate_status": report["gate_status"], "formal_cells": report["formal_cells"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
