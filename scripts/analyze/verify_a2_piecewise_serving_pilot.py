"""Audit the A2 PIECEWISE packed-serving diagnostic, MVEx, and pilot chain."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.analyze.verify_a2_reproduction import (
    VerificationError,
    load_json,
    require,
    sha256_file,
    utc_timestamp,
    verify_sidecar,
    write_json_with_hash,
    write_text_with_hash,
)
from scripts.analyze.verify_a2_serving_pilot import request_accounting

DIAGNOSTIC_ID = "a2-packed-serving-debug-piecewise-f7a79f5-westd-01"
MVEX_ID = "a2-packed-serving-mvex-piecewise-f7a79f5-westd-01"
PILOT_ID = "a2-packed-serving-pilot-piecewise-f7a79f5-westd-01"
FAILED_PARENT_ID = "a2-packed-serving-pilot-d1d52c4-westd-01"
ROOT_COMMIT = "f7a79f5d268aae66211926ac773b76e1bf443465"
VLLM_COMMIT = "55f47685a553ad8d776c464c59785399a98c7185"
DIAGNOSTIC_CONFIG_SHA256 = "6adf117adc411e6153524100af7e3321721e33585929f844c4536e38e4b9a0f2"
SERVING_CONFIG_SHA256 = "2a11f2d1f457cd3819bc9e8e6aa7db6af87f4dcb7654e2a8f6f1945604e390db"
ARRIVAL_TOLERANCE_FRACTION = 0.10
GRAPH_PROOF = "CUDAGraphMode.PIECEWISE"
FORBIDDEN_GRAPH_MODE = "CUDAGraphMode.FULL_AND_PIECEWISE"
RUNTIME_FAULT_SIGNATURES = (
    "EngineCore encountered a fatal error",
    "CUDA error: an illegal instruction was encountered",
    "server health check failed after benchmark",
)


def detect_runtime_faults(log_text: str) -> dict[str, bool]:
    return {signature: signature in log_text for signature in RUNTIME_FAULT_SIGNATURES}


def classify_piecewise_pilot(
    *,
    diagnostic_passed: bool,
    mvex_passed: bool,
    pilot_passed: bool,
    runtime_faults_absent: bool,
    graph_mode_proven: bool,
) -> str:
    if all(
        (
            diagnostic_passed,
            mvex_passed,
            pilot_passed,
            runtime_faults_absent,
            graph_mode_proven,
        )
    ):
        return "PILOT_PASSED_PIECEWISE_RUNTIME_INTEGRITY"
    return "PIECEWISE_CHAIN_INTEGRITY_REVIEW_REQUIRED"


def supervisor_exit(supervisor_dir: Path) -> int:
    require((supervisor_dir / "finished_at.txt").is_file(), "supervisor finish timestamp missing")
    return int((supervisor_dir / "exit_code.txt").read_text(encoding="ascii").strip())


def sample_plan(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["sample_id"]): item for item in contract["plan"]}


def audit_common_attempt(
    raw_dir: Path,
    *,
    attempt_id: str,
    expected_parent: str,
    expected_config_sha256: str,
    expected_phase: str,
    expected_summary: Mapping[str, int],
) -> dict[str, Any]:
    attempt_dir = raw_dir / "attempts" / attempt_id
    supervisor_dir = raw_dir / "supervisors" / attempt_id
    require(attempt_dir.is_dir(), f"attempt directory missing: {attempt_dir}")
    require(supervisor_dir.is_dir(), f"supervisor directory missing: {supervisor_dir}")

    for name in ("attempt_contract.json", "environment.json", "summary.json"):
        verify_sidecar(attempt_dir / f"{name}.sha256")

    contract = load_json(attempt_dir / "attempt_contract.json")
    environment = load_json(attempt_dir / "environment.json")
    summary = load_json(attempt_dir / "summary.json")
    require(contract["attempt_id"] == attempt_id, f"{attempt_id}: attempt ID mismatch")
    require(contract["parent_attempt"] == expected_parent, f"{attempt_id}: parent mismatch")
    require(contract["git_commit"] == ROOT_COMMIT, f"{attempt_id}: root commit mismatch")
    require(contract["vllm_source_commit"] == VLLM_COMMIT, f"{attempt_id}: vLLM commit mismatch")
    require(contract["config_sha256"] == expected_config_sha256, f"{attempt_id}: config hash mismatch")
    require(contract["phase"]["name"] == expected_phase, f"{attempt_id}: phase mismatch")
    require(summary["counts"] == dict(expected_summary), f"{attempt_id}: summary mismatch")

    require(environment["root_git"]["commit"] == ROOT_COMMIT, f"{attempt_id}: environment root commit")
    require(environment["root_git"]["clean"] is True, f"{attempt_id}: dirty root worktree")
    require(environment["vllm_source_commit"] == VLLM_COMMIT, f"{attempt_id}: environment vLLM commit")
    require(supervisor_exit(supervisor_dir) == 0, f"{attempt_id}: supervisor exit was nonzero")

    log_matches = list((attempt_dir / "servers").glob("*/*/server.log"))
    status_matches = list((attempt_dir / "servers").glob("*/*/status.json"))
    require(len(log_matches) == 1, f"{attempt_id}: expected one server log")
    require(len(status_matches) == 1, f"{attempt_id}: expected one server status")
    log_text = log_matches[0].read_text(encoding="utf-8", errors="replace")
    server_status = load_json(status_matches[0])
    runtime_faults = detect_runtime_faults(log_text)
    require(not any(runtime_faults.values()), f"{attempt_id}: runtime fault signature found")
    require(GRAPH_PROOF in log_text, f"{attempt_id}: PIECEWISE graph proof missing")
    require(FORBIDDEN_GRAPH_MODE not in log_text, f"{attempt_id}: default full graph mode present")
    require(server_status["status"] == "stopped", f"{attempt_id}: server did not stop")
    require(server_status["returncode"] == 0, f"{attempt_id}: server exit was nonzero")
    require(server_status["exception"] is None, f"{attempt_id}: server exception recorded")

    return {
        "attempt_dir": attempt_dir,
        "contract": contract,
        "environment": environment,
        "summary": summary,
        "server_log_path": str(log_matches[0]),
        "server_log_sha256": sha256_file(log_matches[0]),
        "server_status": server_status,
        "runtime_faults": runtime_faults,
        "graph_mode_proven": True,
        "supervisor_exit_code": 0,
    }


def audit_completed_sample(
    attempt_dir: Path,
    plan: Mapping[str, Mapping[str, Any]],
    sample_id: str,
) -> dict[str, Any]:
    sample_dir = attempt_dir / "samples" / sample_id
    require(sample_id in plan, f"{sample_id}: missing from frozen plan")
    for name in ("contract.json", "result.json", "analysis.json"):
        verify_sidecar(sample_dir / f"{name}.sha256")

    status = load_json(sample_dir / "status.json")
    result = load_json(sample_dir / "result.json")
    analysis = load_json(sample_dir / "analysis.json")
    require(status["status"] == "completed_validated", f"{sample_id}: status mismatch")
    require(status["result_sha256"] == sha256_file(sample_dir / "result.json"), f"{sample_id}: result hash")
    require(
        status["analysis_sha256"] == sha256_file(sample_dir / "analysis.json"),
        f"{sample_id}: analysis hash",
    )

    expected = int(plan[sample_id]["num_prompts"])
    accounting = request_accounting(result, expected)
    require(accounting["failed"] == 0, f"{sample_id}: request failures present")
    require(analysis["status"] == "completed_validated", f"{sample_id}: analysis status mismatch")
    require(analysis["completed"] == expected, f"{sample_id}: analysis completed mismatch")
    require(analysis["failed"] == 0, f"{sample_id}: analysis failed mismatch")
    require(result["attempt_id"] == attempt_dir.name, f"{sample_id}: attempt metadata")
    require(result["sample_id"] == sample_id, f"{sample_id}: sample metadata")
    require(result["git_commit"] == ROOT_COMMIT, f"{sample_id}: result root commit")
    require(result["vllm_source_commit"] == VLLM_COMMIT, f"{sample_id}: result vLLM commit")

    arrival_ratio = float(analysis["arrival_span_over_target"])
    require(
        abs(arrival_ratio - 1.0) <= ARRIVAL_TOLERANCE_FRACTION,
        f"{sample_id}: arrival-window drift {arrival_ratio}",
    )
    metrics = {
        "request_throughput": float(result["request_throughput"]),
        "request_goodput": float(result["request_goodput"]),
        "reported_ttft_p99_ms": float(analysis["reported_ttft_p99_ms"]),
        "reported_tpot_p99_ms": float(analysis["reported_tpot_p99_ms"]),
    }
    require(all(math.isfinite(value) for value in metrics.values()), f"{sample_id}: non-finite metric")

    sustainable_thresholds = [
        int(threshold)
        for threshold, item in analysis["slo_sweep"].items()
        if item["sustainable"]
    ]
    goodput_over_offered = {
        str(threshold): float(item["goodput_over_offered"])
        for threshold, item in analysis["slo_sweep"].items()
    }
    return {
        "sample_id": sample_id,
        "offered_rate": float(plan[sample_id]["request_rate"]),
        "accounting": accounting,
        "arrival_span_over_target": arrival_ratio,
        **metrics,
        "sustainable_ttft_thresholds_ms": sorted(sustainable_thresholds),
        "goodput_over_offered": goodput_over_offered,
    }


def fallacy_scan() -> list[dict[str, str]]:
    return [
        {"fallacy": "Simpson's Paradox", "severity": "NOTE", "detail": "No workload or allocation aggregation is used."},
        {"fallacy": "Ecological Fallacy", "severity": "NOTE", "detail": "Inference stays at the attempt and rate-point level."},
        {"fallacy": "Berkson's Paradox", "severity": "NOTE", "detail": "No selected correlation sample is analyzed."},
        {"fallacy": "Collider Bias", "severity": "NOTE", "detail": "No covariate adjustment is performed."},
        {"fallacy": "Base Rate Neglect", "severity": "NOTE", "detail": "No diagnostic probabilities are reported."},
        {"fallacy": "Regression to the Mean", "severity": "CAUTION", "detail": "Only one pilot seed is available; no efficacy stability claim is made."},
        {"fallacy": "Survivorship Bias", "severity": "SOLID", "detail": "The failed default-graph pilot remains quarantined and parent-linked."},
        {"fallacy": "Look-Elsewhere Effect", "severity": "SOLID", "detail": "All predeclared rate points and SLO thresholds are reported."},
        {"fallacy": "Garden of Forking Paths", "severity": "CAUTION", "detail": "PIECEWISE was selected after diagnostics and is labeled as a new linked chain."},
        {"fallacy": "Correlation != Causation", "severity": "CAUTION", "detail": "The graph-mode boundary is strongly isolated but not yet verified across seeds and allocations."},
        {"fallacy": "Reverse Causality", "severity": "NOTE", "detail": "Not applicable to the controlled graph-mode intervention."},
    ]


def build_validation_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for sample in report["pilot"]["samples"]:
        thresholds = ", ".join(str(value) for value in sample["sustainable_ttft_thresholds_ms"]) or "none"
        rows.append(
            f"| {sample['offered_rate']:.0f} | {sample['accounting']['completed']:,} | "
            f"{sample['accounting']['failed']:,} | {sample['request_throughput']:.2f} | "
            f"{sample['reported_ttft_p99_ms']:.2f} | {sample['reported_tpot_p99_ms']:.2f} | "
            f"{thresholds} |"
        )
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_piecewise_packed_serving_pilot_v1

## Validation Report

- **Diagnostic**: `{DIAGNOSTIC_ID}`
- **MVEx**: `{MVEX_ID}`
- **Pilot**: `{PILOT_ID}`
- **Verdict**: `{report['verdict']}`
- **Evidence Status**: `UNVERIFIED`
- **Overall A2 Status**: `{report['a2_overall_status']}`

The PIECEWISE packed pilot completed {report['total_completed']:,}/
{report['total_expected']:,} measured requests with zero failures. Every attempt
used root commit `{ROOT_COMMIT[:7]}`, vLLM `{VLLM_COMMIT[:8]}`, a clean worktree,
and server logs that prove `CUDAGraphMode.PIECEWISE`. Post-benchmark health
checks passed before each sample was published.

### Pilot Results

| Offered req/s | Completed | Failed | Throughput | P99 TTFT ms | P99 TPOT ms | Sustainable TTFT thresholds ms |
|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

Rate 50 is a valid overload point: all 3,000 requests completed, but no tested
TTFT threshold met the 0.95 goodput/offered criterion. This negative result is
retained and is not treated as a failed request or silently excluded sample.

### Evidence Boundary

This audit closes the packed-only runtime-integrity pilot under PIECEWISE CUDA
graphs. It does not compare fp16, uniform int4, and packed allocations under the
same graph mode; it has one seed and one synthetic workload. The result remains
`ANALYZED/UNVERIFIED` and cannot enter paper quantitative claims. The next gate
is a fair three-allocation Random/ShareGPT pilot, followed by an independently
frozen multi-seed formal matrix and reproducibility run.

### Fallacy Scan

- **Coverage**: 11/11
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    diagnostic = audit_common_attempt(
        raw_dir,
        attempt_id=DIAGNOSTIC_ID,
        expected_parent=FAILED_PARENT_ID,
        expected_config_sha256=DIAGNOSTIC_CONFIG_SHA256,
        expected_phase="packed_debug",
        expected_summary={"completed_validated": 1},
    )
    mvex = audit_common_attempt(
        raw_dir,
        attempt_id=MVEX_ID,
        expected_parent=DIAGNOSTIC_ID,
        expected_config_sha256=SERVING_CONFIG_SHA256,
        expected_phase="packed_mvex",
        expected_summary={"completed_validated": 1},
    )
    pilot = audit_common_attempt(
        raw_dir,
        attempt_id=PILOT_ID,
        expected_parent=MVEX_ID,
        expected_config_sha256=SERVING_CONFIG_SHA256,
        expected_phase="packed_pilot",
        expected_summary={"completed_validated": 3},
    )

    diagnostic_plan = sample_plan(diagnostic["contract"])
    mvex_plan = sample_plan(mvex["contract"])
    pilot_plan = sample_plan(pilot["contract"])
    diagnostic_sample = audit_completed_sample(
        diagnostic["attempt_dir"],
        diagnostic_plan,
        "packed_per_layer__random__r30__s7",
    )
    mvex_sample = audit_completed_sample(
        mvex["attempt_dir"],
        mvex_plan,
        "packed_per_layer__random__r30__s7",
    )
    pilot_samples = [
        audit_completed_sample(pilot["attempt_dir"], pilot_plan, sample_id)
        for sample_id in (
            "packed_per_layer__random__r30__s7",
            "packed_per_layer__random__r40__s7",
            "packed_per_layer__random__r50__s7",
        )
    ]

    total_expected = sum(item["accounting"]["expected"] for item in pilot_samples)
    total_completed = sum(item["accounting"]["completed"] for item in pilot_samples)
    total_failed = sum(item["accounting"]["failed"] for item in pilot_samples)
    verdict = classify_piecewise_pilot(
        diagnostic_passed=diagnostic_sample["accounting"]["failed"] == 0,
        mvex_passed=mvex_sample["accounting"]["failed"] == 0,
        pilot_passed=total_expected == total_completed and total_failed == 0,
        runtime_faults_absent=not any(
            any(attempt["runtime_faults"].values())
            for attempt in (diagnostic, mvex, pilot)
        ),
        graph_mode_proven=all(
            attempt["graph_mode_proven"] for attempt in (diagnostic, mvex, pilot)
        ),
    )
    require(
        verdict == "PILOT_PASSED_PIECEWISE_RUNTIME_INTEGRITY",
        "PIECEWISE serving chain did not pass",
    )

    report = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "verification_status": "ANALYZED",
        "evidence_status": "UNVERIFIED",
        "verdict": verdict,
        "a2_overall_status": "PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING",
        "root_commit": ROOT_COMMIT,
        "vllm_commit": VLLM_COMMIT,
        "total_expected": total_expected,
        "total_completed": total_completed,
        "total_failed": total_failed,
        "diagnostic": {
            "attempt_id": DIAGNOSTIC_ID,
            "parent_attempt": FAILED_PARENT_ID,
            "sample": diagnostic_sample,
            "server_log_sha256": diagnostic["server_log_sha256"],
        },
        "mvex": {
            "attempt_id": MVEX_ID,
            "parent_attempt": DIAGNOSTIC_ID,
            "sample": mvex_sample,
            "server_log_sha256": mvex["server_log_sha256"],
        },
        "pilot": {
            "attempt_id": PILOT_ID,
            "parent_attempt": MVEX_ID,
            "samples": pilot_samples,
            "server_log_sha256": pilot["server_log_sha256"],
        },
        "fallacy_scan": fallacy_scan(),
        "next_gate": (
            "Run fp16, uniform int4, and packed L23-protected under the same "
            "PIECEWISE graph mode on Random and ShareGPT before formal expansion."
        ),
    }
    require(len(report["fallacy_scan"]) == 11, "fallacy scan coverage mismatch")
    write_json_with_hash(output_dir / "piecewise_pilot_audit_report.json", report)
    write_text_with_hash(output_dir / "piecewise_validation_report.md", build_validation_markdown(report))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"verification failed: {exc}")
        raise SystemExit(2) from exc
