"""Audit the A2 three-allocation PIECEWISE serving MVEx and failed pilot."""

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

MVEX_ID = "a2-comparative-serving-mvex-piecewise-f7a79f5-westd-01"
SUPERSEDED_PILOT_ID = "a2-comparative-serving-pilot-piecewise-f7a79f5-westd-01"
PILOT_ID = "a2-comparative-serving-pilot-piecewise-f7a79f5-westd-02"
PARENT_ID = "a2-packed-serving-pilot-piecewise-f7a79f5-westd-01"
ROOT_COMMIT = "f7a79f5d268aae66211926ac773b76e1bf443465"
VLLM_COMMIT = "55f47685a553ad8d776c464c59785399a98c7185"
MVEX_CONFIG = "a2_comparative_piecewise_serving_f7a79f5.yaml"
PILOT_CONFIG = "a2_comparative_piecewise_serving_f7a79f5_v2.yaml"
MVEX_CONFIG_SHA256 = "35e092f0db613c13e8867d1564cce0073b675fb5c475f1362d36f9711a190d75"
PILOT_CONFIG_SHA256 = "3c821db7f155a072bdd4a545c3f4f4de2a8ea6f5a087c36c4a515e54ac84e792"
ARRIVAL_TOLERANCE_FRACTION = 0.10
GRAPH_PROOF = "CUDAGraphMode.PIECEWISE"
FORBIDDEN_GRAPH_MODE = "CUDAGraphMode.FULL_AND_PIECEWISE"
FD_LIMIT_SIGNATURE = "Too many open files"
RUNTIME_FAULT_SIGNATURES = (
    "EngineCore encountered a fatal error",
    "CUDA error: an illegal instruction was encountered",
    "server health check failed after benchmark",
)
ALLOCATIONS = ("fp16", "int4", "packed_per_layer")
TTFT_THRESHOLDS_MS = (250, 500, 1000, 2000, 3000)


def detect_runtime_faults(log_text: str) -> dict[str, bool]:
    return {signature: signature in log_text for signature in RUNTIME_FAULT_SIGNATURES}


def classify_comparative_pilot(
    *,
    mvex_passed: bool,
    pilot_requests_passed: bool,
    servers_clean: bool,
    graph_mode_proven: bool,
    sharegpt_bracketed: bool,
    fd_limit_failure_proven: bool,
) -> str:
    if all(
        (
            mvex_passed,
            pilot_requests_passed,
            servers_clean,
            graph_mode_proven,
            sharegpt_bracketed,
        )
    ):
        return "PILOT_PASSED_PREDECLARED_CRITERIA"
    if (
        mvex_passed
        and not pilot_requests_passed
        and servers_clean
        and graph_mode_proven
        and not sharegpt_bracketed
        and fd_limit_failure_proven
    ):
        return "PILOT_FAILED_CLIENT_FD_LIMIT_AND_SHAREGPT_WINDOW_BRACKETING"
    return "COMPARATIVE_PILOT_INTEGRITY_REVIEW_REQUIRED"


def sharegpt_bracket_report(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, bool]]], bool]:
    report: dict[str, dict[str, dict[str, bool]]] = {}
    all_bracketed = True
    for allocation in ALLOCATIONS:
        allocation_samples = [
            sample
            for sample in samples
            if sample["allocation"] == allocation and sample["workload"] == "sharegpt"
        ]
        threshold_report: dict[str, dict[str, bool]] = {}
        for threshold in TTFT_THRESHOLDS_MS:
            sustainable = [
                threshold in sample["sustainable_ttft_thresholds_ms"]
                for sample in allocation_samples
            ]
            threshold_report[str(threshold)] = {
                "has_sustainable": any(sustainable),
                "has_unsustainable": bool(sustainable) and not all(sustainable),
            }
        report[allocation] = threshold_report
        allocation_bracketed = any(
            state["has_sustainable"] and state["has_unsustainable"]
            for state in threshold_report.values()
        )
        all_bracketed = all_bracketed and allocation_bracketed
    return report, all_bracketed


def verify_all_sidecars(attempt_dir: Path, expected_count: int) -> int:
    sidecars = sorted(attempt_dir.rglob("*.sha256"))
    require(
        len(sidecars) == expected_count,
        f"{attempt_dir.name}: expected {expected_count} sidecars, found {len(sidecars)}",
    )
    for sidecar in sidecars:
        verify_sidecar(sidecar)
    return len(sidecars)


def supervisor_exit(supervisor_dir: Path) -> int:
    require((supervisor_dir / "finished_at.txt").is_file(), "supervisor finish timestamp missing")
    return int((supervisor_dir / "exit_code.txt").read_text(encoding="ascii").strip())


def sample_plan(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["sample_id"]): item for item in contract["plan"]}


def audit_servers(attempt_dir: Path) -> list[dict[str, Any]]:
    status_paths = sorted((attempt_dir / "servers").glob("*/*/status.json"))
    require(len(status_paths) == len(ALLOCATIONS), f"{attempt_dir.name}: server count mismatch")
    servers: list[dict[str, Any]] = []
    for status_path in status_paths:
        allocation = status_path.parents[1].name
        require(allocation in ALLOCATIONS, f"{attempt_dir.name}: unknown allocation {allocation}")
        log_path = status_path.with_name("server.log")
        require(log_path.is_file(), f"{attempt_dir.name}/{allocation}: server log missing")
        status = load_json(status_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        runtime_faults = detect_runtime_faults(log_text)
        require(status["status"] == "stopped", f"{attempt_dir.name}/{allocation}: server status")
        require(status["returncode"] == 0, f"{attempt_dir.name}/{allocation}: server return code")
        require(status["exception"] is None, f"{attempt_dir.name}/{allocation}: server exception")
        require(not any(runtime_faults.values()), f"{attempt_dir.name}/{allocation}: runtime fault")
        require(GRAPH_PROOF in log_text, f"{attempt_dir.name}/{allocation}: graph proof missing")
        require(
            FORBIDDEN_GRAPH_MODE not in log_text,
            f"{attempt_dir.name}/{allocation}: forbidden full graph mode present",
        )
        servers.append(
            {
                "allocation": allocation,
                "status": status["status"],
                "returncode": status["returncode"],
                "exception": status["exception"],
                "runtime_faults": runtime_faults,
                "graph_mode_proven": True,
                "server_log_sha256": sha256_file(log_path),
            }
        )
    return servers


def audit_common_attempt(
    raw_dir: Path,
    *,
    attempt_id: str,
    expected_parent: str,
    expected_phase: str,
    expected_config_name: str,
    expected_config_sha256: str,
    expected_summary_count: int,
    expected_sidecars: int,
) -> dict[str, Any]:
    attempt_dir = raw_dir / "attempts" / attempt_id
    supervisor_dir = raw_dir / "supervisors" / attempt_id
    require(attempt_dir.is_dir(), f"attempt directory missing: {attempt_dir}")
    require(supervisor_dir.is_dir(), f"supervisor directory missing: {supervisor_dir}")
    sidecars_verified = verify_all_sidecars(attempt_dir, expected_sidecars)

    contract = load_json(attempt_dir / "attempt_contract.json")
    environment = load_json(attempt_dir / "environment.json")
    summary = load_json(attempt_dir / "summary.json")
    config_path = raw_dir / "experiments" / "configs" / expected_config_name
    require(config_path.is_file(), f"config missing: {config_path}")
    require(sha256_file(config_path) == expected_config_sha256, f"{attempt_id}: local config hash")
    require(contract["attempt_id"] == attempt_id, f"{attempt_id}: attempt ID mismatch")
    require(contract["parent_attempt"] == expected_parent, f"{attempt_id}: parent mismatch")
    require(contract["git_commit"] == ROOT_COMMIT, f"{attempt_id}: root commit mismatch")
    require(contract["vllm_source_commit"] == VLLM_COMMIT, f"{attempt_id}: vLLM commit mismatch")
    require(contract["config_sha256"] == expected_config_sha256, f"{attempt_id}: config hash mismatch")
    require(contract["phase"]["name"] == expected_phase, f"{attempt_id}: phase mismatch")
    require(
        summary["counts"] == {"completed_validated": expected_summary_count},
        f"{attempt_id}: summary mismatch",
    )
    require(environment["root_git"]["commit"] == ROOT_COMMIT, f"{attempt_id}: environment commit")
    require(environment["root_git"]["clean"] is True, f"{attempt_id}: dirty root worktree")
    require(environment["root_git"]["status"] == "", f"{attempt_id}: root status not empty")
    require(environment["vllm_source_commit"] == VLLM_COMMIT, f"{attempt_id}: environment vLLM")
    require(supervisor_exit(supervisor_dir) == 0, f"{attempt_id}: supervisor exit")

    return {
        "attempt_dir": attempt_dir,
        "contract": contract,
        "environment": environment,
        "summary": summary,
        "sidecars_verified": sidecars_verified,
        "servers": audit_servers(attempt_dir),
        "supervisor_exit_code": 0,
    }


def output_length_summary(result: Mapping[str, Any], success_mask: Sequence[bool]) -> dict[str, float | int]:
    lengths = sorted(
        int(length)
        for length, success in zip(result["output_lens"], success_mask)
        if success
    )
    require(bool(lengths), "cannot summarize empty output lengths")

    def pick(fraction: float) -> int:
        return lengths[round((len(lengths) - 1) * fraction)]

    return {
        "min": lengths[0],
        "median": pick(0.50),
        "p99": pick(0.99),
        "max": lengths[-1],
        "mean": sum(lengths) / len(lengths),
    }


def audit_sample(
    attempt_dir: Path,
    plan: Mapping[str, Mapping[str, Any]],
    sample_id: str,
) -> dict[str, Any]:
    sample_dir = attempt_dir / "samples" / sample_id
    require(sample_id in plan, f"{sample_id}: missing from frozen plan")
    for name in ("contract.json", "result.json", "analysis.json"):
        verify_sidecar(sample_dir / f"{name}.sha256")

    contract = load_json(sample_dir / "contract.json")
    status = load_json(sample_dir / "status.json")
    result = load_json(sample_dir / "result.json")
    analysis = load_json(sample_dir / "analysis.json")
    require(status["status"] == "completed_validated", f"{sample_id}: status mismatch")
    require(status["result_sha256"] == sha256_file(sample_dir / "result.json"), f"{sample_id}: result hash")
    require(
        status["analysis_sha256"] == sha256_file(sample_dir / "analysis.json"),
        f"{sample_id}: analysis hash",
    )
    accounting = request_accounting(result, int(plan[sample_id]["num_prompts"]))
    require(analysis["status"] == "completed_validated", f"{sample_id}: analysis status")
    require(analysis["completed"] == accounting["completed"], f"{sample_id}: analysis completed")
    require(analysis["failed"] == accounting["failed"], f"{sample_id}: analysis failed")
    require(contract["sample_id"] == sample_id, f"{sample_id}: contract sample ID")
    require(result["attempt_id"] == attempt_dir.name, f"{sample_id}: result attempt ID")
    require(result["sample_id"] == sample_id, f"{sample_id}: result sample ID")
    require(result["git_commit"] == ROOT_COMMIT, f"{sample_id}: root commit")
    require(result["vllm_source_commit"] == VLLM_COMMIT, f"{sample_id}: vLLM commit")

    arrival_ratio = float(analysis["arrival_span_over_target"])
    require(
        abs(arrival_ratio - 1.0) <= ARRIVAL_TOLERANCE_FRACTION,
        f"{sample_id}: arrival-window drift {arrival_ratio}",
    )
    metrics = {
        "request_throughput": float(result["request_throughput"]),
        "reported_ttft_p99_ms": float(analysis["reported_ttft_p99_ms"]),
        "reported_tpot_p99_ms": float(analysis["reported_tpot_p99_ms"]),
        "benchmark_duration_s": float(analysis["benchmark_duration_s"]),
        "drain_after_arrival_window_s": float(analysis["drain_after_arrival_window_s"]),
    }
    require(all(math.isfinite(value) for value in metrics.values()), f"{sample_id}: non-finite metric")

    errors = list(result["errors"])
    success_mask = [not error for error in errors]
    fd_limit_failures = sum(FD_LIMIT_SIGNATURE in str(error) for error in errors if error)
    sustainable_thresholds = sorted(
        int(threshold)
        for threshold, item in analysis["slo_sweep"].items()
        if item["sustainable"]
    )
    return {
        "sample_id": sample_id,
        "allocation": str(contract["allocation"]),
        "workload": str(contract["workload"]),
        "seed": int(contract["seed"]),
        "offered_rate": float(contract["request_rate"]),
        "accounting": accounting,
        "arrival_span_over_target": arrival_ratio,
        **metrics,
        "request_throughput_over_offered": float(analysis["request_throughput_over_offered"]),
        "sustainable_ttft_thresholds_ms": sustainable_thresholds,
        "goodput_over_offered": {
            str(threshold): float(item["goodput_over_offered"])
            for threshold, item in analysis["slo_sweep"].items()
        },
        "fd_limit_failures": fd_limit_failures,
        "output_lengths": output_length_summary(result, success_mask),
    }


def audit_supersession(raw_dir: Path) -> dict[str, Any]:
    path = raw_dir / "contracts" / f"{SUPERSEDED_PILOT_ID}.superseded.json"
    record = load_json(path)
    require(record["attempt_id"] == SUPERSEDED_PILOT_ID, "superseded attempt ID mismatch")
    require(record["status"] == "SUPERSEDED_PRECOMPUTE", "superseded status mismatch")
    require(record["launched"] is False, "superseded plan was unexpectedly launched")
    require(record["scientific_attempt_created"] is False, "superseded plan created an attempt")
    require(record["replacement_attempt"] == PILOT_ID, "superseded replacement mismatch")
    require(
        not (raw_dir / "attempts" / SUPERSEDED_PILOT_ID).exists(),
        "superseded precompute plan unexpectedly has an attempt directory",
    )
    return record


def fallacy_scan() -> list[dict[str, str]]:
    return [
        {
            "fallacy": "Simpson's Paradox",
            "severity": "CAUTION",
            "detail": "Random and ShareGPT saturation behavior differs; workloads are not pooled.",
        },
        {
            "fallacy": "Ecological Fallacy",
            "severity": "NOTE",
            "detail": "Inference stays at the allocation-workload-rate attempt level.",
        },
        {
            "fallacy": "Berkson's Paradox",
            "severity": "NOTE",
            "detail": "No outcome-conditioned correlation sample is analyzed.",
        },
        {
            "fallacy": "Collider Bias",
            "severity": "NOTE",
            "detail": "No post-treatment covariate adjustment is performed.",
        },
        {
            "fallacy": "Base Rate Neglect",
            "severity": "NOTE",
            "detail": "No diagnostic probability is reported.",
        },
        {
            "fallacy": "Regression to the Mean",
            "severity": "CAUTION",
            "detail": "The pilot has one seed and cannot establish boundary stability.",
        },
        {
            "fallacy": "Survivorship Bias",
            "severity": "SOLID",
            "detail": "All 35,100 issued requests remain in the denominator, including eight client failures.",
        },
        {
            "fallacy": "Look-Elsewhere Effect",
            "severity": "SOLID",
            "detail": "All frozen rates and all five TTFT thresholds are reported.",
        },
        {
            "fallacy": "Garden of Forking Paths",
            "severity": "CAUTION",
            "detail": "The replacement rate grid is parent-linked; any protocol correction must use another new attempt.",
        },
        {
            "fallacy": "Correlation != Causation",
            "severity": "CAUTION",
            "detail": "The FD error is directly observed, but the short-window drain diagnosis still requires a new MVEx.",
        },
        {
            "fallacy": "Reverse Causality",
            "severity": "NOTE",
            "detail": "Not applicable to the controlled serving configuration comparison.",
        },
    ]


def build_validation_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for sample in report["pilot"]["samples"]:
        thresholds = ", ".join(str(value) for value in sample["sustainable_ttft_thresholds_ms"]) or "none"
        rows.append(
            f"| {sample['allocation']} | {sample['workload']} | {sample['offered_rate']:.0f} | "
            f"{sample['accounting']['completed']:,} | {sample['accounting']['failed']:,} | "
            f"{sample['reported_ttft_p99_ms']:.2f} | {sample['reported_tpot_p99_ms']:.2f} | "
            f"{sample['drain_after_arrival_window_s']:.2f} | {thresholds} |"
        )
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_comparative_piecewise_serving_pilot_v1

## Validation Report

- **MVEx**: `{MVEX_ID}`
- **Pilot**: `{PILOT_ID}`
- **Pilot Verdict**: `{report['pilot']['verdict']}`
- **MVEx Evidence Status**: `UNVERIFIED`
- **Pilot Evidence Status**: `QUARANTINED`
- **Overall A2 Status**: `{report['a2_overall_status']}`

### Integrity

The comparative MVEx completed {report['mvex']['total_completed']:,}/
{report['mvex']['total_expected']:,} requests with zero failures. The pilot
published all 21 planned samples, but request-level accounting is
{report['pilot']['total_completed']:,} completed plus
{report['pilot']['total_failed']:,} failed out of
{report['pilot']['total_expected']:,}. No request is silently excluded.

All six server sessions exited with return code zero, recorded no exception or
runtime-fatal signature, and prove `CUDAGraphMode.PIECEWISE`. The eight pilot
failures are confined to `fp16__random__r50__s7` and contain the client-side
`Too many open files` signature. The remote soft `nofile` limit was 1024.

### Pilot Results

| Allocation | Workload | Offered req/s | Completed | Failed | P99 TTFT ms | P99 TPOT ms | Drain s | Sustainable TTFT thresholds ms |
|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

### Gate 2 Failure

The frozen success criteria required 35,100 successful requests, zero failures,
and at least one sustainable plus one unsustainable ShareGPT point for every
allocation under at least one tested TTFT threshold. Neither condition holds.
ShareGPT at 10 req/s has low P99 latency but drains for 7.88--11.07 seconds
after the 60-second arrival window; because goodput uses total benchmark
duration, goodput/offered is only 0.844--0.884. Lowering the offered rate does
not remove this fixed-tail bias.

### Evidence Boundary

The pilot is useful as protocol and environment diagnostics, but it is
`ANALYZED/QUARANTINED` and contributes zero rows to formal efficacy evidence.
Formal expansion is blocked. The next linked MVEx must raise the inherited soft
file-descriptor limit and test a longer ShareGPT measurement window while
preserving the real completion-length distribution. A new pilot may start only
after that MVEx demonstrates both zero failures and a sustainable/unsustainable
ShareGPT bracket for all three allocations.

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

    supersession = audit_supersession(raw_dir)
    mvex = audit_common_attempt(
        raw_dir,
        attempt_id=MVEX_ID,
        expected_parent=PARENT_ID,
        expected_phase="comparative_mvex",
        expected_config_name=MVEX_CONFIG,
        expected_config_sha256=MVEX_CONFIG_SHA256,
        expected_summary_count=6,
        expected_sidecars=24,
    )
    pilot = audit_common_attempt(
        raw_dir,
        attempt_id=PILOT_ID,
        expected_parent=MVEX_ID,
        expected_phase="comparative_pilot_v2",
        expected_config_name=PILOT_CONFIG,
        expected_config_sha256=PILOT_CONFIG_SHA256,
        expected_summary_count=21,
        expected_sidecars=69,
    )

    mvex_plan = sample_plan(mvex["contract"])
    pilot_plan = sample_plan(pilot["contract"])
    mvex_samples = [
        audit_sample(mvex["attempt_dir"], mvex_plan, sample_id)
        for sample_id in sorted(mvex_plan)
    ]
    pilot_samples = [
        audit_sample(pilot["attempt_dir"], pilot_plan, sample_id)
        for sample_id in sorted(pilot_plan)
    ]

    mvex_expected = sum(sample["accounting"]["expected"] for sample in mvex_samples)
    mvex_completed = sum(sample["accounting"]["completed"] for sample in mvex_samples)
    mvex_failed = sum(sample["accounting"]["failed"] for sample in mvex_samples)
    pilot_expected = sum(sample["accounting"]["expected"] for sample in pilot_samples)
    pilot_completed = sum(sample["accounting"]["completed"] for sample in pilot_samples)
    pilot_failed = sum(sample["accounting"]["failed"] for sample in pilot_samples)
    require((mvex_expected, mvex_completed, mvex_failed) == (9000, 9000, 0), "MVEx denominator mismatch")
    require(pilot_expected == 35100, "pilot expected denominator mismatch")
    require((pilot_completed, pilot_failed) == (35092, 8), "pilot observed denominator drift")

    failed_samples = [
        sample for sample in pilot_samples if sample["accounting"]["failed"] > 0
    ]
    require(len(failed_samples) == 1, "pilot failure is not confined to one sample")
    failed_sample = failed_samples[0]
    require(failed_sample["sample_id"] == "fp16__random__r50__s7", "unexpected failed sample")
    require(failed_sample["fd_limit_failures"] == 8, "FD failure signature count mismatch")

    bracket_report, sharegpt_bracketed = sharegpt_bracket_report(pilot_samples)
    require(not sharegpt_bracketed, "pilot unexpectedly brackets ShareGPT")
    servers_clean = all(
        not any(server["runtime_faults"].values())
        and server["returncode"] == 0
        and server["exception"] is None
        for attempt in (mvex, pilot)
        for server in attempt["servers"]
    )
    graph_mode_proven = all(
        server["graph_mode_proven"]
        for attempt in (mvex, pilot)
        for server in attempt["servers"]
    )
    verdict = classify_comparative_pilot(
        mvex_passed=mvex_completed == mvex_expected and mvex_failed == 0,
        pilot_requests_passed=pilot_completed == pilot_expected and pilot_failed == 0,
        servers_clean=servers_clean,
        graph_mode_proven=graph_mode_proven,
        sharegpt_bracketed=sharegpt_bracketed,
        fd_limit_failure_proven=failed_sample["fd_limit_failures"] == pilot_failed,
    )
    require(
        verdict == "PILOT_FAILED_CLIENT_FD_LIMIT_AND_SHAREGPT_WINDOW_BRACKETING",
        "comparative pilot failure classification did not close",
    )

    report = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "verification_status": "ANALYZED",
        "a2_overall_status": "PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING",
        "root_commit": ROOT_COMMIT,
        "vllm_commit": VLLM_COMMIT,
        "superseded_precompute": supersession,
        "mvex": {
            "attempt_id": MVEX_ID,
            "parent_attempt": PARENT_ID,
            "evidence_status": "UNVERIFIED",
            "sidecars_verified": mvex["sidecars_verified"],
            "supervisor_exit_code": mvex["supervisor_exit_code"],
            "servers": mvex["servers"],
            "total_expected": mvex_expected,
            "total_completed": mvex_completed,
            "total_failed": mvex_failed,
            "samples": mvex_samples,
        },
        "pilot": {
            "attempt_id": PILOT_ID,
            "parent_attempt": MVEX_ID,
            "evidence_status": "QUARANTINED",
            "verdict": verdict,
            "gate_passed": False,
            "sidecars_verified": pilot["sidecars_verified"],
            "supervisor_exit_code": pilot["supervisor_exit_code"],
            "servers": pilot["servers"],
            "total_expected": pilot_expected,
            "total_completed": pilot_completed,
            "total_failed": pilot_failed,
            "silent_exclusions": 0,
            "failed_sample": failed_sample,
            "sharegpt_bracketed": sharegpt_bracketed,
            "sharegpt_bracket_report": bracket_report,
            "samples": pilot_samples,
        },
        "fallacy_scan": fallacy_scan(),
        "next_gate": {
            "required_soft_nofile": 65535,
            "sharegpt_measurement_window_s_to_mvex": 300,
            "preserve_sharegpt_completion_distribution": True,
            "formal_blocked": True,
        },
    }
    require(len(report["fallacy_scan"]) == 11, "fallacy scan coverage mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_sha = write_json_with_hash(output_dir / "comparative_pilot_audit_report.json", report)
    validation_sha = write_text_with_hash(
        output_dir / "comparative_validation_report.md",
        build_validation_markdown(report),
    )
    source_files = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "files": [
            {
                "path": str(path.relative_to(raw_dir)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
        "generated": {
            "comparative_pilot_audit_report.json": report_sha,
            "comparative_validation_report.md": validation_sha,
        },
    }
    write_json_with_hash(output_dir / "artifact_sha256_manifest.json", manifest)
    print(
        json.dumps(
            {
                "mvex": f"{mvex_completed}/{mvex_expected}",
                "pilot": f"{pilot_completed}/{pilot_expected}",
                "pilot_failed": pilot_failed,
                "sharegpt_bracketed": sharegpt_bracketed,
                "verdict": verdict,
                "a2_overall_status": report["a2_overall_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"verification failed: {exc}")
        raise SystemExit(2) from exc
