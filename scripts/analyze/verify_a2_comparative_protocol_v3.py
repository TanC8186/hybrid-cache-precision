"""Audit the A2 client-FD and ShareGPT 300-second protocol-v3 MVEx chain."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.analyze.verify_a2_comparative_serving_pilot import (
    ALLOCATIONS,
    FD_LIMIT_SIGNATURE,
    FORBIDDEN_GRAPH_MODE,
    GRAPH_PROOF,
    RUNTIME_FAULT_SIGNATURES,
    sharegpt_bracket_report,
)
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

FD_MVEX_ID = "a2-comparative-serving-fd-mvex-piecewise-f7a79f5-westd-03"
SHAREGPT_MVEX_ID = "a2-comparative-serving-sharegpt300-mvex-piecewise-37ce9e3-westd-01"
FAILED_PILOT_ID = "a2-comparative-serving-pilot-piecewise-f7a79f5-westd-02"
FD_ROOT_COMMIT = "f7a79f5d268aae66211926ac773b76e1bf443465"
SHAREGPT_ROOT_COMMIT = "37ce9e3d46eb4de6e8607071eee2446d54a52c26"
VLLM_COMMIT = "55f47685a553ad8d776c464c59785399a98c7185"
FD_CONFIG = "a2_comparative_piecewise_serving_f7a79f5_v2.yaml"
FD_CONFIG_SHA256 = "3c821db7f155a072bdd4a545c3f4f4de2a8ea6f5a087c36c4a515e54ac84e792"
SHAREGPT_CONFIG = "a2_comparative_piecewise_sharegpt300.yaml"
SHAREGPT_CONFIG_SHA256 = "7eb66ae17a361eaf823313f2c11d210b1f67bcc7254e927fa236edaf9184327c"
ARRIVAL_TOLERANCE_FRACTION = 0.10


def classify_protocol_v3_chain(
    *,
    fd_mvex_passed: bool,
    sharegpt_requests_passed: bool,
    sharegpt_bracketed: bool,
    servers_clean: bool,
    graph_mode_proven: bool,
) -> str:
    if all(
        (
            fd_mvex_passed,
            sharegpt_requests_passed,
            sharegpt_bracketed,
            servers_clean,
            graph_mode_proven,
        )
    ):
        return "PROTOCOL_V3_MVEX_CHAIN_PASSED"
    return "PROTOCOL_V3_MVEX_CHAIN_REVIEW_REQUIRED"


def protocol_v3_disposition(verdict: str) -> dict[str, Any]:
    if verdict == "PROTOCOL_V3_MVEX_CHAIN_PASSED":
        return {
            "gate_passed": True,
            "evidence_status": "UNVERIFIED",
            "next_gate": (
                "Freeze a protocol-v3 comparative pilot with an independent "
                "attempt ID; keep the failed protocol-v2 pilot quarantined."
            ),
        }
    return {
        "gate_passed": False,
        "evidence_status": "QUARANTINED",
        "next_gate": (
            "Do not start a comparative pilot. Preserve this MVEx as a negative "
            "bracketing result and freeze a linked upper-neighbor MVEx under a "
            "new attempt ID."
        ),
    }


def verify_all_sidecars(attempt_dir: Path, expected_count: int) -> int:
    sidecars = sorted(attempt_dir.rglob("*.sha256"))
    require(
        len(sidecars) == expected_count,
        f"{attempt_dir.name}: expected {expected_count} sidecars, found {len(sidecars)}",
    )
    for sidecar in sidecars:
        verify_sidecar(sidecar)
    return len(sidecars)


def supervisor_details(supervisor_dir: Path) -> dict[str, int]:
    require((supervisor_dir / "finished_at.txt").is_file(), "supervisor finish timestamp missing")
    details = {
        "exit_code": int((supervisor_dir / "exit_code.txt").read_text(encoding="ascii").strip()),
        "soft_nofile_before": int(
            (supervisor_dir / "soft_nofile_before.txt").read_text(encoding="ascii").strip()
        ),
        "soft_nofile_after": int(
            (supervisor_dir / "soft_nofile_after.txt").read_text(encoding="ascii").strip()
        ),
    }
    require(details["exit_code"] == 0, f"{supervisor_dir.name}: supervisor exit")
    require(details["soft_nofile_before"] == 1024, f"{supervisor_dir.name}: initial nofile")
    require(details["soft_nofile_after"] == 65535, f"{supervisor_dir.name}: raised nofile")
    return details


def sample_plan(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["sample_id"]): item for item in contract["plan"]}


def audit_servers(attempt_dir: Path, expected_allocations: Sequence[str]) -> list[dict[str, Any]]:
    status_paths = sorted((attempt_dir / "servers").glob("*/*/status.json"))
    require(
        len(status_paths) == len(expected_allocations),
        f"{attempt_dir.name}: server count mismatch",
    )
    servers: list[dict[str, Any]] = []
    for status_path in status_paths:
        allocation = status_path.parents[1].name
        require(allocation in expected_allocations, f"{attempt_dir.name}: unknown server allocation")
        status = load_json(status_path)
        log_path = status_path.with_name("server.log")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        runtime_faults = {
            signature: signature in log_text for signature in RUNTIME_FAULT_SIGNATURES
        }
        require(status["status"] == "stopped", f"{attempt_dir.name}/{allocation}: status")
        require(status["returncode"] == 0, f"{attempt_dir.name}/{allocation}: return code")
        require(status["exception"] is None, f"{attempt_dir.name}/{allocation}: exception")
        require(not any(runtime_faults.values()), f"{attempt_dir.name}/{allocation}: runtime fault")
        require(GRAPH_PROOF in log_text, f"{attempt_dir.name}/{allocation}: graph proof")
        require(
            FORBIDDEN_GRAPH_MODE not in log_text,
            f"{attempt_dir.name}/{allocation}: forbidden full graph mode",
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


def audit_attempt(
    raw_dir: Path,
    *,
    attempt_id: str,
    expected_parent: str,
    expected_root_commit: str,
    expected_phase: str,
    expected_config_name: str,
    expected_config_sha256: str,
    expected_samples: int,
    expected_sidecars: int,
    expected_allocations: Sequence[str],
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
    require(sha256_file(config_path) == expected_config_sha256, f"{attempt_id}: config file hash")
    require(contract["attempt_id"] == attempt_id, f"{attempt_id}: attempt ID")
    require(contract["parent_attempt"] == expected_parent, f"{attempt_id}: parent")
    require(contract["git_commit"] == expected_root_commit, f"{attempt_id}: root commit")
    require(contract["vllm_source_commit"] == VLLM_COMMIT, f"{attempt_id}: vLLM commit")
    require(contract["config_sha256"] == expected_config_sha256, f"{attempt_id}: config hash")
    require(contract["phase"]["name"] == expected_phase, f"{attempt_id}: phase")
    require(
        summary["counts"] == {"completed_validated": expected_samples},
        f"{attempt_id}: summary counts",
    )
    require(environment["root_git"]["commit"] == expected_root_commit, f"{attempt_id}: environment root")
    require(environment["root_git"]["clean"] is True, f"{attempt_id}: dirty environment root")
    require(environment["root_git"]["status"] == "", f"{attempt_id}: environment status")
    require(environment["vllm_source_commit"] == VLLM_COMMIT, f"{attempt_id}: environment vLLM")

    return {
        "attempt_dir": attempt_dir,
        "contract": contract,
        "environment": environment,
        "summary": summary,
        "sidecars_verified": sidecars_verified,
        "supervisor": supervisor_details(supervisor_dir),
        "servers": audit_servers(attempt_dir, expected_allocations),
    }


def audit_sample(
    attempt_dir: Path,
    plan: Mapping[str, Mapping[str, Any]],
    sample_id: str,
    *,
    expected_root_commit: str,
    expected_window_s: float,
) -> dict[str, Any]:
    sample_dir = attempt_dir / "samples" / sample_id
    require(sample_id in plan, f"{sample_id}: missing from plan")
    for name in ("contract.json", "result.json", "analysis.json"):
        verify_sidecar(sample_dir / f"{name}.sha256")
    contract = load_json(sample_dir / "contract.json")
    status = load_json(sample_dir / "status.json")
    result = load_json(sample_dir / "result.json")
    analysis = load_json(sample_dir / "analysis.json")
    require(status["status"] == "completed_validated", f"{sample_id}: status")
    require(status["result_sha256"] == sha256_file(sample_dir / "result.json"), f"{sample_id}: result hash")
    require(
        status["analysis_sha256"] == sha256_file(sample_dir / "analysis.json"),
        f"{sample_id}: analysis hash",
    )
    accounting = request_accounting(result, int(plan[sample_id]["num_prompts"]))
    require(analysis["completed"] == accounting["completed"], f"{sample_id}: analysis completed")
    require(analysis["failed"] == accounting["failed"], f"{sample_id}: analysis failed")
    require(result["attempt_id"] == attempt_dir.name, f"{sample_id}: result attempt")
    require(result["sample_id"] == sample_id, f"{sample_id}: result sample")
    require(result["git_commit"] == expected_root_commit, f"{sample_id}: root commit")
    require(result["vllm_source_commit"] == VLLM_COMMIT, f"{sample_id}: vLLM commit")
    require(float(analysis["measurement_window_s"]) == expected_window_s, f"{sample_id}: window")

    arrival_ratio = float(analysis["arrival_span_over_target"])
    require(
        abs(arrival_ratio - 1.0) <= ARRIVAL_TOLERANCE_FRACTION,
        f"{sample_id}: arrival drift {arrival_ratio}",
    )
    metrics = {
        "request_throughput": float(result["request_throughput"]),
        "request_throughput_over_offered": float(analysis["request_throughput_over_offered"]),
        "reported_ttft_p99_ms": float(analysis["reported_ttft_p99_ms"]),
        "reported_tpot_p99_ms": float(analysis["reported_tpot_p99_ms"]),
        "benchmark_duration_s": float(analysis["benchmark_duration_s"]),
        "drain_after_arrival_window_s": float(analysis["drain_after_arrival_window_s"]),
    }
    require(all(math.isfinite(value) for value in metrics.values()), f"{sample_id}: non-finite metric")
    errors = list(result["errors"])
    return {
        "sample_id": sample_id,
        "allocation": str(contract["allocation"]),
        "workload": str(contract["workload"]),
        "seed": int(contract["seed"]),
        "offered_rate": float(contract["request_rate"]),
        "accounting": accounting,
        "arrival_span_over_target": arrival_ratio,
        **metrics,
        "fd_limit_failures": sum(
            FD_LIMIT_SIGNATURE in str(error) for error in errors if error
        ),
        "sustainable_ttft_thresholds_ms": sorted(
            int(threshold)
            for threshold, item in analysis["slo_sweep"].items()
            if item["sustainable"]
        ),
        "goodput_over_offered": {
            str(threshold): float(item["goodput_over_offered"])
            for threshold, item in analysis["slo_sweep"].items()
        },
    }


def fallacy_scan() -> list[dict[str, str]]:
    return [
        {
            "fallacy": "Simpson's Paradox",
            "severity": "NOTE",
            "detail": "This MVEx contains only ShareGPT and does not pool workloads.",
        },
        {
            "fallacy": "Ecological Fallacy",
            "severity": "NOTE",
            "detail": "Inference stays at the allocation-rate sample level.",
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
            "detail": "No diagnostic probabilities are reported.",
        },
        {
            "fallacy": "Regression to the Mean",
            "severity": "CAUTION",
            "detail": "The MVEx has one seed and cannot establish boundary stability.",
        },
        {
            "fallacy": "Survivorship Bias",
            "severity": "SOLID",
            "detail": "Every issued request is retained in its attempt denominator.",
        },
        {
            "fallacy": "Look-Elsewhere Effect",
            "severity": "SOLID",
            "detail": "Both frozen rates and all five TTFT thresholds are reported.",
        },
        {
            "fallacy": "Garden of Forking Paths",
            "severity": "CAUTION",
            "detail": "Protocol v3 follows a preserved failed pilot and uses a new commit and attempt chain.",
        },
        {
            "fallacy": "Correlation != Causation",
            "severity": "CAUTION",
            "detail": "The controlled protocol change supports the drain-bias diagnosis but remains an MVEx.",
        },
        {
            "fallacy": "Reverse Causality",
            "severity": "NOTE",
            "detail": "Not applicable to the controlled serving protocol intervention.",
        },
    ]


def build_validation_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for sample in report["sharegpt_mvex"]["samples"]:
        thresholds = ", ".join(str(value) for value in sample["sustainable_ttft_thresholds_ms"]) or "none"
        rows.append(
            f"| {sample['allocation']} | {sample['offered_rate']:.0f} | "
            f"{sample['accounting']['completed']:,} | {sample['accounting']['failed']:,} | "
            f"{sample['request_throughput_over_offered']:.4f} | "
            f"{sample['reported_ttft_p99_ms']:.2f} | {sample['reported_tpot_p99_ms']:.2f} | "
            f"{sample['drain_after_arrival_window_s']:.2f} | {thresholds} |"
        )
    if report["gate_passed"]:
        gate_result = """Every allocation contains a sustainable rate-20 point and an unsustainable
rate-30 point under at least one tested TTFT threshold. All server logs prove
`CUDAGraphMode.PIECEWISE`, all server processes exited normally, and no client
FD or runtime-fatal signature is present. This closes the protocol-v3 MVEx
gate, but it does not validate an efficacy claim."""
        evidence_boundary = """The chain remains `ANALYZED/UNVERIFIED` because it uses one seed and two rates.
It permits a newly frozen protocol-v3 comparative pilot. The failed protocol-v2
pilot remains `QUARANTINED`, contributes zero rows to the new denominator, and
must not be relabeled or merged."""
    else:
        gate_result = """Request conservation and runtime-integrity checks passed, but the predeclared
rate-20/rate-30 ShareGPT bracket did not hold for every allocation. This is a
scientific gate failure, not a missing-artifact failure. The complete MVEx is
retained as a negative bracketing result and does not permit a comparative
pilot."""
        evidence_boundary = """The chain is `ANALYZED/QUARANTINED`. Its requests remain available for
protocol diagnosis but contribute zero rows to a pilot or formal efficacy
denominator. A linked upper-neighbor MVEx must use a new attempt ID; the failed
protocol-v2 pilot also remains separately `QUARANTINED`."""
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_comparative_protocol_v3_mvex_v1

## Validation Report

- **FD MVEx**: `{FD_MVEX_ID}`
- **ShareGPT Window MVEx**: `{SHAREGPT_MVEX_ID}`
- **Verdict**: `{report['verdict']}`
- **Evidence Status**: `{report['evidence_status']}`
- **Overall A2 Status**: `{report['a2_overall_status']}`

The client-FD MVEx completed 3,000/3,000 requests with zero failures after
raising soft `nofile` from 1024 to 65535. The linked ShareGPT MVEx completed
{report['sharegpt_mvex']['total_completed']:,}/
{report['sharegpt_mvex']['total_expected']:,} requests with
{report['sharegpt_mvex']['total_failed']:,} failures under a 300-second arrival
window while preserving the trace completion-length distribution.

### ShareGPT Results

| Allocation | Offered req/s | Completed | Failed | Throughput/offered | P99 TTFT ms | P99 TPOT ms | Drain s | Sustainable TTFT thresholds ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

### Gate Result

{gate_result}

### Evidence Boundary

{evidence_boundary}

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

    fd_mvex = audit_attempt(
        raw_dir,
        attempt_id=FD_MVEX_ID,
        expected_parent=FAILED_PILOT_ID,
        expected_root_commit=FD_ROOT_COMMIT,
        expected_phase="comparative_pilot_v2",
        expected_config_name=FD_CONFIG,
        expected_config_sha256=FD_CONFIG_SHA256,
        expected_samples=1,
        expected_sidecars=7,
        expected_allocations=("fp16",),
    )
    sharegpt_mvex = audit_attempt(
        raw_dir,
        attempt_id=SHAREGPT_MVEX_ID,
        expected_parent=FD_MVEX_ID,
        expected_root_commit=SHAREGPT_ROOT_COMMIT,
        expected_phase="sharegpt_window_mvex",
        expected_config_name=SHAREGPT_CONFIG,
        expected_config_sha256=SHAREGPT_CONFIG_SHA256,
        expected_samples=6,
        expected_sidecars=24,
        expected_allocations=ALLOCATIONS,
    )

    fd_plan = sample_plan(fd_mvex["contract"])
    sharegpt_plan = sample_plan(sharegpt_mvex["contract"])
    fd_samples = [
        audit_sample(
            fd_mvex["attempt_dir"],
            fd_plan,
            sample_id,
            expected_root_commit=FD_ROOT_COMMIT,
            expected_window_s=60.0,
        )
        for sample_id in sorted(fd_plan)
    ]
    sharegpt_samples = [
        audit_sample(
            sharegpt_mvex["attempt_dir"],
            sharegpt_plan,
            sample_id,
            expected_root_commit=SHAREGPT_ROOT_COMMIT,
            expected_window_s=300.0,
        )
        for sample_id in sorted(sharegpt_plan)
    ]

    fd_expected = sum(sample["accounting"]["expected"] for sample in fd_samples)
    fd_completed = sum(sample["accounting"]["completed"] for sample in fd_samples)
    fd_failed = sum(sample["accounting"]["failed"] for sample in fd_samples)
    require((fd_expected, fd_completed, fd_failed) == (3000, 3000, 0), "FD MVEx denominator")
    require(sum(sample["fd_limit_failures"] for sample in fd_samples) == 0, "FD signature remained")

    sharegpt_expected = sum(
        sample["accounting"]["expected"] for sample in sharegpt_samples
    )
    sharegpt_completed = sum(
        sample["accounting"]["completed"] for sample in sharegpt_samples
    )
    sharegpt_failed = sum(
        sample["accounting"]["failed"] for sample in sharegpt_samples
    )
    require(sharegpt_expected == 45000, "ShareGPT MVEx expected denominator")
    require(
        sharegpt_completed == sharegpt_expected and sharegpt_failed == 0,
        "ShareGPT MVEx request failures",
    )
    require(
        sum(sample["fd_limit_failures"] for sample in sharegpt_samples) == 0,
        "ShareGPT MVEx FD signature",
    )
    bracket_report, sharegpt_bracketed = sharegpt_bracket_report(sharegpt_samples)

    servers_clean = all(
        not any(server["runtime_faults"].values())
        and server["returncode"] == 0
        and server["exception"] is None
        for attempt in (fd_mvex, sharegpt_mvex)
        for server in attempt["servers"]
    )
    graph_mode_proven = all(
        server["graph_mode_proven"]
        for attempt in (fd_mvex, sharegpt_mvex)
        for server in attempt["servers"]
    )
    verdict = classify_protocol_v3_chain(
        fd_mvex_passed=fd_completed == fd_expected and fd_failed == 0,
        sharegpt_requests_passed=(
            sharegpt_completed == sharegpt_expected and sharegpt_failed == 0
        ),
        sharegpt_bracketed=sharegpt_bracketed,
        servers_clean=servers_clean,
        graph_mode_proven=graph_mode_proven,
    )
    disposition = protocol_v3_disposition(verdict)

    report = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "verification_status": "ANALYZED",
        "evidence_status": disposition["evidence_status"],
        "gate_passed": disposition["gate_passed"],
        "verdict": verdict,
        "a2_overall_status": "PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING",
        "fd_mvex": {
            "attempt_id": FD_MVEX_ID,
            "parent_attempt": FAILED_PILOT_ID,
            "root_commit": FD_ROOT_COMMIT,
            "sidecars_verified": fd_mvex["sidecars_verified"],
            "supervisor": fd_mvex["supervisor"],
            "servers": fd_mvex["servers"],
            "total_expected": fd_expected,
            "total_completed": fd_completed,
            "total_failed": fd_failed,
            "samples": fd_samples,
        },
        "sharegpt_mvex": {
            "attempt_id": SHAREGPT_MVEX_ID,
            "parent_attempt": FD_MVEX_ID,
            "root_commit": SHAREGPT_ROOT_COMMIT,
            "sidecars_verified": sharegpt_mvex["sidecars_verified"],
            "supervisor": sharegpt_mvex["supervisor"],
            "servers": sharegpt_mvex["servers"],
            "total_expected": sharegpt_expected,
            "total_completed": sharegpt_completed,
            "total_failed": sharegpt_failed,
            "sharegpt_bracketed": sharegpt_bracketed,
            "sharegpt_bracket_report": bracket_report,
            "samples": sharegpt_samples,
        },
        "fallacy_scan": fallacy_scan(),
        "next_gate": disposition["next_gate"],
    }
    require(len(report["fallacy_scan"]) == 11, "fallacy scan coverage")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_sha = write_json_with_hash(
        output_dir / "protocol_v3_mvex_audit_report.json",
        report,
    )
    validation_sha = write_text_with_hash(
        output_dir / "protocol_v3_validation_report.md",
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
            "protocol_v3_mvex_audit_report.json": report_sha,
            "protocol_v3_validation_report.md": validation_sha,
        },
    }
    write_json_with_hash(output_dir / "artifact_sha256_manifest.json", manifest)
    print(
        json.dumps(
            {
                "fd_mvex": f"{fd_completed}/{fd_expected}",
                "sharegpt_mvex": f"{sharegpt_completed}/{sharegpt_expected}",
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
