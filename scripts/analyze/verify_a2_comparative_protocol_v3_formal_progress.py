"""Audit cumulative progress for the sliced A2 protocol-v3 Random formal run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.analyze.verify_a2_comparative_protocol_v3 import (
    VLLM_COMMIT,
    audit_sample,
    sample_plan,
    supervisor_details,
    verify_all_sidecars,
)
from scripts.analyze.verify_a2_comparative_protocol_v3_upper import request_totals
from scripts.analyze.verify_a2_comparative_serving_pilot import (
    FORBIDDEN_GRAPH_MODE,
    GRAPH_PROOF,
    RUNTIME_FAULT_SIGNATURES,
)
from scripts.analyze.verify_a2_reproduction import (
    VerificationError,
    load_json,
    require,
    sha256_file,
    utc_timestamp,
    write_json_with_hash,
    write_text_with_hash,
)

ROOT_COMMIT = "310865011daf2a9d8d694eddded0411fd956fd95"
ATTEMPT_ID = "a2-comparative-serving-random60-formal-v3-piecewise-3108650-westd-01"
PARENT_ID = (
    "a2-comparative-serving-sharegpt300-pilot-v3-piecewise-fc07868-westd-01"
)
CONFIG_NAME = "a2_comparative_piecewise_protocol_v3_random60_formal.yaml"
CONFIG_SHA256 = "0523188ff20210a376ae35c86d0c3a624990892ac7f5199ca26d86cef091fe02"
PHASE_NAME = "comparative_random60_formal_v3"
PLAN_SIZE = 45
SLICE_SIZE = 5
MAX_SLICES = PLAN_SIZE // SLICE_SIZE
ALLOCATIONS = ("fp16", "int4", "packed_per_layer")


def classify_formal_progress(
    *,
    slice_number: int,
    requests_passed: bool,
    summary_passed: bool,
    servers_clean: bool,
    graph_mode_proven: bool,
) -> str:
    suffix = f"{slice_number:03d}"
    if all(
        (
            requests_passed,
            summary_passed,
            servers_clean,
            graph_mode_proven,
        )
    ):
        return f"PROTOCOL_V3_RANDOM_FORMAL_SLICE_{suffix}_PASSED"
    return f"PROTOCOL_V3_RANDOM_FORMAL_SLICE_{suffix}_REVIEW_REQUIRED"


def audit_server_sessions(
    attempt_dir: Path,
    expected_allocations: Sequence[str],
) -> list[dict[str, Any]]:
    status_paths = sorted((attempt_dir / "servers").glob("*/*/status.json"))
    require(
        len(status_paths) == len(expected_allocations),
        f"{ATTEMPT_ID}: server session count",
    )
    actual_allocations = [path.parents[1].name for path in status_paths]
    require(
        Counter(actual_allocations) == Counter(expected_allocations),
        f"{ATTEMPT_ID}: server allocation sessions",
    )

    servers: list[dict[str, Any]] = []
    for status_path in status_paths:
        allocation = status_path.parents[1].name
        require(allocation in ALLOCATIONS, f"{ATTEMPT_ID}: unknown allocation")
        status = load_json(status_path)
        log_path = status_path.with_name("server.log")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        runtime_faults = {
            signature: signature in log_text for signature in RUNTIME_FAULT_SIGNATURES
        }
        require(status["status"] == "stopped", f"{allocation}: server status")
        require(status["returncode"] == 0, f"{allocation}: server return code")
        require(status["exception"] is None, f"{allocation}: server exception")
        require(not any(runtime_faults.values()), f"{allocation}: runtime fault")
        require(GRAPH_PROOF in log_text, f"{allocation}: graph proof")
        require(FORBIDDEN_GRAPH_MODE not in log_text, f"{allocation}: forbidden graph")
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


def audit_progress(raw_dir: Path, slice_number: int) -> dict[str, Any]:
    require(1 <= slice_number <= MAX_SLICES, "slice number outside formal plan")
    completed_count = slice_number * SLICE_SIZE
    attempt_dir = raw_dir / "attempts" / ATTEMPT_ID
    supervisor_root = raw_dir / "supervisors" / ATTEMPT_ID
    require(attempt_dir.is_dir(), f"attempt directory missing: {attempt_dir}")
    require(supervisor_root.is_dir(), f"supervisor directory missing: {supervisor_root}")

    contract = load_json(attempt_dir / "attempt_contract.json")
    environment = load_json(attempt_dir / "environment.json")
    summary = load_json(attempt_dir / "summary.json")
    config_path = raw_dir / "experiments" / "configs" / CONFIG_NAME
    require(config_path.is_file(), f"config missing: {config_path}")
    require(sha256_file(config_path) == CONFIG_SHA256, "formal config hash")
    require(contract["attempt_id"] == ATTEMPT_ID, "formal attempt ID")
    require(contract["parent_attempt"] == PARENT_ID, "formal parent attempt")
    require(contract["git_commit"] == ROOT_COMMIT, "formal root commit")
    require(contract["vllm_source_commit"] == VLLM_COMMIT, "formal vLLM commit")
    require(contract["config_sha256"] == CONFIG_SHA256, "formal contract config hash")
    require(contract["phase"]["name"] == PHASE_NAME, "formal phase")
    require(len(contract["plan"]) == PLAN_SIZE, "formal plan sample count")
    require(environment["root_git"]["commit"] == ROOT_COMMIT, "environment root commit")
    require(environment["root_git"]["clean"] is True, "environment root cleanliness")
    require(environment["root_git"]["status"] == "", "environment root status")
    require(environment["vllm_source_commit"] == VLLM_COMMIT, "environment vLLM")

    expected_plan = contract["plan"][:completed_count]
    expected_sample_ids = tuple(str(item["sample_id"]) for item in expected_plan)
    expected_requests = sum(int(item["num_prompts"]) for item in expected_plan)
    expected_allocations = [
        str(contract["plan"][(index - 1) * SLICE_SIZE]["allocation"])
        for index in range(1, slice_number + 1)
    ]
    expected_counts = {"completed_validated": completed_count}
    if completed_count < PLAN_SIZE:
        expected_counts["not_started"] = PLAN_SIZE - completed_count

    summary_states = {
        str(item["sample_id"]): str(item["status"]) for item in summary["samples"]
    }
    summary_passed = (
        summary["counts"] == expected_counts
        and len(summary_states) == PLAN_SIZE
        and all(
            summary_states.get(sample_id) == "completed_validated"
            for sample_id in expected_sample_ids
        )
        and all(
            state == "not_started"
            for sample_id, state in summary_states.items()
            if sample_id not in expected_sample_ids
        )
    )
    require(summary_passed, "formal progress summary")

    sample_dirs = {
        path.name for path in (attempt_dir / "samples").iterdir() if path.is_dir()
    }
    require(sample_dirs == set(expected_sample_ids), "formal progress sample directories")
    expected_sidecars = 3 + 3 * completed_count + slice_number
    sidecars_verified = verify_all_sidecars(attempt_dir, expected_sidecars)

    plan = sample_plan(contract)
    samples = [
        audit_sample(
            attempt_dir,
            plan,
            sample_id,
            expected_root_commit=ROOT_COMMIT,
            expected_window_s=60.0,
        )
        for sample_id in expected_sample_ids
    ]
    totals = request_totals(samples)
    requests_passed = totals == {
        "expected": expected_requests,
        "completed": expected_requests,
        "failed": 0,
    }
    require(requests_passed, "formal progress request totals")

    servers = audit_server_sessions(attempt_dir, expected_allocations)
    servers_clean = all(
        server["returncode"] == 0
        and server["exception"] is None
        and not any(server["runtime_faults"].values())
        for server in servers
    )
    graph_mode_proven = all(server["graph_mode_proven"] for server in servers)
    supervisors = [
        {
            "slice_id": f"slice-{index:03d}",
            **supervisor_details(supervisor_root / f"slice-{index:03d}"),
        }
        for index in range(1, slice_number + 1)
    ]
    verdict = classify_formal_progress(
        slice_number=slice_number,
        requests_passed=requests_passed,
        summary_passed=summary_passed,
        servers_clean=servers_clean,
        graph_mode_proven=graph_mode_proven,
    )
    require(verdict.endswith("_PASSED"), "formal progress gate")

    return {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "verification_status": "ANALYZED",
        "evidence_status": "UNVERIFIED",
        "attempt_id": ATTEMPT_ID,
        "slice_id": f"slice-{slice_number:03d}",
        "verdict": verdict,
        "partial_attempt": completed_count < PLAN_SIZE,
        "completed_samples": completed_count,
        "planned_samples": PLAN_SIZE,
        "totals": totals,
        "sidecars_verified": sidecars_verified,
        "summary_passed": summary_passed,
        "servers_clean": servers_clean,
        "graph_mode_proven": graph_mode_proven,
        "supervisors": supervisors,
        "servers": servers,
        "samples": samples,
        "next_gate": (
            f"Continue the same Random formal attempt with slice-{slice_number + 1:03d} "
            "and --resume."
            if completed_count < PLAN_SIZE
            else "Run the linked ShareGPT formal attempt."
        ),
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_protocol_v3_random_formal_progress_v1

## Validation Report

- **Attempt**: `{report['attempt_id']}`
- **Slice**: `{report['slice_id']}`
- **Verdict**: `{report['verdict']}`
- **Partial Attempt**: `{str(report['partial_attempt']).lower()}`
- **Completed Samples**: `{report['completed_samples']}/{report['planned_samples']}`
- **Requests**: `{report['totals']['completed']:,}/{report['totals']['expected']:,}`
- **Failed Requests**: `{report['totals']['failed']:,}`
- **Verified Sidecars**: `{report['sidecars_verified']}`

This report validates cumulative operational progress only. Quantitative formal
claims remain blocked until the complete Random and ShareGPT formal attempts and
their independent reproducibility gate pass.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slice-number", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    report = audit_progress(raw_dir, args.slice_number)

    output_dir.mkdir(parents=True, exist_ok=False)
    stem = f"protocol_v3_random_formal_{report['slice_id'].replace('-', '_')}"
    report_sha = write_json_with_hash(
        output_dir / f"{stem}_audit_report.json",
        report,
    )
    validation_sha = write_text_with_hash(
        output_dir / f"{stem}_validation_report.md",
        build_markdown(report),
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
            f"{stem}_audit_report.json": report_sha,
            f"{stem}_validation_report.md": validation_sha,
        },
    }
    write_json_with_hash(output_dir / "artifact_sha256_manifest.json", manifest)
    print(
        json.dumps(
            {
                "attempt_id": report["attempt_id"],
                "slice_id": report["slice_id"],
                "totals": report["totals"],
                "verdict": report["verdict"],
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
