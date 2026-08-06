"""Audit the first resumable slice of the A2 protocol-v3 Random formal run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.analyze.verify_a2_comparative_protocol_v3 import (
    VLLM_COMMIT,
    audit_sample,
    audit_servers,
    sample_plan,
    supervisor_details,
    verify_all_sidecars,
)
from scripts.analyze.verify_a2_comparative_protocol_v3_upper import request_totals
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
EXPECTED_SAMPLE_IDS = (
    "fp16__random__r30__s7",
    "fp16__random__r30__s42",
    "fp16__random__r30__s2026",
    "fp16__random__r35__s7",
    "fp16__random__r35__s42",
)
EXPECTED_REQUESTS = 9_600
EXPECTED_SIDECARS = 19


def classify_formal_slice(
    *,
    requests_passed: bool,
    summary_passed: bool,
    servers_clean: bool,
    graph_mode_proven: bool,
) -> str:
    if all(
        (
            requests_passed,
            summary_passed,
            servers_clean,
            graph_mode_proven,
        )
    ):
        return "PROTOCOL_V3_RANDOM_FORMAL_SLICE_001_PASSED"
    return "PROTOCOL_V3_RANDOM_FORMAL_SLICE_001_REVIEW_REQUIRED"


def audit_slice(raw_dir: Path) -> dict[str, Any]:
    attempt_dir = raw_dir / "attempts" / ATTEMPT_ID
    supervisor_dir = raw_dir / "supervisors" / ATTEMPT_ID / "slice-001"
    require(attempt_dir.is_dir(), f"attempt directory missing: {attempt_dir}")
    require(supervisor_dir.is_dir(), f"supervisor directory missing: {supervisor_dir}")

    sidecars_verified = verify_all_sidecars(attempt_dir, EXPECTED_SIDECARS)
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
    require(len(contract["plan"]) == 45, "formal plan sample count")
    require(environment["root_git"]["commit"] == ROOT_COMMIT, "environment root commit")
    require(environment["root_git"]["clean"] is True, "environment root cleanliness")
    require(environment["root_git"]["status"] == "", "environment root status")
    require(environment["vllm_source_commit"] == VLLM_COMMIT, "environment vLLM")

    summary_states = {
        str(item["sample_id"]): str(item["status"]) for item in summary["samples"]
    }
    summary_passed = (
        summary["counts"] == {"completed_validated": 5, "not_started": 40}
        and len(summary_states) == 45
        and all(
            summary_states.get(sample_id) == "completed_validated"
            for sample_id in EXPECTED_SAMPLE_IDS
        )
        and all(
            state == "not_started"
            for sample_id, state in summary_states.items()
            if sample_id not in EXPECTED_SAMPLE_IDS
        )
    )
    require(summary_passed, "formal slice summary")

    sample_dirs = {
        path.name for path in (attempt_dir / "samples").iterdir() if path.is_dir()
    }
    require(sample_dirs == set(EXPECTED_SAMPLE_IDS), "formal slice sample directories")
    plan = sample_plan(contract)
    samples = [
        audit_sample(
            attempt_dir,
            plan,
            sample_id,
            expected_root_commit=ROOT_COMMIT,
            expected_window_s=60.0,
        )
        for sample_id in EXPECTED_SAMPLE_IDS
    ]
    totals = request_totals(samples)
    requests_passed = totals == {
        "expected": EXPECTED_REQUESTS,
        "completed": EXPECTED_REQUESTS,
        "failed": 0,
    }
    require(requests_passed, "formal slice request totals")

    servers = audit_servers(attempt_dir, ("fp16",))
    servers_clean = all(
        server["returncode"] == 0
        and server["exception"] is None
        and not any(server["runtime_faults"].values())
        for server in servers
    )
    graph_mode_proven = all(server["graph_mode_proven"] for server in servers)
    supervisor = supervisor_details(supervisor_dir)
    verdict = classify_formal_slice(
        requests_passed=requests_passed,
        summary_passed=summary_passed,
        servers_clean=servers_clean,
        graph_mode_proven=graph_mode_proven,
    )
    require(verdict.endswith("_PASSED"), "formal slice gate")

    return {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "verification_status": "ANALYZED",
        "evidence_status": "UNVERIFIED",
        "attempt_id": ATTEMPT_ID,
        "slice_id": "slice-001",
        "verdict": verdict,
        "partial_attempt": True,
        "completed_samples": len(samples),
        "planned_samples": len(contract["plan"]),
        "totals": totals,
        "sidecars_verified": sidecars_verified,
        "summary_passed": summary_passed,
        "servers_clean": servers_clean,
        "graph_mode_proven": graph_mode_proven,
        "supervisor": supervisor,
        "servers": servers,
        "samples": samples,
        "next_gate": (
            "Continue the same Random formal attempt with slice-002 and --resume; "
            "do not analyze this partial denominator as a complete formal matrix."
        ),
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for sample in report["samples"]:
        thresholds = (
            ", ".join(
                str(value) for value in sample["sustainable_ttft_thresholds_ms"]
            )
            or "none"
        )
        rows.append(
            f"| {sample['sample_id']} | {sample['accounting']['completed']:,} | "
            f"{sample['accounting']['failed']:,} | "
            f"{sample['reported_ttft_p99_ms']:.2f} | {thresholds} |"
        )
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_protocol_v3_random_formal_slice_001_v1

## Validation Report

- **Attempt**: `{report['attempt_id']}`
- **Slice**: `{report['slice_id']}`
- **Verdict**: `{report['verdict']}`
- **Partial Attempt**: `true`
- **Completed Samples**: `{report['completed_samples']}/{report['planned_samples']}`
- **Requests**: `{report['totals']['completed']:,}/{report['totals']['expected']:,}`
- **Failed Requests**: `{report['totals']['failed']:,}`

| Sample | Completed | Failed | P99 TTFT ms | Sustainable TTFT thresholds ms |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

This slice is operational evidence only. The Random formal denominator remains
incomplete at 5/45 samples and cannot support a formal efficacy claim.
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
    report = audit_slice(raw_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    report_sha = write_json_with_hash(
        output_dir / "protocol_v3_random_formal_slice_001_audit_report.json",
        report,
    )
    validation_sha = write_text_with_hash(
        output_dir / "protocol_v3_random_formal_slice_001_validation_report.md",
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
            "protocol_v3_random_formal_slice_001_audit_report.json": report_sha,
            "protocol_v3_random_formal_slice_001_validation_report.md": validation_sha,
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
