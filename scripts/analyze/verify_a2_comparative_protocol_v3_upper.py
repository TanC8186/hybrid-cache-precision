"""Audit the linked A2 protocol-v3 ShareGPT lower/upper MVEx chain."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.analyze.verify_a2_comparative_protocol_v3 import (
    FAILED_PILOT_ID,
    FD_CONFIG,
    FD_CONFIG_SHA256,
    FD_MVEX_ID,
    FD_ROOT_COMMIT,
    SHAREGPT_CONFIG,
    SHAREGPT_CONFIG_SHA256,
    SHAREGPT_MVEX_ID,
    SHAREGPT_ROOT_COMMIT,
    VLLM_COMMIT,
    audit_attempt,
    audit_sample,
    fallacy_scan,
    sample_plan,
)
from scripts.analyze.verify_a2_comparative_serving_pilot import (
    ALLOCATIONS,
    TTFT_THRESHOLDS_MS,
)
from scripts.analyze.verify_a2_reproduction import (
    VerificationError,
    require,
    sha256_file,
    utc_timestamp,
    write_json_with_hash,
    write_text_with_hash,
)

UPPER_MVEX_ID = (
    "a2-comparative-serving-sharegpt300-upper-piecewise-934d7de-westd-01"
)
UPPER_ROOT_COMMIT = "934d7de09c93d030afdb91e14801b0d4a0cac77f"
UPPER_CONFIG = "a2_comparative_piecewise_sharegpt300_upper.yaml"
UPPER_CONFIG_SHA256 = (
    "a56670f1b84cdb4638a43c7095253ebd77fb7cfb8fa696b3ca471ea331bacd2a"
)


def classify_linked_protocol_v3_chain(
    *,
    fd_mvex_passed: bool,
    lower_mvex_integrity_passed: bool,
    upper_mvex_integrity_passed: bool,
    full_threshold_bracketed: bool,
    servers_clean: bool,
    graph_mode_proven: bool,
) -> str:
    if all(
        (
            fd_mvex_passed,
            lower_mvex_integrity_passed,
            upper_mvex_integrity_passed,
            full_threshold_bracketed,
            servers_clean,
            graph_mode_proven,
        )
    ):
        return "PROTOCOL_V3_LINKED_BRACKET_MVEX_PASSED"
    return "PROTOCOL_V3_LINKED_BRACKET_MVEX_REVIEW_REQUIRED"


def full_threshold_bracket_report(
    lower_samples: Sequence[Mapping[str, Any]],
    upper_samples: Sequence[Mapping[str, Any]],
    *,
    lower_rate: float = 30.0,
    upper_rate: float = 40.0,
) -> tuple[dict[str, dict[str, dict[str, bool]]], bool]:
    report: dict[str, dict[str, dict[str, bool]]] = {}
    all_bracketed = True
    for allocation in ALLOCATIONS:
        lower = [
            sample
            for sample in lower_samples
            if sample["allocation"] == allocation
            and sample["workload"] == "sharegpt"
            and sample["offered_rate"] == lower_rate
        ]
        upper = [
            sample
            for sample in upper_samples
            if sample["allocation"] == allocation
            and sample["workload"] == "sharegpt"
            and sample["offered_rate"] == upper_rate
        ]
        threshold_report: dict[str, dict[str, bool]] = {}
        for threshold in TTFT_THRESHOLDS_MS:
            lower_sustainable = (
                len(lower) == 1
                and threshold in lower[0]["sustainable_ttft_thresholds_ms"]
            )
            upper_unsustainable = (
                len(upper) == 1
                and threshold not in upper[0]["sustainable_ttft_thresholds_ms"]
            )
            threshold_report[str(threshold)] = {
                "rate30_sustainable": lower_sustainable,
                "rate40_unsustainable": upper_unsustainable,
                "bracketed": lower_sustainable and upper_unsustainable,
            }
        report[allocation] = threshold_report
        all_bracketed = all_bracketed and all(
            item["bracketed"] for item in threshold_report.values()
        )
    return report, all_bracketed


def audit_samples(
    attempt: Mapping[str, Any],
    *,
    expected_root_commit: str,
    expected_window_s: float,
) -> list[dict[str, Any]]:
    plan = sample_plan(attempt["contract"])
    return [
        audit_sample(
            attempt["attempt_dir"],
            plan,
            sample_id,
            expected_root_commit=expected_root_commit,
            expected_window_s=expected_window_s,
        )
        for sample_id in sorted(plan)
    ]


def request_totals(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "expected": sum(sample["accounting"]["expected"] for sample in samples),
        "completed": sum(sample["accounting"]["completed"] for sample in samples),
        "failed": sum(sample["accounting"]["failed"] for sample in samples),
    }


def build_validation_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for role in ("lower_mvex", "upper_mvex"):
        for sample in report[role]["samples"]:
            thresholds = (
                ", ".join(
                    str(value)
                    for value in sample["sustainable_ttft_thresholds_ms"]
                )
                or "none"
            )
            rows.append(
                f"| {role} | {sample['allocation']} | "
                f"{sample['offered_rate']:.0f} | "
                f"{sample['accounting']['completed']:,} | "
                f"{sample['accounting']['failed']:,} | "
                f"{sample['request_throughput_over_offered']:.4f} | "
                f"{sample['reported_ttft_p99_ms']:.2f} | "
                f"{sample['reported_tpot_p99_ms']:.2f} | "
                f"{sample['drain_after_arrival_window_s']:.2f} | "
                f"{thresholds} |"
            )
    gate_text = (
        "Rate 30 is sustainable and rate 40 is unsustainable for every "
        "allocation at all five TTFT thresholds. The linked MVEx gate passes "
        "without pooling request denominators."
        if report["gate_passed"]
        else "The linked rate-30/rate-40 bracket did not pass every threshold. "
        "No comparative pilot is permitted."
    )
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_comparative_protocol_v3_linked_upper_mvex_v1

## Validation Report

- **FD MVEx**: `{FD_MVEX_ID}`
- **Lower MVEx**: `{SHAREGPT_MVEX_ID}`
- **Upper MVEx**: `{UPPER_MVEX_ID}`
- **Verdict**: `{report['verdict']}`
- **Evidence Status**: `{report['evidence_status']}`
- **Overall A2 Status**: `{report['a2_overall_status']}`

### ShareGPT Results

| Attempt role | Allocation | Offered req/s | Completed | Failed | Throughput/offered | P99 TTFT ms | P99 TPOT ms | Drain s | Sustainable TTFT thresholds ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

### Gate Result

{gate_text}

### Evidence Boundary

The FD, lower, and upper attempts retain independent denominators. The lower
20/30 attempt remains `QUARANTINED` as a standalone failed bracket; this linked
audit uses its intact rate-30 gate observation without promoting any MVEx row
to an efficacy claim. A passing chain permits a new single-seed pilot only.

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
    lower_mvex = audit_attempt(
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
    upper_mvex = audit_attempt(
        raw_dir,
        attempt_id=UPPER_MVEX_ID,
        expected_parent=SHAREGPT_MVEX_ID,
        expected_root_commit=UPPER_ROOT_COMMIT,
        expected_phase="sharegpt_upper_neighbor_mvex",
        expected_config_name=UPPER_CONFIG,
        expected_config_sha256=UPPER_CONFIG_SHA256,
        expected_samples=3,
        expected_sidecars=15,
        expected_allocations=ALLOCATIONS,
    )

    fd_samples = audit_samples(
        fd_mvex,
        expected_root_commit=FD_ROOT_COMMIT,
        expected_window_s=60.0,
    )
    lower_samples = audit_samples(
        lower_mvex,
        expected_root_commit=SHAREGPT_ROOT_COMMIT,
        expected_window_s=300.0,
    )
    upper_samples = audit_samples(
        upper_mvex,
        expected_root_commit=UPPER_ROOT_COMMIT,
        expected_window_s=300.0,
    )

    fd_totals = request_totals(fd_samples)
    lower_totals = request_totals(lower_samples)
    upper_totals = request_totals(upper_samples)
    require(fd_totals == {"expected": 3000, "completed": 3000, "failed": 0}, "FD denominator")
    require(
        lower_totals == {"expected": 45000, "completed": 45000, "failed": 0},
        "lower denominator",
    )
    require(
        upper_totals == {"expected": 36000, "completed": 36000, "failed": 0},
        "upper denominator",
    )
    require(
        all(sample["fd_limit_failures"] == 0 for sample in upper_samples),
        "upper FD signature",
    )

    bracket_report, full_threshold_bracketed = full_threshold_bracket_report(
        lower_samples,
        upper_samples,
    )
    attempts = (fd_mvex, lower_mvex, upper_mvex)
    servers_clean = all(
        not any(server["runtime_faults"].values())
        and server["returncode"] == 0
        and server["exception"] is None
        for attempt in attempts
        for server in attempt["servers"]
    )
    graph_mode_proven = all(
        server["graph_mode_proven"]
        for attempt in attempts
        for server in attempt["servers"]
    )
    verdict = classify_linked_protocol_v3_chain(
        fd_mvex_passed=fd_totals["failed"] == 0,
        lower_mvex_integrity_passed=lower_totals["failed"] == 0,
        upper_mvex_integrity_passed=upper_totals["failed"] == 0,
        full_threshold_bracketed=full_threshold_bracketed,
        servers_clean=servers_clean,
        graph_mode_proven=graph_mode_proven,
    )
    gate_passed = verdict == "PROTOCOL_V3_LINKED_BRACKET_MVEX_PASSED"
    report = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "verification_status": "ANALYZED",
        "evidence_status": "UNVERIFIED" if gate_passed else "QUARANTINED",
        "gate_passed": gate_passed,
        "verdict": verdict,
        "a2_overall_status": "PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING",
        "fd_mvex": {
            "attempt_id": FD_MVEX_ID,
            "root_commit": FD_ROOT_COMMIT,
            "totals": fd_totals,
            "samples": fd_samples,
        },
        "lower_mvex": {
            "attempt_id": SHAREGPT_MVEX_ID,
            "root_commit": SHAREGPT_ROOT_COMMIT,
            "standalone_evidence_status": "QUARANTINED",
            "totals": lower_totals,
            "samples": lower_samples,
        },
        "upper_mvex": {
            "attempt_id": UPPER_MVEX_ID,
            "root_commit": UPPER_ROOT_COMMIT,
            "totals": upper_totals,
            "samples": upper_samples,
        },
        "full_threshold_bracketed": full_threshold_bracketed,
        "bracket_report": bracket_report,
        "fallacy_scan": fallacy_scan(),
        "next_gate": (
            "Freeze a new single-seed comparative pilot with independent "
            "attempt ID and denominators."
            if gate_passed
            else "Do not start a comparative pilot; select a new upper neighbor."
        ),
    }
    require(len(report["fallacy_scan"]) == 11, "fallacy scan coverage")

    output_dir.mkdir(parents=True, exist_ok=False)
    report_sha = write_json_with_hash(
        output_dir / "protocol_v3_linked_upper_audit_report.json",
        report,
    )
    validation_sha = write_text_with_hash(
        output_dir / "protocol_v3_linked_upper_validation_report.md",
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
            "protocol_v3_linked_upper_audit_report.json": report_sha,
            "protocol_v3_linked_upper_validation_report.md": validation_sha,
        },
    }
    write_json_with_hash(output_dir / "artifact_sha256_manifest.json", manifest)
    print(
        json.dumps(
            {
                "fd_mvex": fd_totals,
                "lower_mvex": lower_totals,
                "upper_mvex": upper_totals,
                "full_threshold_bracketed": full_threshold_bracketed,
                "verdict": verdict,
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
