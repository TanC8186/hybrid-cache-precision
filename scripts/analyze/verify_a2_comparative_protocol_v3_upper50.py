"""Audit the A2 protocol-v3 chain through the rate-50 upper-neighbor MVEx."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

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
    audit_attempt,
    fallacy_scan,
)
from scripts.analyze.verify_a2_comparative_protocol_v3_upper import (
    UPPER_CONFIG,
    UPPER_CONFIG_SHA256,
    UPPER_MVEX_ID,
    UPPER_ROOT_COMMIT,
    audit_samples,
    full_threshold_bracket_report,
    request_totals,
)
from scripts.analyze.verify_a2_comparative_serving_pilot import ALLOCATIONS
from scripts.analyze.verify_a2_reproduction import (
    VerificationError,
    require,
    sha256_file,
    utc_timestamp,
    write_json_with_hash,
    write_text_with_hash,
)

UPPER50_MVEX_ID = (
    "a2-comparative-serving-sharegpt300-upper50-piecewise-b3bd79a-westd-01"
)
UPPER50_ROOT_COMMIT = "b3bd79a142b87d602e2de9a35e2702c3ca721924"
UPPER50_CONFIG = "a2_comparative_piecewise_sharegpt300_upper50.yaml"
UPPER50_CONFIG_SHA256 = (
    "709a7d5037ec4762e704586e138344744d9d3d65fcb71551248dd855616a5338"
)


def classify_rate50_linked_chain(
    *,
    all_attempt_integrity_passed: bool,
    full_threshold_bracketed: bool,
    servers_clean: bool,
    graph_mode_proven: bool,
) -> str:
    if all(
        (
            all_attempt_integrity_passed,
            full_threshold_bracketed,
            servers_clean,
            graph_mode_proven,
        )
    ):
        return "PROTOCOL_V3_RATE50_LINKED_BRACKET_MVEX_PASSED"
    return "PROTOCOL_V3_RATE50_LINKED_BRACKET_MVEX_REVIEW_REQUIRED"


def build_validation_markdown(report: Mapping[str, object]) -> str:
    rows = []
    for role in ("lower_mvex", "upper40_mvex", "upper50_mvex"):
        attempt = report[role]
        assert isinstance(attempt, Mapping)
        samples = attempt["samples"]
        assert isinstance(samples, Sequence)
        for sample in samples:
            assert isinstance(sample, Mapping)
            if role == "lower_mvex" and sample["offered_rate"] != 30.0:
                continue
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
        "Rate 30 is sustainable and rate 50 is unsustainable for every "
        "allocation at all five TTFT thresholds. The linked protocol-v3 MVEx "
        "gate passes without pooling request denominators."
        if report["gate_passed"]
        else "The linked rate-30/rate-50 bracket did not pass every threshold. "
        "No comparative pilot is permitted."
    )
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_comparative_protocol_v3_rate50_linked_mvex_v1

## Validation Report

- **FD MVEx**: `{FD_MVEX_ID}`
- **Lower MVEx**: `{SHAREGPT_MVEX_ID}`
- **Preserved rate-40 MVEx**: `{UPPER_MVEX_ID}`
- **Rate-50 MVEx**: `{UPPER50_MVEX_ID}`
- **Verdict**: `{report['verdict']}`
- **Evidence Status**: `{report['evidence_status']}`
- **Overall A2 Status**: `{report['a2_overall_status']}`

### ShareGPT Bracket Evidence

| Attempt role | Allocation | Offered req/s | Completed | Failed | Throughput/offered | P99 TTFT ms | P99 TPOT ms | Drain s | Sustainable TTFT thresholds ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

### Gate Result

{gate_text}

### Evidence Boundary

All four attempts retain independent denominators. The rate-20/30 and rate-40
attempts remain `QUARANTINED` as standalone failed brackets. Their intact gate
observations are linked for protocol validation only; no MVEx request row is a
pilot, formal, or paper efficacy observation. A passing chain permits a new
single-seed comparative pilot.

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
    upper40_mvex = audit_attempt(
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
    upper50_mvex = audit_attempt(
        raw_dir,
        attempt_id=UPPER50_MVEX_ID,
        expected_parent=UPPER_MVEX_ID,
        expected_root_commit=UPPER50_ROOT_COMMIT,
        expected_phase="sharegpt_upper_neighbor_r50_mvex",
        expected_config_name=UPPER50_CONFIG,
        expected_config_sha256=UPPER50_CONFIG_SHA256,
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
    upper40_samples = audit_samples(
        upper40_mvex,
        expected_root_commit=UPPER_ROOT_COMMIT,
        expected_window_s=300.0,
    )
    upper50_samples = audit_samples(
        upper50_mvex,
        expected_root_commit=UPPER50_ROOT_COMMIT,
        expected_window_s=300.0,
    )

    fd_totals = request_totals(fd_samples)
    lower_totals = request_totals(lower_samples)
    upper40_totals = request_totals(upper40_samples)
    upper50_totals = request_totals(upper50_samples)
    expected_totals = (
        (fd_totals, {"expected": 3000, "completed": 3000, "failed": 0}),
        (
            lower_totals,
            {"expected": 45000, "completed": 45000, "failed": 0},
        ),
        (
            upper40_totals,
            {"expected": 36000, "completed": 36000, "failed": 0},
        ),
        (
            upper50_totals,
            {"expected": 45000, "completed": 45000, "failed": 0},
        ),
    )
    for actual, expected in expected_totals:
        require(actual == expected, f"denominator mismatch: {actual} != {expected}")
    require(
        all(sample["fd_limit_failures"] == 0 for sample in upper50_samples),
        "rate-50 FD signature",
    )

    bracket_report, full_threshold_bracketed = full_threshold_bracket_report(
        lower_samples,
        upper50_samples,
        lower_rate=30.0,
        upper_rate=50.0,
    )
    attempts = (fd_mvex, lower_mvex, upper40_mvex, upper50_mvex)
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
    all_attempt_integrity_passed = all(
        totals["failed"] == 0 and totals["completed"] == totals["expected"]
        for totals in (
            fd_totals,
            lower_totals,
            upper40_totals,
            upper50_totals,
        )
    )
    verdict = classify_rate50_linked_chain(
        all_attempt_integrity_passed=all_attempt_integrity_passed,
        full_threshold_bracketed=full_threshold_bracketed,
        servers_clean=servers_clean,
        graph_mode_proven=graph_mode_proven,
    )
    gate_passed = verdict == "PROTOCOL_V3_RATE50_LINKED_BRACKET_MVEX_PASSED"
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
            "totals": fd_totals,
            "samples": fd_samples,
        },
        "lower_mvex": {
            "attempt_id": SHAREGPT_MVEX_ID,
            "standalone_evidence_status": "QUARANTINED",
            "totals": lower_totals,
            "samples": lower_samples,
        },
        "upper40_mvex": {
            "attempt_id": UPPER_MVEX_ID,
            "standalone_evidence_status": "QUARANTINED",
            "totals": upper40_totals,
            "samples": upper40_samples,
        },
        "upper50_mvex": {
            "attempt_id": UPPER50_MVEX_ID,
            "totals": upper50_totals,
            "samples": upper50_samples,
        },
        "full_threshold_bracketed": full_threshold_bracketed,
        "bracket_report": bracket_report,
        "fallacy_scan": fallacy_scan(),
        "next_gate": (
            "Freeze a new single-seed comparative pilot with independent "
            "attempt ID and denominators."
            if gate_passed
            else "Do not start a comparative pilot; select a higher upper neighbor."
        ),
    }
    require(len(report["fallacy_scan"]) == 11, "fallacy scan coverage")

    output_dir.mkdir(parents=True, exist_ok=False)
    report_sha = write_json_with_hash(
        output_dir / "protocol_v3_rate50_linked_audit_report.json",
        report,
    )
    validation_sha = write_text_with_hash(
        output_dir / "protocol_v3_rate50_linked_validation_report.md",
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
            "protocol_v3_rate50_linked_audit_report.json": report_sha,
            "protocol_v3_rate50_linked_validation_report.md": validation_sha,
        },
    }
    write_json_with_hash(output_dir / "artifact_sha256_manifest.json", manifest)
    print(
        json.dumps(
            {
                "fd_mvex": fd_totals,
                "lower_mvex": lower_totals,
                "upper40_mvex": upper40_totals,
                "upper50_mvex": upper50_totals,
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
