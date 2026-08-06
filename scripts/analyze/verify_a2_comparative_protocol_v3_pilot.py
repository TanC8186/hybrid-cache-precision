"""Audit the A2 protocol-v3 Random component or complete comparative pilot."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.analyze.verify_a2_comparative_protocol_v3 import (
    audit_attempt,
    fallacy_scan,
)
from scripts.analyze.verify_a2_comparative_protocol_v3_upper import (
    audit_samples,
    request_totals,
)
from scripts.analyze.verify_a2_comparative_serving_pilot import (
    ALLOCATIONS,
    TTFT_THRESHOLDS_MS,
)
from scripts.analyze.verify_a2_reproduction import (
    VerificationError,
    sha256_file,
    utc_timestamp,
    write_json_with_hash,
    write_text_with_hash,
)

ROOT_COMMIT = "fc078681a6bd789fddb076ca20ac67f2ffc8635c"
RANDOM_ATTEMPT_ID = (
    "a2-comparative-serving-random60-pilot-v3-piecewise-fc07868-westd-01"
)
RANDOM_PARENT_ID = (
    "a2-comparative-serving-sharegpt300-upper50-piecewise-b3bd79a-westd-01"
)
RANDOM_CONFIG = "a2_comparative_piecewise_protocol_v3_random60_pilot.yaml"
RANDOM_CONFIG_SHA256 = (
    "dcc63f7a2ad17f0a64753cc7c3d906dce45033ffba31734d2d8cb00410c31496"
)
SHAREGPT_ATTEMPT_ID = (
    "a2-comparative-serving-sharegpt300-pilot-v3-piecewise-fc07868-westd-01"
)
SHAREGPT_CONFIG = (
    "a2_comparative_piecewise_protocol_v3_sharegpt300_pilot.yaml"
)
SHAREGPT_CONFIG_SHA256 = (
    "96dd2dcb807a494d96f5c481a4293ad6321f6ada28557ff09b474f908d049c18"
)


def classify_pilot(
    *,
    component: str,
    requests_passed: bool,
    bracketed: bool,
    servers_clean: bool,
    graph_mode_proven: bool,
) -> str:
    passed = all(
        (
            requests_passed,
            bracketed,
            servers_clean,
            graph_mode_proven,
        )
    )
    if component == "random":
        return (
            "PROTOCOL_V3_RANDOM_PILOT_COMPONENT_PASSED"
            if passed
            else "PROTOCOL_V3_RANDOM_PILOT_COMPONENT_REVIEW_REQUIRED"
        )
    return (
        "PROTOCOL_V3_COMPARATIVE_PILOT_PASSED"
        if passed
        else "PROTOCOL_V3_COMPARATIVE_PILOT_REVIEW_REQUIRED"
    )


def full_threshold_bracket_report(
    samples: Sequence[Mapping[str, Any]],
    workload: str,
) -> tuple[dict[str, dict[str, dict[str, bool]]], bool]:
    report: dict[str, dict[str, dict[str, bool]]] = {}
    all_bracketed = True
    for allocation in ALLOCATIONS:
        low = [
            sample
            for sample in samples
            if sample["allocation"] == allocation
            and sample["workload"] == workload
            and sample["offered_rate"] == 30.0
        ]
        high = [
            sample
            for sample in samples
            if sample["allocation"] == allocation
            and sample["workload"] == workload
            and sample["offered_rate"] == 50.0
        ]
        threshold_report: dict[str, dict[str, bool]] = {}
        for threshold in TTFT_THRESHOLDS_MS:
            low_sustainable = (
                len(low) == 1
                and threshold in low[0]["sustainable_ttft_thresholds_ms"]
            )
            high_unsustainable = (
                len(high) == 1
                and threshold not in high[0]["sustainable_ttft_thresholds_ms"]
            )
            threshold_report[str(threshold)] = {
                "rate30_sustainable": low_sustainable,
                "rate50_unsustainable": high_unsustainable,
                "bracketed": low_sustainable and high_unsustainable,
            }
        report[allocation] = threshold_report
        all_bracketed = all_bracketed and all(
            item["bracketed"] for item in threshold_report.values()
        )
    return report, all_bracketed


def audit_component(
    raw_dir: Path,
    *,
    attempt_id: str,
    parent_id: str,
    phase: str,
    config_name: str,
    config_sha256: str,
    workload: str,
    window_s: float,
    expected_requests: int,
) -> dict[str, Any]:
    attempt = audit_attempt(
        raw_dir,
        attempt_id=attempt_id,
        expected_parent=parent_id,
        expected_root_commit=ROOT_COMMIT,
        expected_phase=phase,
        expected_config_name=config_name,
        expected_config_sha256=config_sha256,
        expected_samples=9,
        expected_sidecars=33,
        expected_allocations=ALLOCATIONS,
    )
    samples = audit_samples(
        attempt,
        expected_root_commit=ROOT_COMMIT,
        expected_window_s=window_s,
    )
    totals = request_totals(samples)
    requests_passed = totals == {
        "expected": expected_requests,
        "completed": expected_requests,
        "failed": 0,
    }
    bracket_report, bracketed = full_threshold_bracket_report(samples, workload)
    servers_clean = all(
        not any(server["runtime_faults"].values())
        and server["returncode"] == 0
        and server["exception"] is None
        for server in attempt["servers"]
    )
    graph_mode_proven = all(
        server["graph_mode_proven"] for server in attempt["servers"]
    )
    return {
        "attempt_id": attempt_id,
        "parent_attempt": parent_id,
        "workload": workload,
        "window_s": window_s,
        "totals": totals,
        "requests_passed": requests_passed,
        "bracketed": bracketed,
        "bracket_report": bracket_report,
        "servers_clean": servers_clean,
        "graph_mode_proven": graph_mode_proven,
        "sidecars_verified": attempt["sidecars_verified"],
        "supervisor": attempt["supervisor"],
        "servers": attempt["servers"],
        "samples": samples,
    }


def build_validation_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for component_name in ("random", "sharegpt"):
        component = report.get(component_name)
        if component is None:
            continue
        for sample in component["samples"]:
            thresholds = (
                ", ".join(
                    str(value)
                    for value in sample["sustainable_ttft_thresholds_ms"]
                )
                or "none"
            )
            rows.append(
                f"| {component_name} | {sample['allocation']} | "
                f"{sample['offered_rate']:.0f} | "
                f"{sample['accounting']['completed']:,} | "
                f"{sample['accounting']['failed']:,} | "
                f"{sample['request_throughput_over_offered']:.4f} | "
                f"{sample['reported_ttft_p99_ms']:.2f} | "
                f"{sample['reported_tpot_p99_ms']:.2f} | "
                f"{thresholds} |"
            )
    gate_text = (
        "All audited pilot components pass request conservation, full "
        "rate-30/rate-50 bracketing, server integrity, and PIECEWISE proof."
        if report["gate_passed"]
        else "At least one audited pilot gate failed. Formal expansion is blocked."
    )
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_comparative_protocol_v3_pilot_v1

## Validation Report

- **Audited Component**: `{report['component']}`
- **Verdict**: `{report['verdict']}`
- **Evidence Status**: `{report['evidence_status']}`
- **Overall A2 Status**: `PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING`

### Pilot Results

| Workload | Allocation | Offered req/s | Completed | Failed | Throughput/offered | P99 TTFT ms | P99 TPOT ms | Sustainable TTFT thresholds ms |
|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

### Gate Result

{gate_text}

### Evidence Boundary

Pilot rows remain `ANALYZED/UNVERIFIED` and cannot support paper efficacy
claims. Formal execution requires the complete Random plus ShareGPT pilot to
pass; historical failed pilots and every MVEx denominator remain excluded.

### Fallacy Scan

- **Coverage**: 11/11
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--component",
        choices=("random", "suite"),
        default="suite",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    random = audit_component(
        raw_dir,
        attempt_id=RANDOM_ATTEMPT_ID,
        parent_id=RANDOM_PARENT_ID,
        phase="comparative_random60_pilot_v3",
        config_name=RANDOM_CONFIG,
        config_sha256=RANDOM_CONFIG_SHA256,
        workload="random",
        window_s=60.0,
        expected_requests=21600,
    )
    sharegpt = None
    components = [random]
    if args.component == "suite":
        sharegpt = audit_component(
            raw_dir,
            attempt_id=SHAREGPT_ATTEMPT_ID,
            parent_id=RANDOM_ATTEMPT_ID,
            phase="comparative_sharegpt300_pilot_v3",
            config_name=SHAREGPT_CONFIG,
            config_sha256=SHAREGPT_CONFIG_SHA256,
            workload="sharegpt",
            window_s=300.0,
            expected_requests=108000,
        )
        components.append(sharegpt)

    verdict = classify_pilot(
        component=args.component,
        requests_passed=all(item["requests_passed"] for item in components),
        bracketed=all(item["bracketed"] for item in components),
        servers_clean=all(item["servers_clean"] for item in components),
        graph_mode_proven=all(item["graph_mode_proven"] for item in components),
    )
    gate_passed = verdict.endswith("_PASSED")
    report = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "component": args.component,
        "verification_status": "ANALYZED",
        "evidence_status": "UNVERIFIED" if gate_passed else "QUARANTINED",
        "gate_passed": gate_passed,
        "verdict": verdict,
        "random": random,
        "sharegpt": sharegpt,
        "fallacy_scan": fallacy_scan(),
        "next_gate": (
            "Run the linked ShareGPT-300 pilot under the same commit."
            if args.component == "random" and gate_passed
            else (
                "Freeze independently resumable formal slices."
                if gate_passed
                else "Do not expand compute; preserve and diagnose this pilot."
            )
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    report_sha = write_json_with_hash(
        output_dir / "protocol_v3_pilot_audit_report.json",
        report,
    )
    validation_sha = write_text_with_hash(
        output_dir / "protocol_v3_pilot_validation_report.md",
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
            "protocol_v3_pilot_audit_report.json": report_sha,
            "protocol_v3_pilot_validation_report.md": validation_sha,
        },
    }
    write_json_with_hash(output_dir / "artifact_sha256_manifest.json", manifest)
    print(
        json.dumps(
            {
                "component": args.component,
                "random": random["totals"],
                "sharegpt": sharegpt["totals"] if sharegpt else None,
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
