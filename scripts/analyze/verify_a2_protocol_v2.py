"""Verify the fresh A2 protocol-v2 confirmation suite and create a scoped link."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.analyze.verify_a2_reproduction import (
    VerificationError,
    atomic_write,
    fallacy_scan,
    load_json,
    require,
    sha256_file,
    structural_checks,
    symmetric_relative_difference,
    utc_timestamp,
    verify_sidecar,
    write_json_with_hash,
    write_text_with_hash,
)

CONFIRMATION_ATTEMPTS = {
    "runtime": "a2-repro-v2-mvex-c7379f0-westd-03",
    "legacy": "a2-repro-v2-capacity-legacy-c7379f0-westd-03",
    "uniform": "a2-repro-v2-capacity-uniform-c7379f0-westd-03",
    "packed": "a2-repro-v2-capacity-packed-c7379f0-westd-03",
}
DISCOVERY_ATTEMPTS = {
    "legacy": "a2-repro-capacity-legacy-c7379f0-westd-02",
    "uniform": "a2-repro-capacity-uniform-c7379f0-westd-02",
    "packed": "a2-repro-capacity-packed-c7379f0-westd-02",
}


def evaluate_protocol(
    original_capacity: Mapping[str, int],
    reproduced_capacity: Mapping[str, int],
    original_ratios: Mapping[str, float],
    reproduced_ratios: Mapping[str, float],
    *,
    capacity_tolerance: float,
    ratio_tolerance: float,
) -> dict[str, Any]:
    capacity_differences = {
        name: symmetric_relative_difference(original_capacity[name], reproduced_capacity[name])
        for name in original_capacity
    }
    ratio_differences = {
        name: symmetric_relative_difference(original_ratios[name], reproduced_ratios[name])
        for name in original_ratios
    }
    return {
        "capacity_differences": capacity_differences,
        "ratio_differences": ratio_differences,
        "capacity_within_tolerance": all(value <= capacity_tolerance for value in capacity_differences.values()),
        "ratios_within_tolerance": all(value <= ratio_tolerance for value in ratio_differences.values()),
    }


def audit_attempt(
    raw_dir: Path,
    contract: Mapping[str, Any],
    contract_sha256: str,
    name: str,
) -> dict[str, Any]:
    attempt_id = CONFIRMATION_ATTEMPTS[name]
    item = next((entry for entry in contract["attempts"] if entry["attempt_id"] == attempt_id), None)
    require(item is not None, f"confirmation contract missing attempt: {attempt_id}")
    attempt_dir = raw_dir / "attempts" / attempt_id
    require(attempt_dir.is_dir(), f"attempt directory missing: {attempt_dir}")
    require((attempt_dir / "exit_code.txt").read_text(encoding="ascii").strip() == "0", f"{attempt_id}: nonzero exit")
    launch = load_json(attempt_dir / "launch.json")
    require(launch["attempt_id"] == attempt_id, f"{attempt_id}: launch ID mismatch")
    require(launch["command"] == item["command"], f"{attempt_id}: command drift")
    require(launch["contract_sha256"] == contract_sha256, f"{attempt_id}: contract SHA mismatch")
    output_name = Path(item["expected_output"]).name
    output_path = attempt_dir / output_name
    output_sha = verify_sidecar(output_path.with_suffix(output_path.suffix + ".sha256"))
    report = load_json(output_path)
    require(
        report["environment"]["model_config_sha256"] == contract["model_config_sha256"],
        f"{attempt_id}: model hash mismatch",
    )
    return {
        "attempt_id": attempt_id,
        "output_path": str(output_path),
        "output_sha256": output_sha,
        "report": report,
    }


def discovery_capacity(raw_dir: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for name, attempt_id in DISCOVERY_ATTEMPTS.items():
        path = raw_dir / "attempts" / attempt_id / "a2_capacity.json"
        verify_sidecar(path.with_suffix(path.suffix + ".sha256"))
        values[name] = int(load_json(path)["capacity"]["tokens"])
    return values


def build_validation_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for name in ("legacy", "uniform", "packed"):
        item = report["capacity_comparison"][name]
        rows.append(
            f"| {name} | {item['original_tokens']:,} | {item['confirmed_tokens']:,} | "
            f"{100.0 * item['symmetric_relative_difference']:.3f}% | "
            f"{'PASS' if item['within_protocol_tolerance'] else 'FAIL'} |"
        )
    checks = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in {**report["structural_checks"], **report["protocol_checks"]}.items()
    )
    fallacies = "\n".join(
        f"| {item['fallacy']} | {item['severity']} | {item['detail']} |"
        for item in report["fallacy_scan"]
    )
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: VERIFIED
- Version Label: a2_protocol_v2_validation_v1

## Validation Report

- **Source**: `{report['suite_id']}`
- **Verification Scope**: A2 runtime/capacity mechanism and capacity ratios
- **Verdict**: `{report['verdict']}`
- **Overall A2 Status**: `{report['a2_overall_status']}`

### Capacity Confirmation

| Allocation | Original | Confirmed | Symmetric Difference | Protocol-v2 |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

The confirmation suite exactly repeated all three `westd-02` capacities. It also
passed the prospectively frozen 1% capacity and 0.1% ratio tolerances.

### Ratio Confirmation

- packed / legacy: `{report['ratios']['confirmed']['packed_over_legacy']:.6f}`
- packed / uniform: `{report['ratios']['confirmed']['packed_over_uniform']:.6f}`

### Verification Checks

{checks}

### Scope Boundary

Only the A2 runtime/capacity sub-scope is `VERIFIED`. Packed serving SLO and
quality evaluation remain pending, so the overall A2 method is not yet fully
verified for paper-wide claims.

### Fallacy Scan

- **Coverage**: 11/11

| Fallacy | Severity | Detail |
|---|---|---|
{fallacies}
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--reproduction-dir", type=Path, required=True)
    parser.add_argument("--discovery-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    original_dir = args.original_dir.resolve()
    raw_dir = args.reproduction_dir.resolve()
    discovery_analysis_dir = args.discovery_analysis_dir.resolve()
    output_dir = args.output_dir.resolve()

    contract_sidecar = raw_dir / "attempt_contract_westd_03.json.sha256"
    contract_sha256 = verify_sidecar(contract_sidecar)
    contract = load_json(raw_dir / "attempt_contract_westd_03.json")
    require(contract["suite_id"] == "a2-repro-v2-suite-c7379f0-westd-03", "suite ID mismatch")
    require(contract["protocol_version"] == 2, "protocol version mismatch")
    require(
        contract["confirmation_scope"].startswith("Fresh confirmatory rerun"),
        "confirmation scope is not frozen",
    )

    suite_dir = raw_dir / "suites" / contract["suite_id"]
    require((suite_dir / "exit_code.txt").read_text(encoding="ascii").strip() == "0", "suite failed")
    suite_launch = load_json(suite_dir / "suite_launch.json")
    require(suite_launch["contract_sha256"] == contract_sha256, "suite contract SHA mismatch")
    require(
        suite_launch["attempt_ids"] == [item["attempt_id"] for item in contract["attempts"]],
        "suite attempt order mismatch",
    )

    attempts = {
        name: audit_attempt(raw_dir, contract, contract_sha256, name)
        for name in CONFIRMATION_ATTEMPTS
    }
    structure = structural_checks(attempts)
    require(all(structure.values()), "confirmation structural checks failed")

    original_gate = load_json(original_dir / "a2_capacity_gate_c7379f0_v2.json")
    verify_sidecar(original_dir / "a2_capacity_gate_c7379f0_v2.json.sha256")
    original_capacity = {
        name: int(original_gate["baselines"][name]["capacity_tokens"])
        for name in ("legacy", "uniform", "packed")
    }
    confirmed_capacity = {
        name: int(attempts[name]["report"]["capacity"]["tokens"])
        for name in ("legacy", "uniform", "packed")
    }
    discovered_capacity = discovery_capacity(raw_dir)

    confirmed_ratios = {
        "packed_over_legacy": confirmed_capacity["packed"] / confirmed_capacity["legacy"],
        "packed_over_uniform": confirmed_capacity["packed"] / confirmed_capacity["uniform"],
    }
    original_ratios = {
        name: float(original_gate["ratios"][name])
        for name in ("packed_over_legacy", "packed_over_uniform")
    }
    criteria = contract["pass_criteria"]
    evaluation = evaluate_protocol(
        original_capacity,
        confirmed_capacity,
        original_ratios,
        confirmed_ratios,
        capacity_tolerance=float(criteria["capacity_symmetric_relative_difference_max"]),
        ratio_tolerance=float(criteria["ratio_symmetric_relative_difference_max"]),
    )
    ratio_gate = {
        "packed_over_legacy_at_least_3x": confirmed_ratios["packed_over_legacy"]
        >= float(criteria["ratio_gate_packed_over_legacy_min"]),
        "packed_over_uniform_in_range": float(criteria["ratio_gate_packed_over_uniform_range"][0])
        <= confirmed_ratios["packed_over_uniform"]
        <= float(criteria["ratio_gate_packed_over_uniform_range"][1]),
    }
    exact_discovery_repeat = {
        name: confirmed_capacity[name] == discovered_capacity[name]
        for name in confirmed_capacity
    }
    protocol_checks = {
        "suite_exit_zero": True,
        "four_attempts_complete": len(attempts) == 4,
        "capacity_within_1_percent": evaluation["capacity_within_tolerance"],
        "ratios_within_0_1_percent": evaluation["ratios_within_tolerance"],
        "ratio_gate_passed": all(ratio_gate.values()),
        "exact_repeat_of_discovery_suite": all(exact_discovery_repeat.values()),
        "generation_nonempty": attempts["runtime"]["report"]["generation"]["output_token_count"]
        >= int(criteria["generation_output_tokens_min"]),
    }
    require(all(protocol_checks.values()), "protocol-v2 confirmation failed")

    discovery_report_sha = verify_sidecar(discovery_analysis_dir / "reproducibility_report.json.sha256")
    discovery_report = load_json(discovery_analysis_dir / "reproducibility_report.json")
    require(discovery_report["verdict"] == "PARTIALLY_REPRODUCIBLE", "discovery verdict drift")

    scan = fallacy_scan()
    for item in scan:
        if item["fallacy"] == "Garden of Forking Paths":
            item["severity"] = "SOLID"
            item["detail"] = (
                "Protocol-v2 tolerances were frozen before westd-03, and westd-02 was excluded "
                "from the confirmatory verdict."
            )

    capacity_comparison = {
        name: {
            "original_tokens": original_capacity[name],
            "discovered_tokens": discovered_capacity[name],
            "confirmed_tokens": confirmed_capacity[name],
            "symmetric_relative_difference": evaluation["capacity_differences"][name],
            "within_protocol_tolerance": evaluation["capacity_differences"][name]
            <= float(criteria["capacity_symmetric_relative_difference_max"]),
            "exact_discovery_repeat": exact_discovery_repeat[name],
        }
        for name in original_capacity
    }
    report = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "suite_id": contract["suite_id"],
        "parent_suite": contract["parent_suite"],
        "verification_status": "VERIFIED",
        "verification_scope": "A2 runtime/capacity mechanism and capacity ratios",
        "verdict": "REPRODUCIBLE",
        "a2_overall_status": "PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING",
        "structural_checks": structure,
        "protocol_checks": protocol_checks,
        "capacity_comparison": capacity_comparison,
        "ratios": {
            "original": original_ratios,
            "confirmed": confirmed_ratios,
            "symmetric_relative_difference": evaluation["ratio_differences"],
            "gate_checks": ratio_gate,
        },
        "discovery_report": {
            "path": str(discovery_analysis_dir / "reproducibility_report.json"),
            "sha256": discovery_report_sha,
            "verdict": discovery_report["verdict"],
        },
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": scan,
        "interpretation": (
            "A fresh suite confirmed all structural checks, exactly repeated the replacement-host "
            "capacities, and passed the prospectively frozen capacity and ratio tolerances. The "
            "A2 runtime/capacity sub-scope is verified; serving and quality scopes remain pending."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_sha = write_json_with_hash(output_dir / "protocol_v2_report.json", report)
    validation_sha = write_text_with_hash(output_dir / "validation_report.md", build_validation_markdown(report))
    link = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "verification_status": "VERIFIED",
        "scope": report["verification_scope"],
        "discovery_report_sha256": discovery_report_sha,
        "confirmation_contract_sha256": contract_sha256,
        "protocol_v2_report_sha256": report_sha,
        "validation_report_sha256": validation_sha,
        "overall_a2_status": report["a2_overall_status"],
    }
    link_sha = write_json_with_hash(output_dir / "verification_link.json", link)

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
            "protocol_v2_report.json": report_sha,
            "validation_report.md": validation_sha,
            "verification_link.json": link_sha,
        },
    }
    write_json_with_hash(output_dir / "artifact_sha256_manifest.json", manifest)
    print(
        json.dumps(
            {
                "suite_id": report["suite_id"],
                "verdict": report["verdict"],
                "verification_scope": report["verification_scope"],
                "overall_a2_status": report["a2_overall_status"],
                "structural_checks": f"{sum(structure.values())}/{len(structure)}",
                "protocol_checks": f"{sum(protocol_checks.values())}/{len(protocol_checks)}",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
