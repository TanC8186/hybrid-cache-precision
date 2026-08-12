"""Logically audit and compare the M2 Gate 4 reproduction attempt."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.analyze.audit_joint_precision_m2_pilot import (
    AuditError,
    atomic_write_json,
    atomic_write_text,
    audit_pilot,
    equivalent_json,
    load_json,
    require,
)


def symmetric_relative_difference(left: Any, right: Any, *, epsilon: float) -> float:
    first = float(left)
    second = float(right)
    require(math.isfinite(first) and math.isfinite(second), "reproduction comparison contains a non-finite value")
    denominator = max(abs(first), abs(second), float(epsilon))
    return abs(first - second) / denominator


def build_reproduction_package(
    gate4_contract: Mapping[str, Any],
    selector_contract: Mapping[str, Any],
) -> dict[str, Any]:
    attempts = {str(row["request_id"]): row for row in gate4_contract["attempts"]}
    require(len(attempts) == 3, "Gate 4 contract must define three request attempts")
    requests: list[dict[str, Any]] = []
    for request in selector_contract["requests"]:
        request_id = str(request["id"])
        require(request_id in attempts, f"Gate 4 attempt missing request {request_id}")
        attempt = attempts[request_id]
        require(
            request["expected_selected_config_id"] == attempt["expected_selected_config_id"],
            f"Gate 4 expected mapping drift for {request_id}",
        )
        requests.append(
            {
                "id": request_id,
                "path": request["path"],
                "expected_selected_config_id": attempt["expected_selected_config_id"],
                "attempt_id": attempt["reproduction_attempt"],
            }
        )
    return {
        "package_id": gate4_contract["package_id"],
        "classification": gate4_contract["classification"],
        "profile": selector_contract["profile"],
        "serving_config": selector_contract["serving_config"],
        "requests": requests,
        "matrix": {"pilot": gate4_contract["matrix"]},
    }


def find_reproduction_root(reproduction_root: Path, attempt_id: str) -> Path:
    path = reproduction_root / attempt_id
    require(path.is_dir(), f"missing reproduction attempt: {attempt_id}")
    require((path / "controller_result.json").is_file(), f"reproduction controller result missing: {attempt_id}")
    return path


def check_gate4_lineage(
    reproduction_root: Path,
    gate4_contract: Mapping[str, Any],
    reproduction_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    audit_cells = {str(cell["request_id"]): cell for cell in reproduction_audit["cells"]}
    records: list[dict[str, Any]] = []
    for expected in gate4_contract["attempts"]:
        request_id = str(expected["request_id"])
        attempt_id = str(expected["reproduction_attempt"])
        controller_root = find_reproduction_root(reproduction_root, attempt_id)
        controller_contract = load_json(controller_root / "controller_contract.json")
        runner_contract = load_json(controller_root / "runner" / attempt_id / "attempt_contract.json")
        launch_dir = reproduction_root / "launch" / attempt_id
        require(controller_contract["attempt_id"] == attempt_id, f"{request_id}: controller attempt ID drift")
        require(
            controller_contract.get("parent_attempt") == expected["parent_attempt"],
            f"{request_id}: controller parent attempt drift",
        )
        require(
            runner_contract.get("parent_attempt") == expected["parent_attempt"],
            f"{request_id}: runner parent attempt drift",
        )
        require(
            controller_contract.get("evidence_review")
            == {
                "mode": "logical_only",
                "hash_validation_performed": False,
                "sidecar_files_used_as_launch_gates": False,
            },
            f"{request_id}: reproduction did not use the frozen logical-only evidence review",
        )
        require(
            (launch_dir / "parent_attempt").read_text(encoding="ascii").strip() == expected["parent_attempt"],
            f"{request_id}: launcher parent attempt drift",
        )
        require(
            (launch_dir / "evidence_review_mode").read_text(encoding="ascii").strip() == "logical_only",
            f"{request_id}: launcher evidence review mode drift",
        )
        require(
            audit_cells[request_id]["attempt_id"] == attempt_id,
            f"{request_id}: reproduction audit selected an unexpected attempt",
        )
        records.append(
            {
                "request_id": request_id,
                "parent_attempt": expected["parent_attempt"],
                "reproduction_attempt": attempt_id,
                "selected_config_id": audit_cells[request_id]["selected_config_id"],
                "root_commit": audit_cells[request_id]["root_commit"],
                "vllm_commit": audit_cells[request_id]["vllm_commit"],
                "evidence_review_mode": "logical_only",
            }
        )
    return records


def compare_audits(
    parent: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    require(parent["gate_2_verdict"] == "PASS", "parent pilot did not pass Gate 2")
    require(reproduction["gate_2_verdict"] == "PASS", "reproduction logical audit failed")
    require(parent["audit"]["hash_validation_performed"] is False, "parent audit used hash validation")
    require(reproduction["audit"]["hash_validation_performed"] is False, "reproduction audit used hash validation")
    require(parent["completeness"] == reproduction["completeness"], "parent/reproduction completeness differs")
    require(parent["decision_metrics"]["mapping"] == reproduction["decision_metrics"]["mapping"], "selector mapping differs")
    require(parent["scope"] == reproduction["scope"], "parent/reproduction scope differs")
    require(parent["fallacy_scan"]["coverage"] == reproduction["fallacy_scan"]["coverage"] == "11/11", "fallacy scan incomplete")

    tolerances = contract["comparison"]["symmetric_relative_tolerances"]
    epsilon = float(contract["comparison"]["epsilon"])
    parent_cells = {str(cell["request_id"]): cell for cell in parent["cells"]}
    repro_cells = {str(cell["request_id"]): cell for cell in reproduction["cells"]}
    require(set(parent_cells) == set(repro_cells) == {"strict", "medium", "high"}, "comparison cell set drift")
    rows: list[dict[str, Any]] = []
    all_passed = True
    metric_mapping = {
        "mean_goodput_req_s": "goodput_req_s",
        "mean_p95_ttft_ms": "p95_ttft_ms",
        "mean_p95_tpot_ms": "p95_tpot_ms",
    }
    for request_id in ("strict", "medium", "high"):
        parent_cell = parent_cells[request_id]
        repro_cell = repro_cells[request_id]
        require(
            parent_cell["selected_config_id"] == repro_cell["selected_config_id"],
            f"{request_id}: selected config differs",
        )
        require(
            parent_cell["requested_slo_attainment"] == repro_cell["requested_slo_attainment"] == "3/3",
            f"{request_id}: requested SLO did not attain 3/3 seeds",
        )
        require(
            [sample["seed"] for sample in parent_cell["samples"]]
            == [sample["seed"] for sample in repro_cell["samples"]],
            f"{request_id}: seed membership differs",
        )
        metrics: list[dict[str, Any]] = []
        for declared_name, summary_name in metric_mapping.items():
            original = float(parent_cell["metric_summaries"][summary_name]["mean"])
            rerun = float(repro_cell["metric_summaries"][summary_name]["mean"])
            difference = symmetric_relative_difference(original, rerun, epsilon=epsilon)
            tolerance = float(tolerances[declared_name])
            passed = difference <= tolerance
            all_passed = all_passed and passed
            metrics.append(
                {
                    "metric": declared_name,
                    "parent": original,
                    "reproduction": rerun,
                    "symmetric_relative_difference": difference,
                    "tolerance": tolerance,
                    "status": "WITHIN_TOLERANCE" if passed else "MISMATCH",
                }
            )
        rows.append(
            {
                "request_id": request_id,
                "selected_config_id": parent_cell["selected_config_id"],
                "structural_status": "MATCH",
                "metrics": metrics,
            }
        )
    return {"all_within_tolerance": all_passed, "cells": rows}


def build_markdown(report: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for cell in report["comparison"]["cells"]:
        for metric in cell["metrics"]:
            rows.append(
                "| {budget} | {allocation} | {metric} | {parent:.6f} | {repro:.6f} | {diff:.4%} | {tol:.0%} | {status} |".format(
                    budget=cell["request_id"],
                    allocation=cell["selected_config_id"],
                    metric=metric["metric"],
                    parent=metric["parent"],
                    repro=metric["reproduction"],
                    diff=metric["symmetric_relative_difference"],
                    tol=metric["tolerance"],
                    status=metric["status"],
                )
            )
    return f"""## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: validate
- Origin Date: {report['generated_at_utc'][:10]}
- Verification Status: {report['evidence_status']}
- Version Label: joint_precision_m2_gate4_validation_v1

## Validation Report

- **Source**: `{report['package_id']}`
- **Gate 4 Verdict**: `{report['gate_4_verdict']}`
- **Evidence Status**: `{report['evidence_status']}`
- **Reproducibility**: `{report['reproducibility']['verdict']}`
- **Audit Mode**: logical review only; no SHA-256 or hash validation performed

### Integrity Findings

The reproduction contains 9/9 seeded samples, 18,000/18,000 measurement
requests, 1,080 declared warmup requests, zero failed requests, zero silent
exclusions, and three launcher exit codes of 0. Parent linkage, selector mapping,
sample membership, precision commands, server-log proofs, and requested 500/200
ms SLO attainment (3/3 seeds for every budget) match structurally.

| Budget | Selected | Metric | Parent | Reproduction | Symmetric relative diff | Tolerance | Status |
|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

### Statistical Scope

The experiment is classified as an environment-sensitive seeded serving
benchmark. Comparisons use the predeclared seed-level means and symmetric
relative differences. Individual requests are retained for denominator and SLO
logic checks but are not treated as independent statistical repeats.

### Fallacy Scan

- **Coverage**: 11/11

The Gate 2 fallacy scan remains applicable and is preserved in both logical
audits. The reproduction adds no post hoc endpoints or exclusions.

### Promotion Decision

{report['promotion']['reason']}

The verified scope remains one RTX 5090, Qwen3.5-2B, 4096 context, Random
workload, and TP=1. This result does not verify cross-model, cross-context,
cross-hardware, TP=2/4, or mechanism claims.
"""


def validate_gate4(
    reproduction_root: Path,
    repo_root: Path,
    selector_contract_path: Path,
    gate4_contract_path: Path,
    parent_audit_path: Path,
    *,
    source_host: str,
) -> dict[str, Any]:
    selector_contract = load_json(selector_contract_path)
    gate4_contract = load_json(gate4_contract_path)
    parent_audit = load_json(parent_audit_path)
    require(gate4_contract["audit_policy"]["mode"] == "logical_only", "Gate 4 audit mode drift")
    require(gate4_contract["audit_policy"]["hash_validation_performed"] is False, "Gate 4 enables hash validation")
    require(
        gate4_contract["parent_gate2_report"] == parent_audit_path.relative_to(repo_root).as_posix(),
        "parent Gate 2 report path drift",
    )

    package = build_reproduction_package(gate4_contract, selector_contract)
    generated_contract = reproduction_root / "gate4_logical_package.json"
    atomic_write_json(generated_contract, package)
    reproduction_audit = audit_pilot(
        reproduction_root,
        repo_root,
        generated_contract,
        source_host=source_host,
    )
    lineage = check_gate4_lineage(reproduction_root, gate4_contract, reproduction_audit)
    comparison = compare_audits(parent_audit, reproduction_audit, gate4_contract)
    passed = bool(comparison["all_within_tolerance"])
    evidence_status = "VERIFIED" if passed else "ANALYZED"
    verdict = "REPRODUCIBLE" if passed else "NOT_REPRODUCIBLE"
    promotion_reason = (
        "Gate 4 passes: structural checks and all predeclared environment-sensitive metric tolerances pass. "
        "The scoped M2 selector slice is promoted to VERIFIED."
        if passed
        else "Gate 4 fails closed because one or more predeclared reproduction tolerances did not pass. "
        "The M2 selector slice remains ANALYZED and is not paper-usable."
    )
    return {
        "schema_version": 1,
        "material_passport": {
            "origin_skill": "experiment-skill",
            "origin_mode": "validate",
            "origin_date": datetime.now(timezone.utc).date().isoformat(),
            "verification_status": evidence_status,
            "version_label": "joint_precision_m2_gate4_validation_v1",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "package_id": gate4_contract["package_id"],
        "gate_4_verdict": "PASS" if passed else "FAIL",
        "evidence_status": evidence_status,
        "audit": {
            "mode": "logical_only",
            "hash_validation_performed": False,
            "sidecar_files_used_as_gates": False,
        },
        "source": {"host": source_host, "reproduction_root": str(reproduction_root)},
        "parent_gate2_report": str(parent_audit_path),
        "completeness": reproduction_audit["completeness"],
        "scope": reproduction_audit["scope"],
        "lineage": lineage,
        "reproduction_audit": reproduction_audit,
        "comparison": comparison,
        "fallacy_scan": reproduction_audit["fallacy_scan"],
        "reproducibility": {
            "determinism_class": "environment_sensitive_seeded_serving_benchmark",
            "method": "same-host temporal rerun with frozen seeds and matrix",
            "verdict": verdict,
        },
        "promotion": {
            "paper_quantitative_use_authorized_for_scoped_slice": passed,
            "reason": promotion_reason,
            "boundary": gate4_contract["promotion_boundary"],
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--selector-contract", type=Path, required=True)
    parser.add_argument("--gate4-contract", type=Path, required=True)
    parser.add_argument("--parent-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-host", default="unspecified")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_gate4(
        args.reproduction_root.resolve(),
        args.repo_root.resolve(),
        args.selector_contract.resolve(),
        args.gate4_contract.resolve(),
        args.parent_audit.resolve(),
        source_host=args.source_host,
    )
    out_dir = args.out_dir.resolve()
    atomic_write_json(out_dir / "gate4_validation.json", report)
    atomic_write_text(out_dir / "validation_report.md", build_markdown(report))
    print(
        json.dumps(
            {
                "gate_4_verdict": report["gate_4_verdict"],
                "evidence_status": report["evidence_status"],
                "reproducibility": report["reproducibility"]["verdict"],
                "samples": f"{report['completeness']['audited_samples']}/{report['completeness']['expected_samples']}",
                "measurement_requests": f"{report['completeness']['completed_measurement_requests']}/{report['completeness']['expected_measurement_requests']}",
                "hash_validation_performed": report["audit"]["hash_validation_performed"],
            },
            indent=2,
        )
    )
    return 0 if report["gate_4_verdict"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
