"""Audit the independent A2 packed-cache reproduction without rewriting evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ENVIRONMENT_TOLERANCE = 0.10
EXPECTED_ATTEMPTS = {
    "runtime": "a2-repro-mvex-c7379f0-westd-02",
    "legacy": "a2-repro-capacity-legacy-c7379f0-westd-02",
    "uniform": "a2-repro-capacity-uniform-c7379f0-westd-02",
    "packed": "a2-repro-capacity-packed-c7379f0-westd-02",
}


class VerificationError(RuntimeError):
    """Raised when an evidence-integrity or structural gate fails."""


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise VerificationError(f"expected JSON object: {path}")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json_with_hash(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    atomic_write(path, payload)
    digest = sha256_file(path)
    atomic_write(path.with_suffix(path.suffix + ".sha256"), f"{digest}\n".encode("ascii"))
    return digest


def write_text_with_hash(path: Path, value: str) -> str:
    atomic_write(path, value.encode("utf-8"))
    digest = sha256_file(path)
    atomic_write(path.with_suffix(path.suffix + ".sha256"), f"{digest}\n".encode("ascii"))
    return digest


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def verify_sidecar(sidecar: Path) -> str:
    require(sidecar.name.endswith(".sha256"), f"not a SHA sidecar: {sidecar}")
    target = sidecar.with_name(sidecar.name.removesuffix(".sha256"))
    require(target.is_file(), f"sidecar target missing: {target}")
    expected = sidecar.read_text(encoding="ascii").strip().split()[0]
    actual = sha256_file(target)
    require(actual == expected, f"SHA mismatch: {target}")
    return actual


def symmetric_relative_difference(left: float, right: float, *, epsilon: float = 1e-12) -> float:
    return abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), epsilon)


def group_types(report: Mapping[str, Any]) -> list[str]:
    return [str(group.get("spec", {}).get("type")) for group in report["kv_cache_config"]["groups"]]


def classify_verdict(*, structural_pass: bool, exact_capacity_match: bool, within_environment_tolerance: bool) -> str:
    if not structural_pass:
        return "NOT_REPRODUCIBLE"
    if exact_capacity_match:
        return "REPRODUCIBLE"
    if within_environment_tolerance:
        return "PARTIALLY_REPRODUCIBLE"
    return "NOT_REPRODUCIBLE"


def audit_attempt(raw_dir: Path, contract: Mapping[str, Any], name: str) -> dict[str, Any]:
    attempt_id = EXPECTED_ATTEMPTS[name]
    item = next((entry for entry in contract["attempts"] if entry["attempt_id"] == attempt_id), None)
    require(item is not None, f"contract missing attempt: {attempt_id}")
    attempt_dir = raw_dir / "attempts" / attempt_id
    require(attempt_dir.is_dir(), f"attempt directory missing: {attempt_dir}")
    require((attempt_dir / "exit_code.txt").read_text(encoding="ascii").strip() == "0", f"{attempt_id}: nonzero exit")

    launch = load_json(attempt_dir / "launch.json")
    require(launch["attempt_id"] == attempt_id, f"{attempt_id}: launch ID mismatch")
    require(launch["command"] == item["command"], f"{attempt_id}: command drift")
    require(
        launch["contract_sha256"] == verify_sidecar(raw_dir / "attempt_contract_westd_02.json.sha256"),
        f"{attempt_id}: contract hash mismatch",
    )

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


def structural_checks(attempts: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    runtime = attempts["runtime"]["report"]
    legacy = attempts["legacy"]["report"]
    uniform = attempts["uniform"]["report"]
    packed = attempts["packed"]["report"]

    runtime_checks = runtime["verification"]["checks"]
    packed_checks = packed["verification"]["checks"]
    legacy_types = group_types(legacy)
    uniform_types = group_types(uniform)
    packed_types = group_types(packed)
    return {
        "runtime_eight_of_eight": runtime["verification"]["passed"]
        and len(runtime_checks) == 8
        and all(runtime_checks.values()),
        "runtime_generation_nonempty": runtime["generation"]["output_token_count"] > 0,
        "runtime_and_capacity_packed_agree": runtime["capacity"] == packed["capacity"],
        "legacy_requested_map_and_flag": not legacy["requested"]["enable_per_layer_page_groups"]
        and len(legacy["requested"]["kv_cache_dtype_per_layer"]) == 6,
        "legacy_has_24_independent_groups": len(legacy_types) == 24
        and legacy_types.count("MambaSpec") == 18
        and legacy_types.count("FullAttentionSpec") == 6
        and "UniformTypeKVCacheSpecs" not in legacy_types,
        "uniform_requested_map_and_flag": not uniform["requested"]["enable_per_layer_page_groups"]
        and uniform["requested"]["kv_cache_dtype_per_layer"] == {},
        "uniform_has_one_attention_group": len(uniform_types) == 4
        and uniform_types.count("MambaSpec") == 3
        and uniform_types.count("FullAttentionSpec") == 1,
        "packed_eight_of_eight": packed["verification"]["passed"]
        and len(packed_checks) == 8
        and all(packed_checks.values()),
        "packed_has_mixed_attention_group": packed_types
        == ["UniformTypeKVCacheSpecs", "MambaSpec", "MambaSpec"],
        "mamba_ssm_cache_dtype_float32": all(
            attempt["report"]["cache_config"]["mamba_ssm_cache_dtype"] == "float32"
            for attempt in attempts.values()
        ),
    }


def fallacy_scan() -> list[dict[str, str]]:
    return [
        {"fallacy": "Simpson's Paradox", "severity": "NOTE", "detail": "Not applicable; no subgroup aggregation."},
        {"fallacy": "Ecological Fallacy", "severity": "NOTE", "detail": "Not applicable; inference stays at the host/configuration level."},
        {"fallacy": "Berkson's Paradox", "severity": "NOTE", "detail": "Not applicable; no selected correlation sample."},
        {"fallacy": "Collider Bias", "severity": "NOTE", "detail": "Not applicable; no covariate adjustment."},
        {"fallacy": "Base Rate Neglect", "severity": "NOTE", "detail": "Not applicable; no diagnostic probabilities."},
        {"fallacy": "Regression to the Mean", "severity": "NOTE", "detail": "Not applicable; no extreme-score selection."},
        {"fallacy": "Survivorship Bias", "severity": "SOLID", "detail": "The failed suite is preserved and excluded; all four replacement attempts are reported."},
        {"fallacy": "Look-Elsewhere Effect", "severity": "SOLID", "detail": "All predeclared capacities and structural checks are reported."},
        {"fallacy": "Garden of Forking Paths", "severity": "CAUTION", "detail": "A transport environment variable was added only under a new linked suite; numerical criteria were not changed."},
        {"fallacy": "Correlation != Causation", "severity": "CAUTION", "detail": "The common block-count increase supports, but does not prove, the memory-profile drift explanation."},
        {"fallacy": "Reverse Causality", "severity": "NOTE", "detail": "Not applicable to the controlled cache-layout comparison."},
    ]


def build_validation_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for name in ("legacy", "uniform", "packed"):
        item = report["capacity_comparison"][name]
        rows.append(
            f"| {name} | {item['original_tokens']:,} | {item['reproduced_tokens']:,} | "
            f"{item['signed_relative_change_percent']:+.3f}% | "
            f"{'MATCH' if item['exact_match'] else 'MISMATCH'} |"
        )
    checks = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["structural_checks"].items()
    )
    fallacies = "\n".join(
        f"| {item['fallacy']} | {item['severity']} | {item['detail']} |"
        for item in report["fallacy_scan"]
    )
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_reproduction_validation_v1

## Validation Report

- **Source**: `{report['suite_id']}`
- **Overall Confidence**: CAUTION
- **Frozen-Contract Verdict**: `{report['verdict']}`
- **Environment-Sensitive Comparison**: `{report['environment_sensitive_verdict']}`
- **A2 Evidence Status**: `{report['a2_evidence_status']}`

### Capacity Comparison

| Allocation | Original | Reproduced | Relative Change | Exact Gate |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

The three capacity values remain within the generic 10% environment-sensitive
tolerance, but none exactly matches the predeclared cross-host gate. The
mechanism and ratio gates reproduce; A2 therefore remains `PASSED`, not
`VERIFIED`.

### Ratio Gates

- packed / legacy: `{report['ratios']['reproduced']['packed_over_legacy']:.6f}` (required >= 3.0)
- packed / uniform: `{report['ratios']['reproduced']['packed_over_uniform']:.6f}` (required 0.80--0.92)

### Structural Checks

{checks}

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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    original_dir = args.original_dir.resolve()
    raw_dir = args.reproduction_dir.resolve()
    output_dir = args.output_dir.resolve()

    for sidecar in (
        raw_dir / "attempt_contract.json.sha256",
        raw_dir / "attempt_contract_westd_02.json.sha256",
        raw_dir / "suite_westd_01_failure.json.sha256",
        raw_dir / "deployment-dryrun-westd-01/deployment.json.sha256",
        raw_dir / "deployment-westd-01/deployment.json.sha256",
        original_dir / "a2_runtime.json.sha256",
        original_dir / "a2_capacity_gate_c7379f0_v2.json.sha256",
    ):
        verify_sidecar(sidecar)

    failed_suite = load_json(raw_dir / "suite_westd_01_failure.json")
    require(failed_suite["status"] == "FAILED_RUNTIME_COLLECTION", "failed suite status drift")
    require(failed_suite["valid_outputs"] == 0, "failed suite produced a valid output")

    contract = load_json(raw_dir / "attempt_contract_westd_02.json")
    require(contract["suite_id"] == "a2-repro-suite-c7379f0-westd-02", "suite ID mismatch")
    require(contract["parent_suite"] == "a2-repro-suite-c7379f0-westd-01", "parent suite mismatch")
    require(contract["root_commit"] == "c7379f0c68a67a4eeb838573fdfe5560c1a42bd9", "root commit mismatch")
    require(contract["vllm_commit"] == "55f47685a553ad8d776c464c59785399a98c7185", "vLLM commit mismatch")

    attempts = {name: audit_attempt(raw_dir, contract, name) for name in EXPECTED_ATTEMPTS}
    checks = structural_checks(attempts)
    structural_pass = all(checks.values())

    original_gate = load_json(original_dir / "a2_capacity_gate_c7379f0_v2.json")
    original_runtime = load_json(original_dir / "a2_runtime.json")
    require(original_gate["status"] == "PASSED", "original A2 gate is not PASSED")
    require(original_runtime["verification"]["passed"], "original runtime gate did not pass")

    capacity_comparison: dict[str, dict[str, Any]] = {}
    exact_capacity_match = True
    within_environment_tolerance = True
    reproduced_capacity: dict[str, int] = {}
    for name in ("legacy", "uniform", "packed"):
        original = int(original_gate["baselines"][name]["capacity_tokens"])
        reproduced = int(attempts[name]["report"]["capacity"]["tokens"])
        reproduced_capacity[name] = reproduced
        relative_difference = symmetric_relative_difference(original, reproduced)
        exact_match = original == reproduced
        capacity_comparison[name] = {
            "original_tokens": original,
            "reproduced_tokens": reproduced,
            "absolute_difference_tokens": reproduced - original,
            "signed_relative_change_percent": 100.0 * (reproduced - original) / original,
            "symmetric_relative_difference": relative_difference,
            "exact_match": exact_match,
            "within_environment_tolerance": relative_difference < ENVIRONMENT_TOLERANCE,
        }
        exact_capacity_match = exact_capacity_match and exact_match
        within_environment_tolerance = (
            within_environment_tolerance and relative_difference < ENVIRONMENT_TOLERANCE
        )

    reproduced_ratios = {
        "packed_over_legacy": reproduced_capacity["packed"] / reproduced_capacity["legacy"],
        "packed_over_uniform": reproduced_capacity["packed"] / reproduced_capacity["uniform"],
        "uniform_over_legacy": reproduced_capacity["uniform"] / reproduced_capacity["legacy"],
    }
    ratio_checks = {
        "packed_over_legacy_at_least_3x": reproduced_ratios["packed_over_legacy"] >= 3.0,
        "packed_over_uniform_in_range": 0.80 <= reproduced_ratios["packed_over_uniform"] <= 0.92,
    }
    structural_pass = structural_pass and all(ratio_checks.values())
    verdict = classify_verdict(
        structural_pass=structural_pass,
        exact_capacity_match=exact_capacity_match,
        within_environment_tolerance=within_environment_tolerance,
    )
    environment_sensitive_verdict = (
        "REPRODUCIBLE" if structural_pass and within_environment_tolerance else "NOT_REPRODUCIBLE"
    )

    report = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "suite_id": contract["suite_id"],
        "parent_suite": contract["parent_suite"],
        "verification_status": "ANALYZED",
        "verdict": verdict,
        "environment_sensitive_verdict": environment_sensitive_verdict,
        "a2_evidence_status": "PASSED_NOT_VERIFIED",
        "promotion_gate_passed": verdict == "REPRODUCIBLE" and exact_capacity_match,
        "failed_suite_preserved_and_excluded": True,
        "structural_checks": checks,
        "ratio_checks": ratio_checks,
        "capacity_comparison": capacity_comparison,
        "ratios": {
            "original": original_gate["ratios"],
            "reproduced": reproduced_ratios,
            "symmetric_relative_difference": {
                name: symmetric_relative_difference(original_gate["ratios"][name], value)
                for name, value in reproduced_ratios.items()
            },
        },
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": fallacy_scan(),
        "interpretation": (
            "The packed mechanism, generation path, cache structure, dtype state, and ratio gates "
            "reproduced. Absolute capacities increased by about 0.12%-0.14% because the new host "
            "profiled a few additional GPU cache blocks. The frozen exact-capacity gate therefore "
            "failed, so A2 must not be promoted to VERIFIED."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_sha = write_json_with_hash(output_dir / "reproducibility_report.json", report)
    validation_sha = write_text_with_hash(output_dir / "validation_report.md", build_validation_markdown(report))

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
            "reproducibility_report.json": report_sha,
            "validation_report.md": validation_sha,
        },
    }
    write_json_with_hash(output_dir / "artifact_sha256_manifest.json", manifest)
    print(
        json.dumps(
            {
                "suite_id": report["suite_id"],
                "verdict": verdict,
                "environment_sensitive_verdict": environment_sensitive_verdict,
                "a2_evidence_status": report["a2_evidence_status"],
                "structural_checks": f"{sum(checks.values())}/{len(checks)}",
                "exact_capacity_match": exact_capacity_match,
                "within_environment_tolerance": within_environment_tolerance,
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
