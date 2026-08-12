"""Package audited joint-precision calibration evidence for Git archival."""

from __future__ import annotations

import argparse
import json
import re
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

EXPECTED_SAMPLES = 144
EXPECTED_RAW_SIDECARS = 439
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GENERATED_NAMES = {
    "artifact_sha256_manifest.json",
    "artifact_sha256_manifest.json.sha256",
    "raw_source_record.json",
    "raw_source_record.json.sha256",
    "validation_report.md",
    "validation_report.md.sha256",
}


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_sidecar_digest(sidecar: Path) -> str:
    tokens = sidecar.read_text(encoding="ascii").strip().split()
    require(bool(tokens), f"empty SHA sidecar: {sidecar}")
    digest = tokens[0].lower()
    require(SHA256_PATTERN.fullmatch(digest) is not None, f"invalid SHA-256 digest: {sidecar}")
    return digest


def audit_compact_sidecars(artifact_dir: Path) -> dict[str, Any]:
    verified: list[str] = []
    external_raw_results: list[dict[str, str]] = []
    for sidecar in sorted(artifact_dir.rglob("*.sha256")):
        if sidecar.name in GENERATED_NAMES:
            continue
        digest = read_sidecar_digest(sidecar)
        target = sidecar.with_name(sidecar.name.removesuffix(".sha256"))
        if target.is_file():
            require(sha256_file(target) == digest, f"SHA mismatch: {target}")
            verified.append(relative_path(target, artifact_dir))
            continue

        target_relative = relative_path(target, artifact_dir)
        parts = Path(target_relative).parts
        is_raw_result = (
            len(parts) == 4 and parts[0] == "attempt" and parts[1] == "samples" and parts[3] == "result.json"
        )
        require(is_raw_result, f"unexpected missing sidecar target: {target_relative}")
        external_raw_results.append({"relative_path": target_relative, "sha256": digest})

    require(
        len(external_raw_results) == EXPECTED_SAMPLES,
        f"external raw result reference count is not {EXPECTED_SAMPLES}",
    )
    return {
        "sidecars_scanned": len(verified) + len(external_raw_results),
        "local_targets_verified": len(verified),
        "external_raw_result_references": external_raw_results,
    }


def audit_analysis(artifact_dir: Path) -> tuple[dict[str, Any], str]:
    analysis_path = artifact_dir / "calibration_analysis.json"
    digest = verify_sidecar(analysis_path.with_suffix(".json.sha256"))
    analysis = load_json(analysis_path)
    completeness = analysis.get("completeness", {})
    promotion = analysis.get("promotion", {})
    require(analysis.get("gate") == "PASS", "calibration analysis gate did not pass")
    require(analysis.get("evidence_status") == "ANALYZED", "calibration status is not ANALYZED")
    require(completeness.get("expected_samples") == EXPECTED_SAMPLES, "expected sample denominator drift")
    require(completeness.get("audited_samples") == EXPECTED_SAMPLES, "audited sample denominator drift")
    require(completeness.get("request_conservation") is True, "request conservation failed")
    require(completeness.get("silent_exclusions") == 0, "silent exclusions are nonzero")
    require(analysis.get("fallacy_scan", {}).get("coverage") == "11/11", "fallacy scan is incomplete")
    require(promotion.get("confirmatory_run_authorized") is True, "confirmatory run is not authorized")
    require(promotion.get("paper_quantitative_use_authorized") is False, "calibration was promoted to paper evidence")
    return analysis, digest


def audit_profile(artifact_dir: Path) -> tuple[dict[str, Any], str]:
    profile_path = artifact_dir / "calibration_profile.json"
    digest = verify_sidecar(profile_path.with_suffix(".json.sha256"))
    profile = load_json(profile_path)
    require(profile.get("schema_version") == 2, "calibration profile schema is not v2")
    require(profile.get("profile_status") == "CALIBRATION", "profile status is not CALIBRATION")
    candidates = profile.get("candidates")
    require(isinstance(candidates, list) and len(candidates) == 4, "profile must contain four candidates")
    require(
        {row.get("config_id") for row in candidates} == {"full", "kv_only", "state_only", "joint"},
        "profile candidate set drift",
    )
    return profile, digest


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_validation_markdown(analysis: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    completeness = analysis["completeness"]
    aggregation = analysis["aggregation"]
    launch = analysis["launch"]
    fallacies = analysis["fallacy_scan"]["items"]
    rows = "\n".join(
        f"| {markdown_escape(item['fallacy'])} | {markdown_escape(item['severity'])} | "
        f"{markdown_escape(item['detail'])} |"
        for item in fallacies
    )
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: {utc_timestamp()[:10]}
- Verification Status: ANALYZED
- Version Label: joint_precision_calibration_validation_v1

## Validation Report

- **Source**: {analysis["attempt_id"]}
- **Overall Confidence**: CAUTION
- **Gate**: PASS
- **Evidence Status**: ANALYZED

### Integrity

| Check | Result |
|---|---:|
| Launcher exit code | {launch["exit_code"]} |
| Requested / audited samples | {completeness["expected_samples"]} / {completeness["audited_samples"]} |
| Completed / expected requests | {completeness["completed_requests"]} / {completeness["expected_measurement_requests"]} |
| Failed requests | {completeness["failed_requests"]} |
| Silent exclusions | {completeness["silent_exclusions"]} |
| Server sessions | {completeness["server_sessions"]} |
| Aggregated cells | {aggregation["cell_count"]} |
| Profile rows | {aggregation["profile_row_count"]} |
| Raw SHA-256 sidecars verified by analyzer | {source["raw_source"]["sidecars_verified"]} / {source["raw_source"]["sidecars_total"]} |

### Statistical Interpretation

- Unit of analysis: one independent workload trace / seed repeat.
- Each allocation-workload-rate cell uses three repeats and a Student-t 95% confidence interval with df=2.
- Requests are denominator evidence, not independent statistical repeats.
- The 240 pointwise SLO rows are calibration inputs, not confirmatory hypothesis tests.
- P values and multiple-comparison rejection decisions are therefore not used for paper claims at this stage.

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Calibration reuse | Seeds 7, 42, and 2026 construct the controller profile and cannot serve as independent confirmation. | All profile rows |
| Small repeat count | Each cell has n=3 independent repeats; t intervals are valid but imprecise. | CI bounds |
| Deployment scope | Measurements cover one GPU, one model, and the frozen rate grid. | Generalization |
| Evidence boundary | The profile may drive M2 confirmation, but calibration values are not paper-usable quantitative evidence. | Promotion |

### Fallacy Scan

- **Coverage**: {analysis["fallacy_scan"]["coverage"]}

| Fallacy | Severity | Detail |
|---|---|---|
{rows}

### Reproducibility

- **Determinism class**: environment-sensitive seeded serving benchmark
- **Method**: not run for calibration; independent seeds 11, 23, and 47 are reserved for M2 confirmation
- **Verdict**: CANNOT_VERIFY
- **Promotion**: profile construction and confirmatory execution authorized; paper quantitative use not authorized

### Raw Evidence

- Full source: `{source["raw_source"]["path"]}` ({source["raw_source"]["size_bytes"]} bytes)
- Compact archive: `{source["compact_archive"]["path"]}`
- Compact archive SHA-256: `{source["compact_archive"]["sha256"]}`
- Git stores contracts, analyses, statuses, hashes, launch/preflight evidence, and derived profile artifacts; request-level JSON remains on the server data disk.
"""


def manifest_file(path: Path, artifact_dir: Path) -> dict[str, Any]:
    return {
        "relative_path": relative_path(path, artifact_dir),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def package_artifacts(
    artifact_dir: Path, raw_source: Mapping[str, Any], compact_archive: Mapping[str, Any]
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    require(artifact_dir.is_dir(), f"artifact directory is missing: {artifact_dir}")
    for name in GENERATED_NAMES:
        require(not (artifact_dir / name).exists(), f"refusing to overwrite generated artifact: {name}")

    analysis, analysis_sha = audit_analysis(artifact_dir)
    _, profile_sha = audit_profile(artifact_dir)
    sidecars = audit_compact_sidecars(artifact_dir)
    require(raw_source.get("sidecars_total") == EXPECTED_RAW_SIDECARS, "raw sidecar denominator drift")
    require(raw_source.get("sidecars_verified") == EXPECTED_RAW_SIDECARS, "raw sidecars were not fully verified")
    require(raw_source.get("result_count") == EXPECTED_SAMPLES, "raw result count drift")
    compact_digest = str(compact_archive.get("sha256", "")).lower()
    require(SHA256_PATTERN.fullmatch(compact_digest) is not None, "invalid compact archive SHA-256")

    source_record = {
        "schema_version": 1,
        "material_passport": analysis["material_passport"],
        "attempt_id": analysis["attempt_id"],
        "raw_source": dict(raw_source),
        "compact_archive": {**compact_archive, "sha256": compact_digest},
        "git_archive": {
            "local_sidecar_targets_verified": sidecars["local_targets_verified"],
            "external_raw_result_references": len(sidecars["external_raw_result_references"]),
            "excluded_from_git": ["request-level result.json", "*.log", "*.tar.gz"],
        },
    }
    source_sha = write_json_with_hash(artifact_dir / "raw_source_record.json", source_record)
    validation_sha = write_text_with_hash(
        artifact_dir / "validation_report.md",
        build_validation_markdown(analysis, source_record),
    )

    manifest_paths = sorted(
        path
        for path in artifact_dir.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_sha256_manifest.json", "artifact_sha256_manifest.json.sha256"}
        and path.suffix != ".log"
        and not path.name.endswith(".tar.gz")
    )
    manifest = {
        "schema_version": 1,
        "material_passport": analysis["material_passport"],
        "generated_at_utc": utc_timestamp(),
        "attempt_id": analysis["attempt_id"],
        "scope": "Git-trackable compact calibration evidence; request-level JSON remains on the data disk",
        "files": [manifest_file(path, artifact_dir) for path in manifest_paths],
        "external_raw_results": sidecars["external_raw_result_references"],
        "generated": {
            "calibration_analysis.json": analysis_sha,
            "calibration_profile.json": profile_sha,
            "raw_source_record.json": source_sha,
            "validation_report.md": validation_sha,
        },
    }
    manifest_sha = write_json_with_hash(artifact_dir / "artifact_sha256_manifest.json", manifest)
    return {
        "status": "PACKAGED",
        "attempt_id": analysis["attempt_id"],
        "manifest_files": len(manifest_paths),
        "manifest_sha256": manifest_sha,
        "local_sidecars_verified": sidecars["local_targets_verified"],
        "external_raw_result_references": len(sidecars["external_raw_result_references"]),
        "evidence_status": analysis["evidence_status"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--raw-source-path", required=True)
    parser.add_argument("--raw-source-size-bytes", type=int, required=True)
    parser.add_argument("--raw-source-file-count", type=int, required=True)
    parser.add_argument("--raw-result-count", type=int, required=True)
    parser.add_argument("--raw-result-size-bytes", type=int, required=True)
    parser.add_argument("--raw-sidecars-total", type=int, required=True)
    parser.add_argument("--raw-sidecars-verified", type=int, required=True)
    parser.add_argument("--compact-archive-path", required=True)
    parser.add_argument("--compact-archive-size-bytes", type=int, required=True)
    parser.add_argument("--compact-archive-file-count", type=int, required=True)
    parser.add_argument("--compact-archive-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = package_artifacts(
        args.artifact_dir,
        {
            "path": args.raw_source_path,
            "storage": "server_data_disk_source_of_truth",
            "size_bytes": args.raw_source_size_bytes,
            "file_count": args.raw_source_file_count,
            "result_count": args.raw_result_count,
            "result_size_bytes": args.raw_result_size_bytes,
            "sidecars_total": args.raw_sidecars_total,
            "sidecars_verified": args.raw_sidecars_verified,
        },
        {
            "path": args.compact_archive_path,
            "size_bytes": args.compact_archive_size_bytes,
            "file_count": args.compact_archive_file_count,
            "sha256": args.compact_archive_sha256,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"ERROR: {error}")
        raise SystemExit(2)
