from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analyze.package_joint_precision_calibration import package_artifacts
from scripts.analyze.verify_a2_reproduction import VerificationError, load_json, write_json_with_hash


def analysis() -> dict:
    fallacies = [{"fallacy": f"fallacy-{index}", "severity": "NOTE", "detail": "checked"} for index in range(11)]
    return {
        "material_passport": {"verification_status": "ANALYZED"},
        "gate": "PASS",
        "evidence_status": "ANALYZED",
        "attempt_id": "calibration-test",
        "completeness": {
            "expected_samples": 144,
            "audited_samples": 144,
            "expected_measurement_requests": 320400,
            "completed_requests": 320400,
            "failed_requests": 0,
            "request_conservation": True,
            "server_sessions": 4,
            "silent_exclusions": 0,
        },
        "launch": {"exit_code": 0},
        "aggregation": {"cell_count": 48, "profile_row_count": 240},
        "fallacy_scan": {"coverage": "11/11", "items": fallacies},
        "promotion": {
            "confirmatory_run_authorized": True,
            "paper_quantitative_use_authorized": False,
        },
    }


def profile() -> dict:
    return {
        "schema_version": 2,
        "profile_status": "CALIBRATION",
        "candidates": [{"config_id": config_id} for config_id in ("full", "kv_only", "state_only", "joint")],
    }


def fixture_dir(tmp_path: Path) -> Path:
    write_json_with_hash(tmp_path / "calibration_analysis.json", analysis())
    write_json_with_hash(tmp_path / "calibration_profile.json", profile())
    for index in range(144):
        sample = tmp_path / "attempt" / "samples" / f"sample-{index}"
        sample.mkdir(parents=True)
        (sample / "result.json.sha256").write_text(f"{'a' * 64}\n", encoding="ascii")
    return tmp_path


def raw_source() -> dict:
    return {
        "path": "/data/calibration",
        "storage": "server_data_disk_source_of_truth",
        "size_bytes": 1000,
        "file_count": 1175,
        "result_count": 144,
        "result_size_bytes": 900,
        "sidecars_total": 439,
        "sidecars_verified": 439,
    }


def compact_archive() -> dict:
    return {
        "path": "/data/calibration/audit/compact.tar.gz",
        "size_bytes": 100,
        "file_count": 1052,
        "sha256": "b" * 64,
    }


def test_package_builds_report_source_record_and_manifest(tmp_path: Path) -> None:
    artifact_dir = fixture_dir(tmp_path)

    result = package_artifacts(artifact_dir, raw_source(), compact_archive())

    assert result["status"] == "PACKAGED"
    assert result["external_raw_result_references"] == 144
    assert result["evidence_status"] == "ANALYZED"
    assert (artifact_dir / "validation_report.md").is_file()
    assert (artifact_dir / "raw_source_record.json.sha256").is_file()
    manifest = load_json(artifact_dir / "artifact_sha256_manifest.json")
    assert len(manifest["external_raw_results"]) == 144
    assert all(not item["relative_path"].endswith(".tar.gz") for item in manifest["files"])


def test_package_rejects_unexpected_missing_sidecar_target(tmp_path: Path) -> None:
    artifact_dir = fixture_dir(tmp_path)
    (artifact_dir / "unexpected.json.sha256").write_text(f"{'c' * 64}\n", encoding="ascii")

    with pytest.raises(VerificationError, match="unexpected missing sidecar target"):
        package_artifacts(artifact_dir, raw_source(), compact_archive())
