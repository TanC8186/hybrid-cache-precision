from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "quality" / "analyze_m4_formal.py"
_SPEC = importlib.util.spec_from_file_location("analyze_m4_formal", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
bh_adjust = _MODULE.bh_adjust
paired_summary = _MODULE.paired_summary
audit_launch = _MODULE.audit_launch
audit_server_sessions = _MODULE.audit_server_sessions
require_no_partial_artifacts = _MODULE.require_no_partial_artifacts
AuditError = _MODULE.AuditError
PRECISION_PROFILES = _MODULE.PRECISION_PROFILES


def test_paired_summary_constant_zero_is_finite() -> None:
    summary = paired_summary([0.0, 0.0, 0.0])

    assert summary["mean"] == 0.0
    assert summary["sd"] == 0.0
    assert summary["ci95_low"] == 0.0
    assert summary["ci95_high"] == 0.0
    assert summary["p_value"] == 1.0
    assert summary["cohen_dz"] is None


def test_paired_summary_constant_nonzero_is_finite() -> None:
    summary = paired_summary([2.0, 2.0, 2.0])

    assert summary["mean"] == 2.0
    assert summary["sd"] == 0.0
    assert summary["p_value"] == 0.0
    assert summary["cohen_dz"] is None
    assert all(math.isfinite(summary[key]) for key in ("ci95_low", "ci95_high"))


def test_bh_adjust_accepts_degenerate_test_results() -> None:
    adjusted = bh_adjust([1.0, 0.0, 0.5])

    assert adjusted == [1.0, 0.0, 0.75]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def make_server_sessions(attempt_dir: Path) -> None:
    for allocation, (kv_dtype, state_dtype) in PRECISION_PROFILES.items():
        session_dir = attempt_dir / "servers" / allocation / f"session-{allocation}"
        session_dir.mkdir(parents=True)
        command = [
            "vllm",
            "serve",
            "model",
            "--kv-cache-dtype",
            kv_dtype,
            "--mamba-ssm-cache-dtype",
            state_dtype,
        ]
        write_json(
            session_dir / "contract.json",
            {"allocation": allocation, "command": command, "started_at": "2026-08-13T01:00:00Z"},
        )
        write_json(
            session_dir / "status.json",
            {
                "status": "stopped",
                "returncode": 0,
                "exception": None,
                "updated_at": "2026-08-13T02:00:00Z",
            },
        )
        (session_dir / "server.log").write_text(
            f"'mamba_ssm_cache_dtype': '{state_dtype}'\n"
            f"kv_cache_dtype={kv_dtype}\n"
            "CUDAGraphMode.PIECEWISE\n",
            encoding="utf-8",
        )


def test_audit_server_sessions_checks_realized_precision_logs(tmp_path: Path) -> None:
    make_server_sessions(tmp_path)

    sessions = audit_server_sessions(tmp_path)

    assert len(sessions) == 4
    assert {row["precision_log_evidence"] for row in sessions} == {"PASS"}
    assert {row["returncode"] for row in sessions} == {0}


def test_audit_server_sessions_rejects_precision_log_drift(tmp_path: Path) -> None:
    make_server_sessions(tmp_path)
    log = tmp_path / "servers" / "joint" / "session-joint" / "server.log"
    log.write_text("wrong precision\n", encoding="utf-8")

    with pytest.raises(AuditError, match="precision log evidence missing"):
        audit_server_sessions(tmp_path)


def test_audit_launch_reports_incomplete_provenance_without_backfill(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt-r2"
    attempt_dir.mkdir()
    launch_dir = tmp_path / "launch" / attempt_dir.name
    launch_dir.mkdir(parents=True)
    (launch_dir / "started_at").write_text("2026-08-13T01:00:00Z\n", encoding="ascii")
    (launch_dir / "run.log").write_text("completed\n", encoding="ascii")

    result = audit_launch(attempt_dir)

    assert result["complete"] is False
    assert "pid" in result["warning"]
    assert result["files"]["started_at"] is True


def test_audit_launch_accepts_complete_successful_wrapper(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt-r3"
    attempt_dir.mkdir()
    launch_dir = tmp_path / "launch" / attempt_dir.name
    launch_dir.mkdir(parents=True)
    (launch_dir / "pid").write_text("1234\n", encoding="ascii")
    (launch_dir / "started_at").write_text("2026-08-13T01:00:00Z\n", encoding="ascii")
    (launch_dir / "finished_at").write_text("2026-08-13T02:00:00Z\n", encoding="ascii")
    (launch_dir / "exit_code").write_text("0\n", encoding="ascii")
    (launch_dir / "run.log").write_text("completed\n", encoding="ascii")

    result = audit_launch(attempt_dir)

    assert result["complete"] is True
    assert result["exit_code"] == 0
    assert result["duration_s"] == 3600.0
    assert result["warning"] is None


def test_partial_artifacts_fail_closed(tmp_path: Path) -> None:
    partial = tmp_path / "samples" / "cell" / "bench.log.partial"
    partial.parent.mkdir(parents=True)
    partial.write_text("incomplete\n", encoding="utf-8")

    with pytest.raises(AuditError, match=r"partial artifacts remain: samples[\\/]cell[\\/]bench\.log\.partial"):
        require_no_partial_artifacts(tmp_path)
