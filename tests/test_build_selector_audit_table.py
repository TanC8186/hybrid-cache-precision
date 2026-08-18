from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.controller.build_selector_audit_table import (
    AuditError,
    build_audit,
    render_latex,
)


def write_decision(path: Path, *, feasible: bool) -> None:
    reasons = [] if feasible else ["quality_guardrail_violated:gsm8k"]
    decision = {
        "status": "SELECTED" if feasible else "NO_FEASIBLE_CANDIDATE",
        "request": {
            "workload": "random",
            "offered_rate_req_s": 40,
            "required_allocator_equivalent_sequence_slots": 250,
            "quality_constraints": {"gsm8k": -0.01},
            "memory_budget": {"max_cache_bytes": 100},
            "slo": {"p95_ttft_ms": 500, "p95_tpot_ms": 200},
        },
        "evaluations": [
            {
                "config_id": "full",
                "feasible": feasible,
                "rejection_reasons": reasons,
                "objective_lcb_req_s": 29.5,
                "allocator_equivalent_sequence_slots": 270,
                "constraint_checks": {
                    "capacity": {"profile_found": True},
                    "serving": {"p95_ttft_ucb_ms": 200, "p95_tpot_ucb_ms": 20},
                    "quality": {"gsm8k": {"delta_ci95": [0.0, 0.0]}},
                },
            }
        ],
        "selected": {"config_id": "full", "objective_lcb_req_s": 29.5} if feasible else None,
    }
    path.write_text(json.dumps(decision), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}\n", encoding="ascii")


def test_build_audit_and_render_latex(tmp_path: Path) -> None:
    selected = tmp_path / "strict.json"
    rejected = tmp_path / "medium.json"
    write_decision(selected, feasible=True)
    write_decision(rejected, feasible=False)

    audit = build_audit([("strict", selected), ("medium", rejected)])
    latex = render_latex(audit["decisions"])

    assert audit["decisions"][0]["selected_config_id"] == "full"
    assert audit["decisions"][1]["status"] == "NO_FEASIBLE_CANDIDATE"
    assert r"\textbf{selected}" in latex
    assert "reject: quality" in latex
    assert "allocator-equivalent sequence slots" in latex


def test_build_audit_rejects_tampered_decision(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    write_decision(path, feasible=True)
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(AuditError, match="SHA-256 verification failed"):
        build_audit([("strict", path)])
