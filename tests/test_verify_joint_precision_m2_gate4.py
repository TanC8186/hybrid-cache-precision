from __future__ import annotations

import math

import pytest

from scripts.analyze.audit_joint_precision_m2_pilot import AuditError
from scripts.analyze.verify_joint_precision_m2_gate4 import (
    build_reproduction_package,
    compare_audits,
    symmetric_relative_difference,
)


def test_symmetric_relative_difference_handles_zero() -> None:
    assert symmetric_relative_difference(0.0, 0.0, epsilon=1e-12) == 0.0
    assert math.isclose(symmetric_relative_difference(10.0, 9.0, epsilon=1e-12), 0.1)


def test_build_reproduction_package_preserves_request_paths_and_removes_hash_fields() -> None:
    selector = {
        "profile": {"path": "profile.json"},
        "serving_config": {"path": "serving.yaml"},
        "requests": [
            {"id": "strict", "path": "strict.json", "sha256": "ignored", "expected_selected_config_id": "full"},
            {"id": "medium", "path": "medium.json", "sha256": "ignored", "expected_selected_config_id": "state_only"},
            {"id": "high", "path": "high.json", "sha256": "ignored", "expected_selected_config_id": "joint"},
        ],
    }
    gate4 = {
        "package_id": "gate4",
        "classification": "reproduction",
        "attempts": [
            {"request_id": "strict", "expected_selected_config_id": "full", "reproduction_attempt": "strict-r1"},
            {"request_id": "medium", "expected_selected_config_id": "state_only", "reproduction_attempt": "medium-r1"},
            {"request_id": "high", "expected_selected_config_id": "joint", "reproduction_attempt": "high-r1"},
        ],
        "matrix": {"seeds": [11, 23, 47]},
    }

    package = build_reproduction_package(gate4, selector)

    assert package["requests"][0] == {
        "id": "strict",
        "path": "strict.json",
        "expected_selected_config_id": "full",
        "attempt_id": "strict-r1",
    }
    assert "sha256" not in package["requests"][0]


def _audit(goodput: float, ttft: float, tpot: float) -> dict:
    cells = []
    mapping = {"strict": "full", "medium": "state_only", "high": "joint"}
    for request_id, allocation in mapping.items():
        cells.append(
            {
                "request_id": request_id,
                "selected_config_id": allocation,
                "requested_slo_attainment": "3/3",
                "samples": [{"seed": 11}, {"seed": 23}, {"seed": 47}],
                "metric_summaries": {
                    "goodput_req_s": {"mean": goodput},
                    "p95_ttft_ms": {"mean": ttft},
                    "p95_tpot_ms": {"mean": tpot},
                },
            }
        )
    return {
        "gate_2_verdict": "PASS",
        "audit": {"hash_validation_performed": False},
        "completeness": {"audited_samples": 9},
        "decision_metrics": {"mapping": mapping},
        "scope": {"model_id": "model"},
        "fallacy_scan": {"coverage": "11/11"},
        "cells": cells,
    }


def test_compare_audits_passes_frozen_tolerances() -> None:
    contract = {
        "comparison": {
            "symmetric_relative_tolerances": {
                "mean_goodput_req_s": 0.10,
                "mean_p95_ttft_ms": 0.20,
                "mean_p95_tpot_ms": 0.20,
            },
            "epsilon": 1e-12,
        }
    }

    comparison = compare_audits(_audit(40, 400, 40), _audit(38, 440, 36), contract)

    assert comparison["all_within_tolerance"] is True


def test_compare_audits_fails_closed_on_slo_seed_loss() -> None:
    contract = {
        "comparison": {
            "symmetric_relative_tolerances": {
                "mean_goodput_req_s": 0.10,
                "mean_p95_ttft_ms": 0.20,
                "mean_p95_tpot_ms": 0.20,
            },
            "epsilon": 1e-12,
        }
    }
    reproduction = _audit(40, 400, 40)
    reproduction["cells"][2]["requested_slo_attainment"] = "2/3"

    with pytest.raises(AuditError, match="SLO"):
        compare_audits(_audit(40, 400, 40), reproduction, contract)
