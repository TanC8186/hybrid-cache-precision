from __future__ import annotations

import math

import pytest

from scripts.analyze.analyze_joint_precision_calibration import (
    aggregate_calibration,
    calibration_fallacy_scan,
    student_t_summary,
)
from scripts.analyze.verify_a2_reproduction import VerificationError

T_CRITICAL_DF2 = 4.302652729911275


def matrix() -> dict:
    return {
        "allocations": [{"id": "full"}],
        "seeds": [7, 42, 2026],
        "workload_rates_req_s": {"random": [30]},
        "ttft_thresholds_ms": [250, 500],
        "tpot_threshold_ms": 200,
    }


def sample(seed: int, offset: float, *, failed: int = 0) -> dict:
    expected = 1800
    return {
        "sample_id": f"full__random__r30__s{seed}",
        "allocation": "full",
        "workload": "random",
        "offered_rate_req_s": 30.0,
        "seed": seed,
        "accounting": {
            "expected": expected,
            "completed": expected - failed,
            "failed": failed,
        },
        "request_throughput_req_s": 28.0 + offset,
        "ttft_p95_ms": 180.0 + offset,
        "tpot_p95_ms": 18.0 + offset,
        "slo_sweep": {
            "250": {"goodput_req_s": 27.0 + offset},
            "500": {"goodput_req_s": 28.0 + offset},
        },
    }


def test_student_t_summary_uses_independent_repeat_count() -> None:
    summary = student_t_summary([1.0, 2.0, 3.0], T_CRITICAL_DF2)

    expected_margin = T_CRITICAL_DF2 / math.sqrt(3)
    assert summary["n"] == 3
    assert summary["mean"] == 2.0
    assert summary["sample_sd"] == 1.0
    assert summary["ci95_low"] == pytest.approx(2.0 - expected_margin)
    assert summary["ci95_high"] == pytest.approx(2.0 + expected_margin)


def test_aggregate_builds_profile_bounds_and_preserves_failures() -> None:
    samples = [
        sample(7, 0.0),
        sample(42, 1.0, failed=2),
        sample(2026, 2.0),
    ]

    result = aggregate_calibration(samples, matrix(), t_critical=T_CRITICAL_DF2)

    assert result["cell_count"] == 1
    assert result["profile_row_count"] == 2
    cell = result["cells"][0]
    assert cell["expected_requests"] == 5400
    assert cell["completed_requests"] == 5398
    assert cell["failed_requests"] == 2
    profile = result["profile_inputs"]["full"]["random"]["30"]
    assert profile["n_independent_repeats"] == 3
    assert profile["p95_ttft_ucb_ms"] == cell["p95_ttft_ms"]["ci95_high"]
    assert profile["p95_tpot_ucb_ms"] == cell["p95_tpot_ms"]["ci95_high"]
    assert profile["slo_sweep"]["250"]["slo_goodput_lcb_req_s"] == cell["slo_sweep"]["250"]["goodput_req_s"]["ci95_low"]


@pytest.mark.parametrize(
    "samples",
    [
        [sample(7, 0.0), sample(42, 1.0)],
        [sample(7, 0.0), sample(42, 1.0), sample(42, 2.0)],
    ],
)
def test_aggregate_rejects_missing_or_duplicate_seed(samples: list[dict]) -> None:
    with pytest.raises(VerificationError, match="repeat seeds"):
        aggregate_calibration(samples, matrix(), t_critical=T_CRITICAL_DF2)


def test_calibration_fallacy_scan_covers_all_eleven_categories() -> None:
    scan = calibration_fallacy_scan()

    assert len(scan) == 11
    assert len({item["fallacy"] for item in scan}) == 11
