from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[1] / "scripts" / "quality" / "verify_m4_gate4.py"
_SPEC = importlib.util.spec_from_file_location("verify_m4_gate4", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compare_boundaries = _MODULE.compare_boundaries
compare_cells = _MODULE.compare_cells
symmetric_relative_difference = _MODULE.symmetric_relative_difference


def row(sample_id: str, goodput: float, *, sustainable: bool = True) -> dict:
    return {
        "sample_id": sample_id,
        "allocation": "full",
        "workload": "random",
        "rate": 30.0,
        "seed": 11,
        "expected_requests": 1800,
        "completed_requests": 1800,
        "failed_requests": 0,
        "goodput_250": goodput,
        "sustainable": {"250": sustainable},
    }


def test_symmetric_relative_difference_uses_larger_magnitude() -> None:
    assert symmetric_relative_difference(9.0, 10.0) == pytest.approx(0.1)
    assert symmetric_relative_difference(0.0, 0.0) == 0.0


def test_compare_cells_requires_strictly_less_than_ten_percent() -> None:
    comparisons = compare_cells(
        [row("full__random__r30__s11", 10.0)],
        [row("full__random__r30__s11", 9.0)],
        thresholds=(250.0,),
        tolerance=0.10,
    )

    assert len(comparisons) == 1
    assert comparisons[0]["symmetric_relative_difference"] == pytest.approx(0.1)
    assert comparisons[0]["within_tolerance"] is False


def test_compare_cells_reports_boundary_point_flips_separately() -> None:
    comparisons = compare_cells(
        [row("full__random__r30__s11", 29.0, sustainable=True)],
        [row("full__random__r30__s11", 28.0, sustainable=False)],
        thresholds=(250.0,),
        tolerance=0.10,
    )

    assert comparisons[0]["within_tolerance"] is True
    assert comparisons[0]["boundary_point_exact"] is False


def test_compare_boundaries_is_exact_and_stratified() -> None:
    original = [
        {"allocation": "full", "workload": "random", "ttft_threshold_ms": 250.0, "boundary_req_s": 40.0},
        {"allocation": "joint", "workload": "random", "ttft_threshold_ms": 250.0, "boundary_req_s": 45.0},
    ]
    rerun = [
        {"allocation": "full", "workload": "random", "ttft_threshold_ms": 250.0, "boundary_req_s": 40.0},
        {"allocation": "joint", "workload": "random", "ttft_threshold_ms": 250.0, "boundary_req_s": 40.0},
    ]

    comparisons = compare_boundaries(original, rerun)

    assert sum(item["exact_match"] for item in comparisons) == 1
    assert len(comparisons) == 2
