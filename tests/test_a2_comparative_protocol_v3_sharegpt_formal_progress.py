import pytest

from scripts.analyze.verify_a2_comparative_protocol_v3_sharegpt_formal_progress import (
    classify_sharegpt_formal_progress,
    completed_count_for_slice,
    expected_server_allocations,
    expected_sidecar_count,
)
from scripts.analyze.verify_a2_reproduction import VerificationError


def test_sharegpt_formal_progress_passes_only_when_every_gate_passes() -> None:
    verdict = classify_sharegpt_formal_progress(
        slice_number=1,
        requests_passed=True,
        summary_passed=True,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_SHAREGPT_FORMAL_SLICE_001_PASSED"


def test_sharegpt_formal_progress_fails_closed() -> None:
    verdict = classify_sharegpt_formal_progress(
        slice_number=1,
        requests_passed=True,
        summary_passed=True,
        servers_clean=False,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_SHAREGPT_FORMAL_SLICE_001_REVIEW_REQUIRED"


def test_final_sharegpt_slice_caps_completed_count_at_plan_size() -> None:
    assert completed_count_for_slice(12) == 60
    assert completed_count_for_slice(13) == 63


def test_sharegpt_slice_number_fails_closed_outside_plan() -> None:
    with pytest.raises(VerificationError):
        completed_count_for_slice(14)


def test_server_allocations_include_cross_allocation_slice_transitions() -> None:
    plan = (
        [{"allocation": "fp16"}] * 21
        + [{"allocation": "int4"}] * 21
        + [{"allocation": "packed_per_layer"}] * 21
    )

    assert expected_server_allocations(plan, 4) == ("fp16",) * 4
    assert expected_server_allocations(plan, 5) == (
        "fp16",
        "fp16",
        "fp16",
        "fp16",
        "fp16",
        "int4",
    )
    assert expected_server_allocations(plan, 9) == (
        ("fp16",) * 5 + ("int4",) * 5 + ("packed_per_layer",)
    )
    assert expected_server_allocations(plan, 13) == (
        ("fp16",) * 5
        + ("int4",) * 5
        + ("packed_per_layer",) * 5
    )


def test_sidecar_count_tracks_all_server_sessions() -> None:
    plan = (
        [{"allocation": "fp16"}] * 21
        + [{"allocation": "int4"}] * 21
        + [{"allocation": "packed_per_layer"}] * 21
    )

    allocations = expected_server_allocations(plan, 5)

    assert len(allocations) == 6
    assert expected_sidecar_count(25, allocations) == 84
