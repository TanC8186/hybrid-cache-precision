from scripts.analyze.verify_a2_comparative_protocol_v3_formal_progress import (
    classify_formal_progress,
)


def test_formal_progress_passes_only_when_every_gate_passes() -> None:
    verdict = classify_formal_progress(
        slice_number=2,
        requests_passed=True,
        summary_passed=True,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_RANDOM_FORMAL_SLICE_002_PASSED"


def test_formal_progress_fails_closed() -> None:
    verdict = classify_formal_progress(
        slice_number=2,
        requests_passed=True,
        summary_passed=True,
        servers_clean=False,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_RANDOM_FORMAL_SLICE_002_REVIEW_REQUIRED"
