from scripts.analyze.verify_a2_comparative_protocol_v3_formal_slice import (
    classify_formal_slice,
)


def test_formal_slice_passes_only_when_every_gate_passes() -> None:
    verdict = classify_formal_slice(
        requests_passed=True,
        summary_passed=True,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_RANDOM_FORMAL_SLICE_001_PASSED"


def test_formal_slice_fails_closed() -> None:
    verdict = classify_formal_slice(
        requests_passed=True,
        summary_passed=False,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_RANDOM_FORMAL_SLICE_001_REVIEW_REQUIRED"
