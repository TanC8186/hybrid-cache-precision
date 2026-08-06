from scripts.analyze.verify_a2_comparative_protocol_v3_upper50 import (
    classify_rate50_linked_chain,
)


def test_rate50_linked_chain_passes_only_when_every_gate_passes() -> None:
    verdict = classify_rate50_linked_chain(
        all_attempt_integrity_passed=True,
        full_threshold_bracketed=True,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_RATE50_LINKED_BRACKET_MVEX_PASSED"


def test_rate50_linked_chain_fails_closed() -> None:
    verdict = classify_rate50_linked_chain(
        all_attempt_integrity_passed=True,
        full_threshold_bracketed=False,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_RATE50_LINKED_BRACKET_MVEX_REVIEW_REQUIRED"
