from scripts.analyze.verify_a2_comparative_protocol_v3 import (
    classify_protocol_v3_chain,
    protocol_v3_disposition,
)


def test_protocol_v3_chain_requires_every_gate() -> None:
    verdict = classify_protocol_v3_chain(
        fd_mvex_passed=True,
        sharegpt_requests_passed=True,
        sharegpt_bracketed=True,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_MVEX_CHAIN_PASSED"


def test_protocol_v3_chain_fails_closed() -> None:
    verdict = classify_protocol_v3_chain(
        fd_mvex_passed=True,
        sharegpt_requests_passed=True,
        sharegpt_bracketed=False,
        servers_clean=True,
        graph_mode_proven=True,
    )

    assert verdict == "PROTOCOL_V3_MVEX_CHAIN_REVIEW_REQUIRED"


def test_passed_chain_permits_pilot_without_promoting_evidence() -> None:
    disposition = protocol_v3_disposition("PROTOCOL_V3_MVEX_CHAIN_PASSED")

    assert disposition["gate_passed"] is True
    assert disposition["evidence_status"] == "UNVERIFIED"
    assert "comparative pilot" in disposition["next_gate"]


def test_failed_bracket_is_quarantined_and_requires_new_mvex() -> None:
    disposition = protocol_v3_disposition(
        "PROTOCOL_V3_MVEX_CHAIN_REVIEW_REQUIRED"
    )

    assert disposition["gate_passed"] is False
    assert disposition["evidence_status"] == "QUARANTINED"
    assert "Do not start a comparative pilot" in disposition["next_gate"]
    assert "upper-neighbor MVEx" in disposition["next_gate"]
