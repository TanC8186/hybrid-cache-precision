import pytest

from scripts.analyze.verify_a2_reproduction import VerificationError
from scripts.analyze.verify_a2_serving_pilot import (
    classify_pilot,
    detect_fatal_signatures,
    request_accounting,
)


def result_fixture(*, completed: int, failed: int) -> dict:
    expected = completed + failed
    errors = [""] * completed + ["request failed"] * failed
    return {
        "completed": completed,
        "failed": failed,
        "ttfts": [0.1] * expected,
        "itls": [[] for _ in range(expected)],
        "output_lens": [1] * expected,
        "start_times": [float(index) for index in range(expected)],
        "errors": errors,
    }


def test_request_accounting_accepts_explicit_failures() -> None:
    accounting = request_accounting(result_fixture(completed=3, failed=2), expected=5)

    assert accounting["request_conservation"] is True
    assert accounting["completed"] == 3
    assert accounting["failed"] == 2


def test_request_accounting_rejects_reduced_denominator() -> None:
    with pytest.raises(VerificationError, match=r"completed \+ failed"):
        request_accounting(result_fixture(completed=3, failed=1), expected=5)


def test_fatal_signature_detection_requires_both_runtime_markers() -> None:
    signatures = detect_fatal_signatures(
        "EngineCore encountered a fatal error\n"
        "torch.AcceleratorError: CUDA error: an illegal instruction was encountered\n"
    )

    assert all(signatures.values())


def test_failed_pilot_classification_is_fail_closed() -> None:
    verdict = classify_pilot(
        supervisor_exit_code=2,
        summary_counts={"completed_validated": 1, "failed": 1, "not_started": 1},
        fatal_signatures={
            "EngineCore encountered a fatal error": True,
            "CUDA error: an illegal instruction was encountered": True,
        },
        failed_sample_zero_success=True,
    )

    assert verdict == "PILOT_FAILED_RUNTIME_CUDA_ILLEGAL_INSTRUCTION"
