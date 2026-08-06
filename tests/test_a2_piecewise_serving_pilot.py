from scripts.analyze.verify_a2_piecewise_serving_pilot import (
    classify_piecewise_pilot,
    detect_runtime_faults,
)


def test_shutdown_engine_dead_error_is_not_a_runtime_fault() -> None:
    faults = detect_runtime_faults(
        "[shutdown] EngineCore: trigger received signal=SIGTERM\n"
        "EngineDeadError: EngineCore encountered an issue.\n"
    )

    assert not any(faults.values())


def test_cuda_illegal_instruction_is_a_runtime_fault() -> None:
    faults = detect_runtime_faults(
        "EngineCore encountered a fatal error\n"
        "CUDA error: an illegal instruction was encountered\n"
    )

    assert faults["EngineCore encountered a fatal error"] is True
    assert faults["CUDA error: an illegal instruction was encountered"] is True


def test_piecewise_chain_classification_requires_every_gate() -> None:
    verdict = classify_piecewise_pilot(
        diagnostic_passed=True,
        mvex_passed=True,
        pilot_passed=True,
        runtime_faults_absent=True,
        graph_mode_proven=True,
    )

    assert verdict == "PILOT_PASSED_PIECEWISE_RUNTIME_INTEGRITY"


def test_piecewise_chain_classification_fails_closed() -> None:
    verdict = classify_piecewise_pilot(
        diagnostic_passed=True,
        mvex_passed=True,
        pilot_passed=False,
        runtime_faults_absent=True,
        graph_mode_proven=True,
    )

    assert verdict == "PIECEWISE_CHAIN_INTEGRITY_REVIEW_REQUIRED"
