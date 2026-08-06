from pathlib import Path

from scripts.bench.run_steady_state import (
    build_sample_plan,
    load_config,
    resolve_phase,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "experiments" / "configs"
ALLOCATIONS = {"fp16", "int4", "packed_per_layer"}
SEEDS = {7, 42, 2026}


def load_plan(config_name: str, phase_name: str):
    config = load_config(CONFIG_ROOT / config_name)
    phase = resolve_phase(config, phase_name)
    return config, build_sample_plan(config, phase)


def test_random_formal_contract() -> None:
    config, plan = load_plan(
        "a2_comparative_piecewise_protocol_v3_random60_formal.yaml",
        "comparative_random60_formal_v3",
    )

    assert config["protocol"]["protocol_version"] == 3
    assert config["protocol"]["measurement_window_s"] == 60
    assert len(plan) == 45
    assert {sample["allocation"] for sample in plan} == ALLOCATIONS
    assert {sample["seed"] for sample in plan} == SEEDS
    assert {sample["request_rate"] for sample in plan} == {
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
    }
    assert {sample["workload"] for sample in plan} == {"random"}
    assert sum(sample["num_prompts"] for sample in plan) == 108_000


def test_sharegpt_formal_contract() -> None:
    config, plan = load_plan(
        "a2_comparative_piecewise_protocol_v3_sharegpt300_formal.yaml",
        "comparative_sharegpt300_formal_v3",
    )

    assert config["protocol"]["protocol_version"] == 3
    assert config["protocol"]["measurement_window_s"] == 300
    assert len(plan) == 63
    assert {sample["allocation"] for sample in plan} == ALLOCATIONS
    assert {sample["seed"] for sample in plan} == SEEDS
    assert {sample["request_rate"] for sample in plan} == {
        20.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
        50.0,
    }
    assert {sample["workload"] for sample in plan} == {"sharegpt"}
    assert sum(sample["num_prompts"] for sample in plan) == 661_500


def test_all_formal_allocations_require_piecewise_graph_mode() -> None:
    for config_name in (
        "a2_comparative_piecewise_protocol_v3_random60_formal.yaml",
        "a2_comparative_piecewise_protocol_v3_sharegpt300_formal.yaml",
    ):
        config = load_config(CONFIG_ROOT / config_name)
        for allocation in ALLOCATIONS:
            required = config["allocations"][allocation]["required_log_substrings"]
            assert "CUDAGraphMode.PIECEWISE" in required
