from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.bench.run_steady_state import (
    ExperimentError,
    analyze_result,
    build_benchmark_command,
    build_sample_plan,
    load_config,
    resolve_phase,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "e3_steady_state_2b.yaml"


def test_pilot_plan_is_one_seed_and_twelve_samples() -> None:
    config = load_config(CONFIG_PATH)
    phase = resolve_phase(config, "pilot")
    plan = build_sample_plan(config, phase)

    assert len(plan) == 12
    assert {sample["seed"] for sample in plan} == {7}
    assert {sample["num_prompts"] for sample in plan} == {1800, 2400, 3000}
    assert plan[0]["sample_id"] == "fp16__random__r30__s7"


def test_benchmark_command_freezes_window_and_sharegpt_pairing(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    sample = {
        "sample_id": "int4__sharegpt__r30__s7",
        "allocation": "int4",
        "workload": "sharegpt",
        "request_rate": 30.0,
        "seed": 7,
        "num_prompts": 1800,
    }
    command = build_benchmark_command(
        config,
        sample,
        repo_root=ROOT,
        attempt_id="pilot-test",
        result_dir=tmp_path,
        contract_sha256="a" * 64,
        git_commit="b" * 40,
        vllm_source_commit="c" * 40,
    )

    joined = " ".join(command)
    assert "--num-prompts 1800" in joined
    assert "--num-warmups 120" in joined
    assert "--request-rate 30" in joined
    assert "--dataset-name sharegpt" in joined
    assert "--ignore-eos" in command
    assert "--no-oversample" in command
    assert "--max-concurrency" not in command
    assert "ttft:3000" in command
    assert "tpot:200" in command


def make_result(
    *,
    duration: float = 60.0,
    start_span: float = 59.0,
    ttfts: list[float] | None = None,
    itls: list[list[float]] | None = None,
) -> dict:
    ttfts = ttfts or [0.1, 0.2, 0.6, 1.5]
    itls = itls or [[0.01], [0.02], [0.03], [0.04]]
    completed = len(ttfts)
    return {
        "duration": duration,
        "completed": completed,
        "failed": 0,
        "request_throughput": completed / duration,
        "request_goodput": completed / duration,
        "p99_ttft_ms": 1473.0,
        "p99_tpot_ms": 39.4,
        "ttfts": ttfts,
        "itls": itls,
        "output_lens": [2] * completed,
        "start_times": [100.0 + start_span * index / (completed - 1) for index in range(completed)],
        "errors": [None] * completed,
    }


def test_analysis_recomputes_threshold_sweep() -> None:
    protocol = {
        "measurement_window_s": 60,
        "ttft_thresholds_ms": [250, 500, 1000, 2000, 3000],
        "tpot_threshold_ms": 200,
        "sustainable_goodput_ratio": 0.95,
        "arrival_window_tolerance_fraction": 0.10,
        "goodput_crosscheck_abs_tolerance": 0.02,
    }
    sample = {
        "sample_id": "sample",
        "request_rate": 4 / 60,
        "num_prompts": 4,
    }
    analysis = analyze_result(make_result(), sample, protocol)

    assert analysis["slo_sweep"]["250"]["good_requests"] == 2
    assert analysis["slo_sweep"]["1000"]["good_requests"] == 3
    assert analysis["slo_sweep"]["2000"]["good_requests"] == 4
    assert analysis["slo_sweep"]["2000"]["sustainable"] is True


def test_analysis_fails_closed_on_arrival_window_drift() -> None:
    protocol = {
        "measurement_window_s": 60,
        "ttft_thresholds_ms": [3000],
        "tpot_threshold_ms": 200,
        "sustainable_goodput_ratio": 0.95,
        "arrival_window_tolerance_fraction": 0.10,
        "goodput_crosscheck_abs_tolerance": 0.02,
    }
    sample = {
        "sample_id": "sample",
        "request_rate": 4 / 60,
        "num_prompts": 4,
    }
    with pytest.raises(ExperimentError, match="arrival window drift"):
        analyze_result(make_result(start_span=40), sample, protocol)


def test_config_is_json_serializable_after_resolution() -> None:
    config = load_config(CONFIG_PATH)
    phase = resolve_phase(config, "formal", seed_filter=[42], rate_filter=[35])
    plan = build_sample_plan(config, phase)
    json.dumps(plan)
    assert len(plan) == 4
