from pathlib import Path

import pytest

from scripts.bench.analyze_capacity_phase_diagram import (
    expected_cells,
    parse_cell_name,
    validate_cell_record,
)


@pytest.mark.parametrize(
    ("phase", "count"),
    [("mvex", 2), ("pilot", 18), ("formal", 112)],
)
def test_expected_capacity_phase_cell_counts(phase: str, count: int) -> None:
    assert len(expected_cells(phase)) == count


def test_parse_capacity_phase_cell_name() -> None:
    meta = parse_cell_name(
        Path("capacity-phase-pilot-20260811__2b__kvint4__statebfloat16__L16384__u080.json"),
        "capacity-phase-pilot-20260811",
    )

    assert meta == {
        "model": "2b",
        "kv": "int4",
        "state": "bfloat16",
        "length": 16384,
        "util": 0.8,
    }


def test_expected_cells_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="phase must be"):
        expected_cells("unknown")


def make_capacity_record() -> dict:
    return {
        "schema_version": 1,
        "probe": "probe_ssm_state_dtype.py",
        "args": {
            "model": "/models/Qwen3.5-2B",
            "dtype_arg": "bfloat16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.85,
            "seed": 42,
            "kv_cache_dtype": "int4_per_token_head",
            "kv_cache_dtype_per_layer": {},
        },
        "resolved_mamba_ssm_cache_dtype": "bfloat16",
        "capacity": {"tokens": 1000, "max_concurrency": 0.244, "max_model_len": 4096},
        "cache_config": {
            "mamba_ssm_cache_dtype": "bfloat16",
            "cache_dtype": "int4_per_token_head",
            "num_gpu_blocks": 10,
        },
        "generation": None,
        "elapsed_seconds": 1.0,
    }


def test_validate_capacity_cell_accepts_frozen_protocol() -> None:
    meta = {"model": "2b", "kv": "int4", "state": "bfloat16", "length": 4096, "util": 0.85}

    validate_cell_record(make_capacity_record(), meta, Path("cell.json"))


def test_validate_capacity_cell_rejects_wrong_seed() -> None:
    meta = {"model": "2b", "kv": "int4", "state": "bfloat16", "length": 4096, "util": 0.85}
    record = make_capacity_record()
    record["args"]["seed"] = 7

    with pytest.raises(SystemExit, match="seed mismatch"):
        validate_cell_record(record, meta, Path("cell.json"))
