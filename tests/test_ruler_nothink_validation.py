import hashlib
import json
from pathlib import Path

import pytest

from scripts.eval.validate_ruler_nothink_5cell import (
    CellSpec,
    ValidationError,
    expected_specs,
    validate_cell_record,
    verify_sha256_sidecar,
)


def make_record(spec: CellSpec) -> dict:
    cases = [
        {
            "index": 1000 + index * 7,
            "references": [f"ref-{index}-a", f"ref-{index}-b"],
            "hits": [True, index != 0],
            "prompt_tokens": 100 + index,
            "output_tokens": 4,
        }
        for index in range(20)
    ]
    accuracy = round(
        100.0 * sum(sum(case["hits"]) for case in cases) / sum(len(case["hits"]) for case in cases),
        2,
    )
    state_dtype = "bfloat16" if spec.allocation == "fp16_statebf16" else "float32"
    kwargs = {
        "seed": 7,
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "disable_log_stats": True,
    }
    if spec.allocation == "fp16_statebf16":
        kwargs["mamba_ssm_cache_dtype"] = "bfloat16"
    seed_dir = "" if spec.dataset_seed == 42 else f"seed{spec.dataset_seed}/"
    task_type = "freq_words_extraction" if spec.task == "ruler_fwe" else "niah"
    return {
        "schema_version": 1,
        "attempt_id": spec.attempt,
        "status": "completed_validated",
        "task": spec.task,
        "task_type": task_type,
        "length": spec.length,
        "allocation": spec.allocation,
        "seed": 7,
        "thinking": "disabled",
        "max_tokens": 256,
        "metric": "string_match_all",
        "num_samples": 20,
        "sampling_params": {"max_tokens": 256, "temperature": 0.0, "thinking": "disabled"},
        "data_file": f"data/ruler/{spec.task}_L{spec.length}/{seed_dir}validation.jsonl",
        "data_sha256": "a" * 64,
        "config_effect": {
            "ok": True,
            "allocation": spec.allocation,
            "detail": {
                "cache_dtype": "auto",
                "per_layer": {},
                "a2_flag": False,
                "mamba_ssm_cache_dtype": state_dtype,
            },
        },
        "engine": {
            "model": f"Qwen3.5-{'2B' if spec.model == '2b' else '9B'}",
            "kwargs": kwargs,
            "vllm_version": "test-version",
        },
        "cases": cases,
        "accuracy": accuracy,
        "elapsed_s": 1.0,
        "ruler_commit": "test-commit",
        "host": "test-host",
    }


@pytest.mark.parametrize("allocation", ["fp16", "fp16_statebf16"])
def test_validate_cell_record_accepts_exact_protocol(allocation: str) -> None:
    spec = CellSpec("ruler_fwe", 4096, "2b", 11, allocation, "attempt")
    summary = validate_cell_record(make_record(spec), spec, Path("cell.json"))

    assert summary["state_dtype"] == ("bfloat16" if allocation.endswith("statebf16") else "float32")
    assert summary["accuracy"] == summary["recomputed_accuracy"]


def test_validate_cell_record_rejects_wrong_state_dtype() -> None:
    spec = CellSpec("ruler_fwe", 4096, "2b", 42, "fp16_statebf16", "attempt")
    record = make_record(spec)
    record["config_effect"]["detail"]["mamba_ssm_cache_dtype"] = "float32"

    with pytest.raises(ValidationError, match="state dtype mismatch"):
        validate_cell_record(record, spec, Path("cell.json"))


def test_validate_cell_record_rejects_accuracy_mismatch() -> None:
    spec = CellSpec("ruler_niah_multiquery", 8192, "9b", 23, "fp16", "attempt")
    record = make_record(spec)
    record["accuracy"] += 1.0

    with pytest.raises(ValidationError, match="accuracy recomputation mismatch"):
        validate_cell_record(record, spec, Path("cell.json"))


def test_verify_sha256_sidecar_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "cell.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    Path(str(path) + ".sha256").write_text("0" * 64 + "\n", encoding="ascii")

    with pytest.raises(ValidationError, match="SHA256 mismatch"):
        verify_sha256_sidecar(path)


def test_verify_sha256_sidecar_accepts_match(tmp_path: Path) -> None:
    path = tmp_path / "cell.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(digest + "\n", encoding="ascii")

    assert verify_sha256_sidecar(path) == digest


def test_expected_specs_has_frozen_30_cell_matrix() -> None:
    specs = expected_specs("attempt-2b", "attempt-9b")

    assert len(specs) == 30
    assert sum(spec.model == "2b" for spec in specs) == 12
    assert sum(spec.model == "9b" for spec in specs) == 18
