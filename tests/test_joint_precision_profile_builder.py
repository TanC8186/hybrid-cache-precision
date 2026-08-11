from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from kvcache.policy import canonical_precision_args
from scripts.controller.build_joint_precision_profile import (
    ProfileBuildError,
    build_profile,
    resolve_json_pointer,
    verify_profile_evidence,
)

MODEL_ID = "Qwen/Qwen3.5-2B"


def write_evidence(repo_root: Path, relative_path: str, value: dict) -> str:
    path = repo_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def source(evidence_id: str, json_pointer: str) -> dict:
    return {"source": {"evidence_id": evidence_id, "json_pointer": json_pointer}}


def recipe(capacity_sha: str, serving_sha: str, quality_sha: str) -> dict:
    return {
        "schema_version": 1,
        "profile_status": "VERIFIED",
        "evidence": [
            {
                "evidence_id": "capacity",
                "path": "evidence/capacity.json",
                "expected_sha256": capacity_sha,
                "verification_status": "VERIFIED",
            },
            {
                "evidence_id": "serving",
                "path": "evidence/serving.json",
                "expected_sha256": serving_sha,
                "verification_status": "VERIFIED",
            },
            {
                "evidence_id": "quality",
                "path": "evidence/quality.json",
                "expected_sha256": quality_sha,
                "verification_status": "ANALYZED",
            },
        ],
        "candidates": [
            {
                "config_id": "joint",
                "kv_cache_dtype": "int4_per_token_head",
                "state_cache_dtype": "bfloat16",
                "deployment": {
                    "engine": "vllm",
                    "allocation": "joint",
                    "precision_args": canonical_precision_args("int4_per_token_head", "bfloat16"),
                    "restart_required": True,
                },
                "capacity_profiles": [
                    {
                        "model_id": MODEL_ID,
                        "max_model_len": 4096,
                        "gpu_memory_utilization": 0.85,
                        "cache_bytes": source("capacity", "/cache/total_bytes"),
                        "max_concurrency": source("capacity", "/capacity/max_concurrency"),
                    }
                ],
                "serving_profiles": [
                    {
                        "model_id": MODEL_ID,
                        "max_model_len": 4096,
                        "workload": "random",
                        "offered_rate_req_s": 40,
                        "slo": {"p95_ttft_ms": 500, "p95_tpot_ms": 200},
                        "slo_goodput_lcb_req_s": source("serving", "/metrics/goodput_lcb"),
                        "p95_ttft_ucb_ms": source("serving", "/metrics/ttft_ucb"),
                        "p95_tpot_ucb_ms": source("serving", "/metrics/tpot_ucb"),
                        "n_independent_repeats": source("serving", "/n"),
                    }
                ],
                "quality_profiles": [
                    {
                        "model_id": MODEL_ID,
                        "task": "gsm8k",
                        "delta_ci95_low": source("quality", "/ci/0"),
                        "delta_ci95_high": source("quality", "/ci/1"),
                        "n_independent_repeats": source("quality", "/n"),
                    }
                ],
            }
        ],
    }


def evidence_fixture(repo_root: Path) -> tuple[str, str, str]:
    capacity_sha = write_evidence(
        repo_root,
        "evidence/capacity.json",
        {"cache": {"total_bytes": 80}, "capacity": {"max_concurrency": 64}},
    )
    serving_sha = write_evidence(
        repo_root,
        "evidence/serving.json",
        {"metrics": {"goodput_lcb": 39, "ttft_ucb": 450, "tpot_ucb": 150}, "n": 3},
    )
    quality_sha = write_evidence(repo_root, "evidence/quality.json", {"ci": [-1.0, -0.5], "n": 9})
    return capacity_sha, serving_sha, quality_sha


def test_builder_materializes_metrics_and_computes_evidence_ids(tmp_path: Path) -> None:
    built = build_profile(recipe(*evidence_fixture(tmp_path)), tmp_path)

    row = built["candidates"][0]
    assert row["capacity_profiles"][0]["cache_bytes"] == 80
    assert row["capacity_profiles"][0]["evidence_ids"] == ["capacity"]
    assert row["serving_profiles"][0]["slo_goodput_lcb_req_s"] == 39
    assert row["serving_profiles"][0]["evidence_ids"] == ["serving"]
    assert row["quality_profiles"][0]["delta_ci95_low"] == -1.0
    assert verify_profile_evidence(built, tmp_path)[0]["evidence_id"] == "capacity"


def test_builder_rejects_tampered_evidence(tmp_path: Path) -> None:
    hashes = evidence_fixture(tmp_path)
    (tmp_path / "evidence" / "capacity.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ProfileBuildError, match="recipe digest mismatch"):
        build_profile(recipe(*hashes), tmp_path)


def test_builder_requires_every_profile_row_to_source_evidence(tmp_path: Path) -> None:
    calibration_recipe = recipe(*evidence_fixture(tmp_path))
    calibration_recipe["candidates"][0]["capacity_profiles"][0]["cache_bytes"] = 80
    calibration_recipe["candidates"][0]["capacity_profiles"][0]["max_concurrency"] = 64

    with pytest.raises(ProfileBuildError, match="must be sourced from evidence"):
        build_profile(calibration_recipe, tmp_path)


def test_json_pointer_handles_arrays_and_escaping() -> None:
    document = {"a/b": {"~key": [10, 20]}}

    assert resolve_json_pointer(document, "/a~1b/~0key/1") == 20
