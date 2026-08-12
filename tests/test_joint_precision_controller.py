from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from kvcache.policy import canonical_precision_args
from scripts.controller import run_joint_precision_controller as controller

MODEL_ID = "Qwen/Qwen3.5-2B"


def config() -> dict:
    return {
        "schema_version": 1,
        "name": "joint_precision_controller_test",
        "controller": {"model_id": MODEL_ID},
        "environment": {
            "vllm_executable": "/data/venv/bin/vllm",
            "vllm_source_dir": "vendor/vllm",
            "model_path": "/data/models/qwen3.5-2b",
            "env": {"VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "75"},
        },
        "server": {
            "host": "127.0.0.1",
            "port": 8000,
            "args": [
                "--max-model-len",
                "4096",
                "--gpu-memory-utilization",
                "0.85",
            ],
        },
        "protocol": {
            "request_failure_policy": "count_as_slo_miss",
            "benchmark_client_keepalive_timeout_s": 60,
            "measurement_window_s": 60,
            "require_clean_git": False,
        },
        "allocations": {
            "joint": {
                "server_args": canonical_precision_args("int4_per_token_head", "bfloat16"),
            },
        },
        "workloads": {"random": {"dataset_name": "random"}},
        "phases": {
            "mvex": {
                "allocations": ["joint"],
                "seeds": [7, 42],
                "workload_rates": {"random": [35, 40]},
            }
        },
    }


def request() -> dict:
    return {
        "model_id": MODEL_ID,
        "max_model_len": 4096,
        "workload": "random",
        "offered_rate_req_s": 40,
        "memory_budget": {"gpu_memory_utilization": 0.85, "max_cache_bytes": 100},
        "required_concurrency": 50,
        "slo": {"p95_ttft_ms": 500, "p95_tpot_ms": 200},
        "quality_constraints": {},
    }


def decision() -> dict:
    return {
        "status": "SELECTED",
        "request": request(),
        "selected": {
            "config_id": "joint",
            "deployment": {
                "allocation": "joint",
                "precision_args": canonical_precision_args("int4_per_token_head", "bfloat16"),
                "restart_required": True,
            },
        },
    }


def profile(*, evidence_path: str = "results/fixture.json", evidence_sha256: str = "a" * 64) -> dict:
    evidence = {
        "evidence_id": "fixture",
        "path": evidence_path,
        "sha256": evidence_sha256,
        "verification_status": "VERIFIED",
    }
    return {
        "schema_version": 2,
        "profile_status": "VERIFIED",
        "evidence": [evidence],
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
                        "cache_bytes": 80,
                        "max_concurrency": 64,
                        "evidence_ids": ["fixture"],
                    }
                ],
                "serving_profiles": [
                    {
                        "model_id": MODEL_ID,
                        "max_model_len": 4096,
                        "workload": "random",
                        "offered_rate_req_s": 40,
                        "slo": {"p95_ttft_ms": 500, "p95_tpot_ms": 200},
                        "slo_goodput_lcb_req_s": 39,
                        "p95_ttft_ucb_ms": 450,
                        "p95_tpot_ucb_ms": 150,
                        "n_independent_repeats": 3,
                        "evidence_ids": ["fixture"],
                    }
                ],
                "quality_profiles": [],
            }
        ],
    }


def test_validate_deployment_mapping_proves_exact_precision_args(tmp_path: Path) -> None:
    mapping = controller.validate_deployment_mapping(
        config(),
        decision(),
        request(),
        repo_root=tmp_path,
    )

    assert mapping["allocation"] == "joint"
    assert mapping["restart_required"] is True
    assert mapping["server_command"][-4:] == canonical_precision_args(
        "int4_per_token_head",
        "bfloat16",
    )


def test_validate_deployment_mapping_rejects_runner_drift(tmp_path: Path) -> None:
    serving_config = config()
    serving_config["allocations"]["joint"]["server_args"][-1] = "float32"

    with pytest.raises(controller.ControllerError, match="runner mapping mismatch"):
        controller.validate_deployment_mapping(
            serving_config,
            decision(),
            request(),
            repo_root=tmp_path,
        )


def test_validate_execution_slice_is_exact() -> None:
    phase, plan = controller.validate_execution_slice(
        config(),
        "mvex",
        decision(),
        request(),
        [7],
    )

    assert phase == {
        "name": "mvex",
        "allocations": ["joint"],
        "seeds": [7],
        "workload_rates": {"random": [40.0]},
    }
    assert [row["sample_id"] for row in plan] == ["joint__random__r40__s7"]


def test_collect_controller_result_checks_denominator_and_startup(tmp_path: Path) -> None:
    command = ["vllm", "serve", "model"]
    (tmp_path / "servers" / "joint" / "session").mkdir(parents=True)
    (tmp_path / "samples" / "sample").mkdir(parents=True)
    (tmp_path / "summary.json").write_text(
        json.dumps({"counts": {"completed_validated": 1}}),
        encoding="utf-8",
    )
    (tmp_path / "servers" / "joint" / "session" / "contract.json").write_text(
        json.dumps({"command": command}),
        encoding="utf-8",
    )
    (tmp_path / "servers" / "joint" / "session" / "status.json").write_text(
        json.dumps({"status": "stopped", "exception": None, "startup_duration_s": 12.5}),
        encoding="utf-8",
    )
    (tmp_path / "samples" / "sample" / "analysis.json").write_text(
        json.dumps({"sample_id": "sample"}),
        encoding="utf-8",
    )

    result = controller.collect_controller_result(tmp_path, command)

    assert result["startup_duration_s"] == 12.5
    assert len(result["sample_analyses"]) == 1


def test_dry_run_keeps_controller_and_runner_attempt_paths_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    config_path = repo_root / "configs" / "experiments" / "controller.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config(), sort_keys=False), encoding="utf-8")
    profile_path = repo_root / "profile.json"
    request_path = repo_root / "request.json"
    evidence_path = repo_root / "results" / "fixture.json"
    evidence_path.parent.mkdir()
    evidence_path.write_text(json.dumps({"fixture": True}), encoding="utf-8")
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    evidence_path.with_suffix(".json.sha256").write_text(f"{evidence_sha256}\n", encoding="ascii")
    profile_path.write_text(json.dumps(profile(evidence_sha256=evidence_sha256)), encoding="utf-8")
    request_path.write_text(json.dumps(request()), encoding="utf-8")
    output_root = repo_root / "outputs"
    monkeypatch.setattr(
        controller,
        "get_git_state",
        lambda root, require_clean: {"commit": "b" * 40, "status": "", "clean": True},
    )

    rc = controller.main(
        [
            "--profile",
            str(profile_path),
            "--request",
            str(request_path),
            "--serving-config",
            str(config_path),
            "--phase",
            "mvex",
            "--attempt-id",
            "gate0-dry-run",
            "--output-root",
            str(output_root),
            "--seeds",
            "7",
            "--dry-run",
        ]
    )

    assert rc == 0
    attempt_dir = output_root / "gate0-dry-run"
    contract = json.loads((attempt_dir / "controller_contract.json").read_text(encoding="utf-8"))
    runner_output_index = contract["runner_argv"].index("--output-root") + 1
    assert Path(contract["runner_argv"][runner_output_index]) == attempt_dir / "runner"
    assert contract["runner_attempt_dir"] == str(attempt_dir / "runner" / "gate0-dry-run")
    assert not (attempt_dir / "runner").exists()


def test_fixture_profile_is_blocked_from_real_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    config_path = repo_root / "configs" / "experiments" / "controller.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config(), sort_keys=False), encoding="utf-8")
    evidence_path = repo_root / "results" / "fixture.json"
    evidence_path.parent.mkdir()
    evidence_path.write_text(json.dumps({"fixture": True}), encoding="utf-8")
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    evidence_path.with_suffix(".json.sha256").write_text(f"{evidence_sha256}\n", encoding="ascii")
    fixture_profile = profile(evidence_sha256=evidence_sha256)
    fixture_profile["profile_status"] = "TEST_FIXTURE"
    fixture_profile["evidence"][0]["verification_status"] = "FIXTURE"
    profile_path = repo_root / "profile.json"
    request_path = repo_root / "request.json"
    profile_path.write_text(json.dumps(fixture_profile), encoding="utf-8")
    request_path.write_text(json.dumps(request()), encoding="utf-8")
    output_root = repo_root / "outputs"
    monkeypatch.setattr(
        controller,
        "get_git_state",
        lambda root, require_clean: {"commit": "b" * 40, "status": "", "clean": True},
    )
    monkeypatch.setattr(
        controller,
        "run_steady_state",
        lambda argv: pytest.fail("fixture profile reached the real runner"),
    )

    rc = controller.main(
        [
            "--profile",
            str(profile_path),
            "--request",
            str(request_path),
            "--serving-config",
            str(config_path),
            "--phase",
            "mvex",
            "--attempt-id",
            "fixture-real-run",
            "--output-root",
            str(output_root),
            "--seeds",
            "7",
        ]
    )

    assert rc == 3
    failure = json.loads((output_root / "fixture-real-run" / "controller_failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "NON_EXECUTABLE_PROFILE"


def test_legacy_logical_view_profile_is_rejected_before_execution() -> None:
    with pytest.raises(controller.ProfileBuildError, match="legacy logical-view"):
        controller.validate_profile_capacity_semantics({"profile_status": "CALIBRATION"})


def test_mapping_preflight_failure_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    config_path = repo_root / "configs" / "experiments" / "controller.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config(), sort_keys=False), encoding="utf-8")
    evidence_path = repo_root / "results" / "fixture.json"
    evidence_path.parent.mkdir()
    evidence_path.write_text(json.dumps({"fixture": True}), encoding="utf-8")
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    evidence_path.with_suffix(".json.sha256").write_text(f"{evidence_sha256}\n", encoding="ascii")
    profile_path = repo_root / "profile.json"
    request_path = repo_root / "request.json"
    profile_path.write_text(json.dumps(profile(evidence_sha256=evidence_sha256)), encoding="utf-8")
    request_path.write_text(json.dumps(request()), encoding="utf-8")
    output_root = repo_root / "outputs"
    monkeypatch.setattr(
        controller,
        "get_git_state",
        lambda root, require_clean: {"commit": "b" * 40, "status": "", "clean": True},
    )
    monkeypatch.setattr(
        controller,
        "validate_deployment_mapping",
        lambda *args, **kwargs: (_ for _ in ()).throw(controller.ControllerError("mapping drift")),
    )

    rc = controller.main(
        [
            "--profile",
            str(profile_path),
            "--request",
            str(request_path),
            "--serving-config",
            str(config_path),
            "--phase",
            "mvex",
            "--attempt-id",
            "mapping-failure",
            "--output-root",
            str(output_root),
            "--seeds",
            "7",
            "--dry-run",
        ]
    )

    assert rc == 4
    attempt_dir = output_root / "mapping-failure"
    failure = json.loads((attempt_dir / "controller_failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "PREFLIGHT_FAILED"
    assert (attempt_dir / "decision.json").is_file()
