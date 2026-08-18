from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.bench.run_steady_state import (
    ExperimentError,
    ServerSession,
    build_sample_plan,
    load_config,
    resolve_phase,
)
from scripts.controller.run_joint_precision_controller import validate_deployment_mapping

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "joint_precision_controller_2b.yaml"
PROFILE_PATH = ROOT / "configs" / "controller" / "fixtures" / "joint_precision_gate0_profile.json"
REQUEST_PATH = ROOT / "configs" / "controller" / "fixtures" / "joint_precision_gate0_request.json"
CALIBRATION_CONTRACT_PATH = (
    ROOT / "configs" / "controller" / "contracts" / "joint_precision_four_config_calibration_r1_20260811.json"
)


def test_four_allocation_config_has_exact_orthogonal_precision_controls() -> None:
    config = load_config(CONFIG_PATH)
    expected = {
        "full": ("auto", "float32"),
        "kv_only": ("int4_per_token_head", "float32"),
        "state_only": ("auto", "bfloat16"),
        "joint": ("int4_per_token_head", "bfloat16"),
    }

    for allocation, (kv_dtype, state_dtype) in expected.items():
        args = config["allocations"][allocation]["server_args"]
        assert args[args.index("--kv-cache-dtype") + 1] == kv_dtype
        assert args[args.index("--mamba-ssm-cache-dtype") + 1] == state_dtype
        assert "--tensor-parallel-size" not in args


def test_four_allocation_log_proofs_are_exact_and_cross_reject(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    controls = {
        "full": ("auto", "float32"),
        "kv_only": ("int4_per_token_head", "float32"),
        "state_only": ("auto", "bfloat16"),
        "joint": ("int4_per_token_head", "bfloat16"),
    }

    session_dirs: dict[str, Path] = {}
    for allocation, (kv_dtype, state_dtype) in controls.items():
        expected_patterns = [
            f"'mamba_ssm_cache_dtype': '{state_dtype}'",
            f"kv_cache_dtype={kv_dtype}",
            "CUDAGraphMode.PIECEWISE",
        ]
        assert config["allocations"][allocation]["required_log_substrings"] == expected_patterns
        assert "Using the user-specified value" not in expected_patterns

        session_dir = tmp_path / allocation
        session_dir.mkdir()
        (session_dir / "server.log").write_text(
            "non-default args: "
            f"{{'mamba_ssm_cache_dtype': '{state_dtype}', "
            "'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>}}\n"
            f"resolved engine config: kv_cache_dtype={kv_dtype}, device=cuda\n",
            encoding="utf-8",
        )
        session_dirs[allocation] = session_dir

    for allocation in controls:
        session = ServerSession.__new__(ServerSession)
        session.config = config
        session.allocation_name = allocation
        session.session_dir = session_dirs[allocation]
        session._verify_log_patterns()

        for other_allocation in controls.keys() - {allocation}:
            session.session_dir = session_dirs[other_allocation]
            with pytest.raises(ExperimentError, match="server log does not prove allocation"):
                session._verify_log_patterns()


def test_confirmatory_matrix_uses_disjoint_seeds_and_has_full_denominator() -> None:
    config = load_config(CONFIG_PATH)
    calibration = resolve_phase(config, "calibration")
    confirmatory = resolve_phase(config, "confirmatory")

    assert set(calibration["seeds"]).isdisjoint(confirmatory["seeds"])
    assert len(build_sample_plan(config, calibration)) == 144
    assert len(build_sample_plan(config, confirmatory)) == 144


def test_calibration_contract_matches_runner_plan_and_launcher() -> None:
    contract = json.loads(CALIBRATION_CONTRACT_PATH.read_text(encoding="utf-8"))
    sidecar = CALIBRATION_CONTRACT_PATH.with_suffix(".json.sha256").read_text(encoding="ascii").strip()
    assert hashlib.sha256(CALIBRATION_CONTRACT_PATH.read_bytes()).hexdigest() == sidecar

    code_entries = (
        ("config_path", "config_sha256"),
        ("runner_path", "runner_sha256"),
        ("launcher_path", "launcher_sha256"),
    )
    for path_key, hash_key in code_entries:
        source = ROOT / contract["code"][path_key]
        assert source.is_file()
        assert len(contract["code"][hash_key]) == 64
        assert all(char in "0123456789abcdef" for char in contract["code"][hash_key])

    # The runner evolves after a frozen experiment. Its recorded hash remains
    # provenance; current runner behavior is checked by the plan assertions below.
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("launcher_path", "launcher_sha256"),
    ):
        source = ROOT / contract["code"][path_key]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == contract["code"][hash_key]

    config = load_config(ROOT / contract["code"]["config_path"])
    plan = build_sample_plan(config, resolve_phase(config, "calibration"))
    canonical_plan = json.dumps(
        plan,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    sample_ids = [row["sample_id"] for row in plan]
    canonical_ids = json.dumps(sample_ids, ensure_ascii=True, separators=(",", ":")).encode()
    matrix = contract["matrix"]
    assert len(plan) == matrix["expected_samples"] == 144
    assert sum(row["num_prompts"] for row in plan) == matrix["expected_measurement_requests"] == 320400
    assert len(plan) * config["protocol"]["warmup_requests"] == matrix["expected_warmup_requests"] == 17280
    assert hashlib.sha256(canonical_plan).hexdigest() == matrix["plan_sha256"]
    assert hashlib.sha256(canonical_ids).hexdigest() == matrix["sample_ids_sha256"]

    launcher = (ROOT / contract["code"]["launcher_path"]).read_text(encoding="utf-8")
    assert "--phase calibration" in launcher
    assert '--parent-attempt "${PARENT_ATTEMPT}"' in launcher
    assert "timeout --signal=TERM --kill-after=30s 28800" in launcher
    assert "--resume" not in launcher
    assert all(
        option not in launcher for option in ("--max-samples", "--allocations", "--workloads", "--seeds", "--rates")
    )


def test_gate0_fixture_maps_all_four_decisions_to_executable_commands() -> None:
    config = load_config(CONFIG_PATH)
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))

    for candidate in profile["candidates"]:
        decision = {
            "status": "SELECTED",
            "selected": {
                "config_id": candidate["config_id"],
                "deployment": candidate["deployment"],
            },
        }
        mapping = validate_deployment_mapping(config, decision, request, repo_root=ROOT)
        assert mapping["allocation"] == candidate["config_id"]
