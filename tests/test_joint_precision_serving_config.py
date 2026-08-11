from __future__ import annotations

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
