from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.eval.analyze_gsm8k_dependence import analyze_attempt, load_cells


def write_cell(
    attempt_dir: Path,
    allocation: str,
    seed: int,
    indices: list[int],
    hits: list[bool],
) -> None:
    record = {
        "status": "completed_validated",
        "bench": "gsm8k",
        "allocation": allocation,
        "seed": seed,
        "seed_semantics": "dataset sampling seed",
        "sampled_indices": indices,
        "num_samples": len(indices),
        "accuracy": sum(hits) / len(hits),
        "cases": [
            {"question": f"question-{item}", "hit": hit}
            for item, hit in zip(indices, hits)
        ],
    }
    path = attempt_dir / f"gsm8k__{allocation}__seed{seed}.json"
    path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}\n", encoding="ascii")


def dependence_fixture(tmp_path: Path) -> Path:
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    indices_by_seed = {1: [0, 1, 2], 2: [1, 2, 3], 3: [0, 2, 3]}
    baseline = {1: [False, True, False], 2: [True, False, True], 3: [True, False, False]}
    treatment = {1: [False, False, False], 2: [True, False, False], 3: [False, True, False]}
    for seed in indices_by_seed:
        write_cell(attempt_dir, "fp16", seed, indices_by_seed[seed], baseline[seed])
        write_cell(attempt_dir, "fp16_statebf16", seed, indices_by_seed[seed], treatment[seed])
    return attempt_dir


def test_dependence_analysis_reports_draw_and_cluster_units(tmp_path: Path) -> None:
    result = analyze_attempt(
        dependence_fixture(tmp_path),
        ["fp16", "fp16_statebf16"],
        [1, 2, 3],
        bootstrap_reps=1000,
        bootstrap_seed=7,
    )

    assert result["schema_version"] == 2
    assert result["diagnostics"]["n_seed_item_draws"] == 9
    assert result["diagnostics"]["n_unique_items"] == 4
    contrast = result["rows"][1]["cluster_robust_inference"]
    assert contrast["estimate"] == pytest.approx(-2 / 9)
    assert contrast["n_seed_item_draws"] == 9
    assert contrast["n_item_clusters"] == 4
    assert contrast["n_seed_clusters"] == 3
    assert contrast["degrees_of_freedom"] == 2


def test_dependence_analysis_rejects_tampered_sidecar(tmp_path: Path) -> None:
    attempt_dir = dependence_fixture(tmp_path)
    path = next(attempt_dir.glob("gsm8k__*.json"))
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        load_cells(attempt_dir, ["fp16", "fp16_statebf16"], [1, 2, 3])


def test_seed_iid_entrypoint_is_fail_closed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/eval/analyze_gsm8k_state3seed_v2.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "DEPRECATED" in result.stderr
    assert "analyze_gsm8k_dependence.py" in result.stderr
