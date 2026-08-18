"""Fail-closed verification for paper figures and the capacity matrix table."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
VECTOR_DIR = Path(__file__).resolve().parent / "vector_redesign"
DRAWIO_DIR = Path(__file__).resolve().parent / "drawio"
PLOT_SCRIPTS = [
    VECTOR_DIR / "make_vector_figures.py",
    VECTOR_DIR / "make_top_venue_figures.py",
]
NON_PLOT_SOURCES: set[str] = set()

EXPECTED_HASHES = {
    "results/verified/2026-08-14/capacity-phase-formal-corrected.analysis.json":
        "879dc059579eff231b71d8c4513ee856a904e66791df84f8e8889da82173dd02",
    "results/verified/2026-08-14/capacity-2x2-analysis-corrected.json":
        "97d6199a6a6bd6d3c9e211ff26b0f294393e67b331c31856beb73e967e5dd01a",
    "results/quality/gsm8k-state9seed-v2-dependence-aware-20260814.json":
        "564aecf99685172dc80a97a16471d9e6e990e4ee0082259e916555aac0325993",
    "results/quality/gsm8k-9b-state9seed-v2-dependence-aware-20260814.json":
        "d40fb7acd69d4196dac53e271f66d5732cafd01c29a4a070bdb7a8dd151bef54",
    "results/quality/ppl-stacking-analysis-20260809.json":
        "4e0b5f413167c0b8b6adba4fca2537427e8b496043302dfa1b784c6b2696aef5",
    "results/reproduction/2026-08-13/ruler-nothink/"
    "ruler-nothink-5cell-gate4-20260813/gate4_validation.json":
        "fb231af4945cfdaec12acdf0058db47f673f9d6e8b407e13c86a02368c8face7",
    "results/verified/2026-08-09/statebf16-serving-formal-analysis.json":
        "300b7d45f7ddd53c46c6c4355c7ec870c33789e85a180ff65212e9b4ebe7b8b9",
    "results/verified/2026-08-09/statebf16-serving-repro-analysis.json":
        "b4e9c7bce07662885d371ad2d9c3c1a5fdca575be172891aeb2953ac829e6b07",
    "results/quality/serving-direction/serving-direction-agreement-20260811.json":
        "ed4a4a66e34477110ac76437e4301441915b3d455f65ab009500b58642db9df0",
    "results/reproduction/2026-08-13/m4-four-config/gate4-r3/"
    "m4_gate4_validation.json":
        "7933182cb23e44c2d4b24277ba52fb3a6e14ccbf1320e155f1c4e60c447d1201",
    "results/quality/state-sensitivity-analysis-20260809-bonf.json":
        "5b09608d4f6af9ac92a4582f23934ca334fc79fda48ba2074d86c6f07670d2dc",
    "results/verified/2026-08-14/controller-decisions/selector-audit.json":
        "dab8f055a047c5eeefb01545518e95568b70ce94577a194d33cf2ddb2ba04848",
    "results/quality/chunk-ablation/chunk-ablation-20260809__statefp32__chunk128__2b.csv":
        "00f7795e1844ca44151411d494ab295239b7b2bfab0f75ee77033ee5c946a1a1",
    "results/quality/chunk-ablation/chunk-ablation-20260809__statefp32__chunk1__2b.csv":
        "f468ef0dcb7f849965299fd6c763238589f29f6d46f26c7d2d6bee30c5884d29",
    "results/quality/chunk-ablation/chunk-ablation-20260809__statebf16__chunk128__2b.csv":
        "8a11bb31d70984cf15db0c7ca71c7ec8683c8cccb6ba4bf20dababe358e4bd4d",
    "results/quality/chunk-ablation/chunk-ablation-20260809__statebf16__chunk1__2b.csv":
        "1dcab697280ac33ffbd12be8e4a65ed11a4e4d943618c0d77c71382b00fc4e7a",
}


def sha256_file(path: Path) -> str:
    # Git stores these tracked text artifacts with LF. Normalize stale Windows
    # worktrees so the verification contract is identical in every clone.
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def load_json(relative_path: str) -> dict[str, Any]:
    value = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict), f"JSON root is not an object: {relative_path}"
    return value


def assert_close(observed: float, expected: float, *, atol: float = 1e-8) -> None:
    assert math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=atol), (
        f"numeric mismatch: observed={observed!r}, expected={expected!r}"
    )


def verify_hashes() -> None:
    for relative_path, expected in EXPECTED_HASHES.items():
        path = ROOT / relative_path
        assert path.is_file(), f"missing frozen figure source: {relative_path}"
        observed = sha256_file(path)
        assert observed == expected, (
            f"SHA-256 mismatch for {relative_path}: observed={observed}, expected={expected}"
        )
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.is_file():
            assert sidecar.read_text(encoding="ascii").strip() == expected, (
                f"sidecar mismatch for {relative_path}"
            )


def verify_capacity() -> None:
    full = load_json(
        "results/verified/2026-08-14/capacity-phase-formal-corrected.analysis.json"
    )
    assert full["attempt"] == "capacity-phase-formal-20260811"
    assert full["n_cells"] == 112
    assert len(full["rows"]) == 52
    assert len(full["float16_state_frontier"]) == 8
    gains = [
        100.0 * (row["bf16_state_tokens"] / row["fp32_state_tokens"] - 1.0)
        for row in full["rows"]
    ]
    assert all(gain > 0.0 for gain in gains)
    assert_close(statistics.median(gains), 15.44, atol=5e-3)
    residuals = [float(row["prediction_residual_pct"]) for row in full["rows"]]
    assert_close(min(residuals), -5.1604)
    assert_close(max(residuals), 13.2134)
    assert full["prediction_residual_summary"] | {"interpretation": None} == {
        "n_pairs": 52,
        "median_absolute_pct": 2.3795,
        "mean_absolute_pct": 2.7955,
        "max_absolute_pct": 13.2134,
        "interpretation": None,
    }

    table_path = ROOT / "paper/mlsys2026/capacity_matrix_table.tex"
    table = table_path.read_text(encoding="utf-8")
    table_sidecar = table_path.with_suffix(table_path.suffix + ".sha256")
    assert table_sidecar.read_text(encoding="ascii").strip() == sha256_file(table_path)
    assert "Source SHA-256: 879dc059579eff231b71d8c4513ee856" in table
    observed_rows = [
        line for line in table.splitlines() if line.startswith(("2B &", "9B &"))
    ]
    expected_rows = []
    for row in full["rows"]:
        fp32_tokens = int(row["fp32_state_tokens"])
        bf16_tokens = int(row["bf16_state_tokens"])
        gain_pct = 100.0 * (bf16_tokens / fp32_tokens - 1.0)
        expected_rows.append(
            f"{str(row['model']).upper()} & {row['kv_dtype']} & {int(row['length']):,} & "
            f"{float(row['gpu_memory_utilization']):.2f} & "
            f"{int(row['fp32_num_gpu_blocks']):,} & {int(row['bf16_num_gpu_blocks']):,} & "
            f"{fp32_tokens:,} & {bf16_tokens:,} & {gain_pct:.2f}\\% & "
            f"{float(row['prediction_residual_pct']):+.2f}\\% \\\\"
        )
    assert observed_rows == expected_rows, "supplementary 52-row capacity table drifted"

    cap = load_json("results/verified/2026-08-14/capacity-2x2-analysis-corrected.json")
    assert len(cap["rows"]) == 7
    assert cap["model_parameters"] == {
        "2b": {"A_f": 12288.0, "A_q": 3168.0, "G_fp32": 19537920.0, "G_bf16": 10100736.0},
        "9b": {"A_f": 16384.0, "A_q": 4224.0, "G_fp32": 26050560.0, "G_bf16": 13467648.0},
    }
    assert cap["layout_accounting"]["int4_per_attention_layer_bytes_per_token"] == 528
    assert cap["layout_accounting"]["state_per_gdn_layer_bytes"] == {
        "fp32_temporal_plus_bf16_conv": 1085440,
        "bf16_temporal_plus_bf16_conv": 561152,
    }
    assert cap["missing_cells"] == [
        {
            "model": "9b",
            "length": 16384,
            "kv_dtype": "fp16",
            "reason": "not probed in the frozen gpu_memory_utilization=0.85 fp16 attempt",
        }
    ]
    by = {(row["model"], row["length"], row["kv_dtype"]): row for row in cap["rows"]}
    representative = by[("2b", 4096, "int4")]
    assert representative["fp32_state_tokens"] == 2692710
    assert representative["bf16_state_tokens"] == 3703954
    assert representative["fp32_num_gpu_blocks"] == 3287
    assert representative["bf16_num_gpu_blocks"] == 6330
    assert representative["fp32_block_size"] == 2064
    assert representative["bf16_block_size"] == 1072
    assert_close(representative["measured_r_state"], 1.3755, atol=5e-5)
    assert_close(representative["predicted_r_state"], 1.4089, atol=5e-5)
    assert_close(representative["signed_gap_pct"], -2.37, atol=5e-3)
    gaps = [row["signed_gap_pct"] for row in cap["rows"]]
    assert min(gaps) == -3.24 and max(gaps) == 2.46


def verify_gsm8k() -> None:
    g2b = load_json("results/quality/gsm8k-state9seed-v2-dependence-aware-20260814.json")
    g9b = load_json("results/quality/gsm8k-9b-state9seed-v2-dependence-aware-20260814.json")
    assert g2b["schema_version"] == g9b["schema_version"] == 2
    assert g2b["diagnostics"] | {"allocation_outcome_stability": None} == {
        "n_seed_item_draws": 1800,
        "n_unique_items": 1017,
        "pairwise_seed_overlap_min": 19,
        "pairwise_seed_overlap_median": 30.0,
        "pairwise_seed_overlap_max": 39,
        "allocation_outcome_stability": None,
    }
    rows2 = {row["allocation"]: row for row in g2b["rows"]}
    expected = {
        "fp16_statebf16": (-0.01, [-0.02044313, 0.00044313], 0.0582485),
        "uniform_int4": (-0.02722222, [-0.05275141, -0.00169303], 0.03938427),
        "uniform_int4_statebf16": (-0.01555556, [-0.04446884, 0.01335773], 0.24988483),
    }
    for allocation, (estimate, interval, p_value) in expected.items():
        row = rows2[allocation]
        assert_close(row["delta_vs_fp16"], estimate)
        assert row["ci95_vs_fp16"] == interval
        inference = row["cluster_robust_inference"]
        assert_close(inference["p_value"], p_value)
        assert inference["n_seed_item_draws"] == 1800
        assert inference["n_item_clusters"] == 1017
        assert inference["n_seed_clusters"] == 9
        assert inference["degrees_of_freedom"] == 8
    assert g2b["stacking_marginal"]["ci95"] == [-0.0098717, 0.03320503]
    state_sensitivity = rows2["fp16_statebf16"]["item_equal_weight_sensitivity"]
    assert_close(state_sensitivity["estimate"], -0.00778433)
    assert state_sensitivity["gain_items"] == 29
    assert state_sensitivity["loss_items"] == 37
    row9 = next(row for row in g9b["rows"] if row["allocation"] == "fp16_statebf16")
    assert_close(row9["delta_vs_fp16"], 0.00333333)
    assert row9["ci95_vs_fp16"] == [-0.00137791, 0.00804458]


def verify_secondary_panels() -> None:
    ppl = load_json("results/quality/ppl-stacking-analysis-20260809.json")
    assert ppl["tables"]["c4"]["delta_bf16_vs_fp32"] == -0.0029
    assert ppl["tables"]["c4"]["ci95_delta"] == [-0.0129, 0.0072]
    assert ppl["tables"]["pg19"]["delta_bf16_vs_fp32"] == 0.0065
    assert ppl["tables"]["pg19"]["ci95_delta"] == [-0.0447, 0.0578]

    ruler = load_json(
        "results/reproduction/2026-08-13/ruler-nothink/"
        "ruler-nothink-5cell-gate4-20260813/gate4_validation.json"
    )
    assert len(ruler["statistical_findings"]) == 5
    assert all(row["mean_delta_accuracy_points"] == 0.0 for row in ruler["statistical_findings"])
    assert all(row["ci95_delta_accuracy_points"] == [0.0, 0.0] for row in ruler["statistical_findings"])

    direction = load_json(
        "results/quality/serving-direction/serving-direction-agreement-20260811.json"
    )
    summary = direction["input_summary"]
    assert summary["n_cells_per_run"] == 60
    assert summary["bh_formal_n_q_lt_0_05"] == 0
    assert summary["bh_repro_n_q_lt_0_05"] == 0

    gate4 = load_json(
        "results/reproduction/2026-08-13/m4-four-config/gate4-r3/"
        "m4_gate4_validation.json"
    )
    assert gate4["same_seed_second_formal_run"] is True
    assert gate4["independent_replication"] is False
    assert gate4["reproducibility_verdict"] == "NOT_REPRODUCIBLE"
    comparison = gate4["comparison"]
    assert comparison["within_tolerance"] == 537
    assert comparison["outside_tolerance"] == 183
    assert comparison["boundary_points_exact"] == 713
    assert comparison["boundary_points_total"] == 720
    assert comparison["boundaries_exact"] == 38
    assert comparison["boundaries_total"] == 40

    sensitivity = load_json("results/quality/state-sensitivity-analysis-20260809-bonf.json")
    layer_rows = [row for row in sensitivity["rows"] if row["config"].startswith("bf16_L")]
    assert len(layer_rows) == 18
    assert sum(bool(row["c4_sensitive"]) for row in layer_rows) == 2
    assert sum(bool(row["pg19_sensitive"]) for row in layer_rows) == 0
    assert not any(bool(row.get("c4_bonf_significant")) for row in layer_rows)
    assert not any(bool(row.get("c4_bh_significant")) for row in layer_rows)

    expected_chunks = {
        ("fp32", 128): 19.3480,
        ("fp32", 1): 36.1643,
        ("bf16", 128): 19.3498,
        ("bf16", 1): 36.1347,
    }
    for (state, chunk), expected in expected_chunks.items():
        path = ROOT / (
            "results/quality/chunk-ablation/"
            f"chunk-ablation-20260809__state{state}__chunk{chunk}__2b.csv"
        )
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert_close(float(rows[0]["ppl_mean"]), expected, atol=5e-5)


def verify_plot_contract_and_outputs() -> None:
    script = "\n".join(path.read_text(encoding="utf-8") for path in PLOT_SCRIPTS)
    for relative_path in EXPECTED_HASHES:
        if (
            relative_path.endswith(".json")
            and "statebf16-serving" not in relative_path
            and relative_path not in NON_PLOT_SOURCES
        ):
            assert Path(relative_path).name in script, (
                f"plot script does not reference {relative_path}"
            )
    assert "capacity-2x2-analysis.json" not in script
    assert "gsm8k-state9seed-v2-analysis-20260809.json" not in script
    assert "independent repro" not in script
    assert "temporal rerun" in script
    assert "mean_accuracy_over_seed_item_draws" in script

    for index in range(1, 9):
        prefix = next(VECTOR_DIR.glob(f"fig{index}_*.svg"), None)
        assert prefix is not None, f"missing SVG for figure source {index}"
        svg = prefix.read_text(encoding="utf-8")
        assert "<text" in svg, f"SVG text was converted to paths: {prefix.name}"
        assert "<image" not in svg, f"SVG embeds a raster image: {prefix.name}"
        pdf = prefix.with_suffix(".pdf")
        assert pdf.is_file() and pdf.stat().st_size > 10_000, f"invalid PDF: {pdf.name}"
        assert pdf.read_bytes().startswith(b"%PDF-"), f"invalid PDF header: {pdf.name}"

    expected_drawio = {
        "fig1_hybrid_allocator": [
            "bytes / sequence = A L + G",
            "allocator-equivalent sequence slots",
            "Not measured: scheduler admission or SLO completion",
        ],
        "fig2_discrete_allocator": [
            "1,085,440 B / GDN layer",
            "561,152 B / GDN layer",
            "657.4",
            "904.3",
            "not concurrent requests",
        ],
    }
    for stem, required_text in expected_drawio.items():
        source = DRAWIO_DIR / f"{stem}.drawio"
        svg = DRAWIO_DIR / f"{stem}.svg"
        pdf = DRAWIO_DIR / f"{stem}.pdf"
        assert source.is_file(), f"missing Draw.io source: {source.name}"
        source_text = source.read_text(encoding="utf-8")
        assert source_text.startswith("<mxfile"), f"invalid Draw.io XML: {source.name}"
        root = ET.fromstring(source_text)
        displayed = " ".join(
            re.sub(r"<[^>]+>", " ", html.unescape(cell.get("value", "")))
            for cell in root.iter("mxCell")
        )
        displayed = " ".join(displayed.split())
        for text in required_text:
            assert text in displayed, f"missing {text!r} from {source.name}"
        assert svg.is_file() and "<text" in svg.read_text(encoding="utf-8")
        assert "<image" not in svg.read_text(encoding="utf-8")
        assert pdf.is_file() and pdf.read_bytes().startswith(b"%PDF-")


def main() -> int:
    verify_hashes()
    verify_capacity()
    verify_gsm8k()
    verify_secondary_panels()
    verify_plot_contract_and_outputs()
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_hashes_verified": len(EXPECTED_HASHES),
                "vector_figure_pairs_verified": 8,
                "capacity_source": "corrected-20260814",
                "gsm8k_inference": "item-and-seed two-way CR1",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
