from __future__ import annotations

from scripts.bench.build_capacity_matrix_table import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    format_row,
    load_analysis,
    render_table,
)


def test_format_row_reports_blocks_tokens_gain_and_residual() -> None:
    row = {
        "model": "2b",
        "kv_dtype": "int4",
        "length": 4096,
        "gpu_memory_utilization": 0.8,
        "fp32_num_gpu_blocks": 3287,
        "bf16_num_gpu_blocks": 6330,
        "fp32_state_tokens": 2692710,
        "bf16_state_tokens": 3703954,
        "prediction_residual_pct": -2.379,
    }

    assert format_row(row) == (
        "2B & int4 & 4,096 & 0.80 & 3,287 & 6,330 & 2,692,710 & "
        "3,703,954 & 37.55\\% & -2.38\\% \\\\"
    )


def test_frozen_matrix_table_is_complete_and_current() -> None:
    data, digest = load_analysis(DEFAULT_SOURCE)
    rendered = render_table(data, DEFAULT_SOURCE, digest)

    assert digest == "879dc059579eff231b71d8c4513ee856a904e66791df84f8e8889da82173dd02"
    assert len([line for line in rendered.splitlines() if line.startswith(("2B &", "9B &"))]) == 52
    assert DEFAULT_OUTPUT.read_text(encoding="utf-8") == rendered
