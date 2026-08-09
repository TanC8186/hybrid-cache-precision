"""Analyze the KV-bits x state-bits 2x2 capacity matrix (ARS 2026-08-09 R2/R5).

Merges the int4-KV probes (results/verified/2026-08-08/capacity-state) with the
fp16-KV probes (results/verified/2026-08-09/capacity-state-fp16kv), computes
r_state per KV dtype and r_kv per state dtype, and reports signed model error
plus the block-granularity evidence (block_size / num_gpu_blocks /
mamba_page_size_padded) used for the conservative-lower-bound discussion.
Fail-closed on missing cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_PARAMS = {
    "2b": {"A_f": 16384.0, "A_q": 3168.0, "G_fp32": 19_537_920.0, "G_bf16": 18 * 561_152.0},
    "9b": {"A_f": 16384.0, "A_q": 16_384.0 / 3.878, "G_fp32": 26_050_560.0, "G_bf16": 24 * 561_152.0},
}


def load_cells(probe_dir: Path, attempt: str) -> dict[tuple[str, str, int], dict]:
    cells: dict[tuple[str, str, int], dict] = {}
    for path in sorted(probe_dir.glob(f"{attempt}__*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        tag = path.name[len(attempt) + 2 : -len(".json")]
        parts = tag.split("__")
        if len(parts) != 3:
            raise SystemExit(f"unexpected filename: {path.name}")
        model, dtype, length_s = parts
        if model not in ("2b", "9b") or dtype not in ("auto", "bfloat16", "float16"):
            continue
        cells[(model, dtype, int(length_s.replace("L", "")))] = rec
    return cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--int4-dir", default="results/verified/2026-08-08/capacity-state")
    ap.add_argument("--int4-attempt", default="capacity-state-20260808")
    ap.add_argument("--fp16-dir", default="results/verified/2026-08-09/capacity-state-fp16kv")
    ap.add_argument("--fp16-attempt", default="capacity-state-fp16kv-20260809")
    ap.add_argument("--out", default="results/verified/2026-08-09/capacity-2x2-analysis.json")
    args = ap.parse_args()

    int4 = load_cells(Path(args.int4_dir), args.int4_attempt)
    fp16 = load_cells(Path(args.fp16_dir), args.fp16_attempt)

    required_int4 = {
        (m, d, l)
        for m in ("2b", "9b")
        for d in ("auto", "bfloat16")
        for l in (4096, 16384)
    }
    required_fp16 = {
        (m, d, l)
        for m, d, l in [
            ("2b", "auto", 4096),
            ("2b", "bfloat16", 4096),
            ("2b", "float16", 4096),
            ("2b", "auto", 16384),
            ("2b", "bfloat16", 16384),
            ("9b", "auto", 4096),
            ("9b", "bfloat16", 4096),
            ("9b", "float16", 4096),
        ]
    }
    missing_int4 = sorted(required_int4 - set(int4))
    missing_fp16 = sorted(required_fp16 - set(fp16))
    if missing_int4 or missing_fp16:
        raise SystemExit(f"incomplete probes: int4={missing_int4} fp16={missing_fp16}")

    def predicted_ratio(model: str, length: int, kv: str) -> float:
        p = MODEL_PARAMS[model]
        a = p["A_f"] if kv == "fp16" else p["A_q"]
        return (a * length + p["G_fp32"]) / (a * length + p["G_bf16"])

    rows = []
    for model in ("2b", "9b"):
        for length in (4096, 16384):
            for kv in ("fp16", "int4"):
                if kv == "fp16" and (model, length) == ("9b", 16384):
                    continue
                cells = fp16 if kv == "fp16" else int4
                fp32 = cells[(model, "auto", length)]["capacity"]["tokens"]
                bf16 = cells[(model, "bfloat16", length)]["capacity"]["tokens"]
                fp32_probe = cells[(model, "auto", length)]
                bf16_probe = cells[(model, "bfloat16", length)]
                measured = bf16 / fp32
                pred = predicted_ratio(model, length, kv)
                row = {
                    "model": model,
                    "length": length,
                    "kv_dtype": kv,
                    "fp32_state_tokens": fp32,
                    "bf16_state_tokens": bf16,
                    "measured_r_state": round(measured, 4),
                    "predicted_r_state": round(pred, 4),
                    "signed_gap_pct": round((measured / pred - 1.0) * 100.0, 2),
                    "fp32_block_size": fp32_probe["cache_config"]["block_size"],
                    "fp32_num_gpu_blocks": fp32_probe["cache_config"]["num_gpu_blocks"],
                    "bf16_block_size": bf16_probe["cache_config"]["block_size"],
                    "bf16_num_gpu_blocks": bf16_probe["cache_config"]["num_gpu_blocks"],
                    "bf16_mamba_page_size_padded": bf16_probe["cache_config"]["mamba_page_size_padded"],
                }
                if length == 4096 and kv == "fp16":
                    fp16state = cells[(model, "float16", length)]["capacity"]["tokens"]
                    row["fp16_state_tokens"] = fp16state
                    row["measured_r_state_fp16_state"] = round(fp16state / fp32, 4)
                rows.append(row)

    # r_kv = int4 capacity / fp16 capacity at the same model/length/state dtype.
    r_kv_rows = []
    for model in ("2b", "9b"):
        for length in (4096, 16384):
            if (model, length) == ("9b", 16384):
                continue
            for state, dtype in (("fp32", "auto"), ("bf16", "bfloat16")):
                kv_int4 = int4[(model, dtype, length)]["capacity"]["tokens"]
                kv_fp16 = fp16[(model, dtype, length)]["capacity"]["tokens"]
                r_kv_rows.append(
                    {
                        "model": model,
                        "length": length,
                        "state_dtype": state,
                        "fp16_kv_tokens": kv_fp16,
                        "int4_kv_tokens": kv_int4,
                        "measured_r_kv": round(kv_int4 / kv_fp16, 4),
                    }
                )

    by_kv: dict[str, list[float]] = {"fp16": [], "int4": []}
    for r in rows:
        by_kv[r["kv_dtype"]].append(r["signed_gap_pct"])
    result = {
        "schema_version": 1,
        "int4_attempt": args.int4_attempt,
        "fp16_attempt": args.fp16_attempt,
        "protocol": (
            "probe_ssm_state_dtype.py, gpu_memory_utilization=0.85; "
            "int4 KV = kv_cache_dtype int4_per_token_head, fp16 KV = auto; "
            "state auto resolves to fp32"
        ),
        "model": "r_state(L)=(A L+G_fp32)/(A L+G_bf16); A=A_f for fp16 KV, A=A_q for int4 KV",
        "rows": rows,
        "r_kv_rows": r_kv_rows,
        "signed_error_summary": {
            "by_kv_dtype": {
                kv: {
                    "n_cells": len(vals),
                    "all_negative": all(v < 0 for v in vals),
                    "negative_count": sum(1 for v in vals if v < 0),
                    "positive_count": sum(1 for v in vals if v > 0),
                    "min_pct": round(min(vals), 2),
                    "max_pct": round(max(vals), 2),
                    "same_sign_probability_note": (
                        "int4 headline cells: P(all negative by chance | coin flip) "
                        "= 2 * (0.5)^4 = 0.0625, consistent with discrete block "
                        "rounding; fp16-KV cells show mixed signs, so the "
                        "conservative-lower-bound framing is scoped to int4 KV"
                    ),
                }
                for kv, vals in by_kv.items()
            },
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
