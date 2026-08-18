"""Validate the capacity model for GDN state dtype under uniform int4.

Reads probe_ssm_state_dtype.py outputs and compares measured capacity ratios
(bf16/fp32, fp16/fp32) against the paper's model

    r_state(L) = (A_q * L + G_fp32) / (A_q * L + G_bf16)

where G_bf16 halves only the temporal state (conv state stays bf16).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Paper §3.3 parameters (uniform int4 per-token-head attention KV).
# 2B: A_q = 6 layers x 528 B/token = 3168; G = 18 x 1,085,440 B.
# 9B: A_q = 8 layers x 528 B/token; G = 24 x 1,085,440 B.
# G_bf16 = layers x (temporal bf16 524,288 + conv bf16 36,864) = layers x 561,152.
MODEL_PARAMS = {
    "2b": {"A_q": 3168.0, "G_fp32": 19_537_920.0, "G_bf16": 18 * 561_152.0},
    "9b": {"A_q": 8 * 528.0, "G_fp32": 24 * 1_085_440.0, "G_bf16": 24 * 561_152.0},
}


def predicted_ratio(model: str, length: int) -> float:
    p = MODEL_PARAMS[model]
    num = p["A_q"] * length + p["G_fp32"]
    den = p["A_q"] * length + p["G_bf16"]
    return num / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/verified/2026-08-08/capacity-state")
    ap.add_argument("--attempt", default="capacity-state-20260808")
    ap.add_argument("--out", default="results/verified/2026-08-08/capacity-state-analysis.json")
    args = ap.parse_args()
    probe_dir = Path(args.dir)

    cells: dict[tuple[str, str, int], dict] = {}
    for path in sorted(probe_dir.glob(f"{args.attempt}__*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        tag = path.name[len(args.attempt) + 2 : -len(".json")]
        parts = tag.split("__")
        if len(parts) != 3:
            raise SystemExit(f"unexpected filename: {path.name}")
        model, dtype, length_s = parts
        if model not in ("2b", "9b") or dtype not in ("auto", "bfloat16", "float16"):
            continue
        cells[(model, dtype, int(length_s.replace("L", "")))] = rec

    expected = {
        (m, d, l)
        for m in ("2b", "9b")
        for d in ("auto", "bfloat16", "float16")
        for l in (4096,)
    } | {
        (m, d, l)
        for m in ("2b", "9b")
        for d in ("auto", "bfloat16")
        for l in (16384,)
    }
    missing = sorted(expected - set(cells))
    if missing:
        raise SystemExit(f"incomplete probes: {missing}")

    rows = []
    for model in ("2b", "9b"):
        for length in (4096, 16384):
            fp32 = cells[(model, "auto", length)]["capacity"]["tokens"]
            bf16 = cells[(model, "bfloat16", length)]["capacity"]["tokens"]
            row = {
                "model": model,
                "length": length,
                "fp32_capacity_tokens": fp32,
                "bf16_capacity_tokens": bf16,
                "measured_ratio_bf16": round(bf16 / fp32, 4),
                "predicted_ratio": round(predicted_ratio(model, length), 4),
                "gap_pct": round((bf16 / fp32 / predicted_ratio(model, length) - 1.0) * 100.0, 2),
            }
            if length == 4096:
                fp16 = cells[(model, "float16", length)]["capacity"]["tokens"]
                row["fp16_capacity_tokens"] = fp16
                row["measured_ratio_fp16"] = round(fp16 / fp32, 4)
            rows.append(row)

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "protocol": (
            "probe_ssm_state_dtype.py, uniform int4 (kv_cache_dtype_per_layer={}), "
            "gpu_memory_utilization=0.85; auto resolves to fp32 state"
        ),
        "model": "r_state(L) = (A_q L + G_fp32) / (A_q L + G_bf16), "
                 "A_q/G from paper mainline §3.3",
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
