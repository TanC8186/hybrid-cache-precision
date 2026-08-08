"""Aggregate M3/M4 capacity probes and compare against the VERIFIED 2B gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/verified/2026-08-08/capacity-probe-extra")
    ap.add_argument("--out", default="results/verified/2026-08-08/capacity-probe-extra-analysis.json")
    args = ap.parse_args()

    root = Path(args.dir)
    cells: dict[str, dict] = {}
    for path in root.glob("*.json"):
        if path.name.endswith("-analysis.json"):
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        cells[path.stem] = rec

    expected = ["9b_legacy", "9b_uniform", "9b_packed", "attn_fp16", "attn_int4"]
    present = [name for name in expected if name in cells]
    missing = [name for name in expected if name not in cells]
    if not present:
        raise SystemExit("no probes found")

    def cap(name: str) -> float:
        return float(cells[name]["capacity"]["tokens"])

    def conc(name: str) -> float:
        return float(cells[name]["capacity"]["max_concurrency"])

    ratios: dict[str, float] = {}
    if {"9b_packed", "9b_legacy"} <= set(present):
        ratios["9b_packed_over_legacy"] = round(cap("9b_packed") / cap("9b_legacy"), 4)
    if {"9b_packed", "9b_uniform"} <= set(present):
        ratios["9b_packed_over_uniform"] = round(cap("9b_packed") / cap("9b_uniform"), 4)
    if {"9b_uniform", "9b_legacy"} <= set(present):
        ratios["9b_uniform_over_legacy"] = round(cap("9b_uniform") / cap("9b_legacy"), 4)
    if {"attn_int4", "attn_fp16"} <= set(present):
        ratios["attn_int4_over_fp16"] = round(cap("attn_int4") / cap("attn_fp16"), 4)

    result = {
        "schema_version": 1,
        "protocol": "inspect_kv_config.py, max_model_len=4096, gpu_memory_utilization=0.85, seed=42, enforce_eager",
        "missing_probes": missing,
        "probes": {
            name: {
                "tokens": cap(name),
                "max_concurrency": conc(name),
                "model": cells[name]["environment"]["model_path"],
                "cache_dtype": cells[name]["cache_config"]["cache_dtype"],
                "per_layer": cells[name]["cache_config"].get("kv_cache_dtype_per_layer"),
                "packed_flag": bool(cells[name]["cache_config"].get("enable_per_layer_page_groups")),
            }
            for name in present
        },
        "ratios": ratios,
        "reference_2b_gate": {
            "legacy_tokens": 705_604,
            "uniform_tokens": 2_736_947,
            "packed_tokens": 2_280_448,
            "packed_over_legacy": 3.232,
            "packed_over_uniform": 0.833,
            "hybrid_2b_int4_over_fp16": 2.245,
            "pure_attention_mechanism_ratio": 3.88,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
