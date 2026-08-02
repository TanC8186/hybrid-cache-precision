"""从逐层敏感度生成 per-layer KV dtype 分配（灵敏度引导分配）。

从 results/ablations/layer_sensitivity.csv 读敏感度，按阈值决定每层 dtype：
- 敏感度 > threshold → 'float16'（保护敏感层，不量化）
- 否则 → 'int4_per_token_head'（默认量化）

这是 MVP 路径（用 vLLM 现有 int4 内核 + per-layer 保护）。2/3-bit 扩展留到内核支持后。

用法:
  python scripts/exp/gen_allocation.py --csv results/ablations/layer_sensitivity.csv --threshold 0.15
输出: 打印 {layer_idx: dtype} 映射 + 保存 JSON
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/ablations/layer_sensitivity.csv")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="敏感度超过此值则保护（float16），否则 int4")
    ap.add_argument("--quant-dtype", default="int4_per_token_head", help="非保护层的量化 dtype")
    ap.add_argument("--out", default="results/ablations/allocation.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    # layer_sensitivity.csv: config, layer_idx, ppl, kv_quant_bytes, kv_fp16_bytes
    # 敏感度 = (ppl - ppl_8bit) / (ppl_2bit - ppl_8bit)，从 all_8bit 与 all_2bit 行计算
    p8 = next(float(r["ppl"]) for r in rows if r["config"] == "all_8bit")
    p2 = next(float(r["ppl"]) for r in rows if r["config"] == "all_2bit")

    allocation = {}
    for r in rows:
        if r["config"] == "all_8bit" or r["config"] == "all_2bit":
            continue
        layer_idx = r["layer_idx"]
        sens = (float(r["ppl"]) - p8) / (p2 - p8)
        dtype = "float16" if sens > args.threshold else args.quant_dtype
        allocation[str(layer_idx)] = dtype
        print(f"layer{layer_idx}: sens={sens:.3f} -> {dtype}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(allocation, indent=2))
    print(f"→ {out}")
    print("CLI: --kv-cache-dtype int4_per_token_head --kv-cache-dtype-per-layer "
          + ",".join(f"{k}:{v}" for k, v in sorted(allocation.items())))


if __name__ == "__main__":
    main()
