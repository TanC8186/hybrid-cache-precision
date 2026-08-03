#!/usr/bin/env python3
"""Extract bench_lat JSON provenance from rate_*.log Namespace lines (code-review H3 fix).

E2/E3's 20 JSONs (vllm bench serve native output) do not record seed / num_warmups /
server config / commit. The full config is frozen in each rate_*.log's Namespace line.
This script extracts key fields per log and emits JSON_PROVENANCE.md.
Usage: python scripts/bench/gen_json_provenance.py   (from repo root)
"""
import glob, os, re, sys

FIELDS = {
    "request_rate": r"request_rate=([\d.]+)",
    "num_prompts": r"num_prompts=(\d+)",
    "seed": r"seed=(\d+)",
    "num_warmups": r"num_warmups=(\d+)",
    "max_concurrency": r"max_concurrency=(\d+)",
    "random_input_len": r"random_input_len=(\d+)",
    "random_output_len": r"random_output_len=(\d+)",
    "base_url": r"base_url='([^']+)'",
    "dataset_name": r"dataset_name='([^']+)'",
    "model": r"model='([^']+)'",
}
HDR = ["alloc", "file", "request_rate", "num_prompts", "seed", "num_warmups",
       "max_concurrency", "random_input_len", "random_output_len", "base_url",
       "dataset_name", "model"]


def extract_ns(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"Namespace\((.*)\)", txt, re.S)
    if not m:
        return None
    ns = m.group(1)
    out = {}
    for k, pat in FIELDS.items():
        mm = re.search(pat, ns)
        out[k] = mm.group(1) if mm else "?"
    return out


def main():
    rows = []
    for sub in ["int4", "fp16"]:
        d = os.path.join("results", "ablations", "bench_lat", sub)
        for f in sorted(glob.glob(os.path.join(d, "rate_*.log"))):
            e = extract_ns(f)
            if e:
                e["alloc"] = sub
                e["file"] = os.path.basename(f)
                rows.append(e)
    out = os.path.join("results", "ablations", "bench_lat", "JSON_PROVENANCE.md")
    with open(out, "w", encoding="utf-8") as o:
        o.write("# bench_lat JSON Provenance（逐 rate_*.log Namespace 自动提取，2026-08-03）\n\n")
        o.write("> 补 code-review H3 缺口：E2/E3 的 20 个 JSON 自身不含 seed/warmup/server 配置；"
                "配置固化于各 rate_*.log 的 Namespace。本表由 `scripts/bench/gen_json_provenance.py` 生成。\n\n")
        o.write("| " + " | ".join(HDR) + " |\n")
        o.write("|" + "|".join(["---"] * len(HDR)) + "|\n")
        for r in rows:
            o.write("| " + " | ".join(str(r.get(k, "?")) for k in HDR) + " |\n")
        o.write("\n注释：`seed=0`, `num_warmups=0` 为 vllm bench serve 默认（本矩阵未显式传 --seed/--warmup-n）。"
                "server 层配置（kv_cache_dtype_per_layer 等）见 logs/PROVENANCE.md。\n")
    print(f"wrote {out}: {len(rows)} rows")


if __name__ == "__main__":
    sys.exit(main())
