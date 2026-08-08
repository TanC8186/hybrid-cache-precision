"""Probe the impact of GDN SSM state dtype on vLLM cache capacity and generation.

Feasibility probe for the proposed "state-compression" direction: the vLLM fork
already exposes ``--mamba-ssm-cache-dtype`` (auto/float32/float16/bfloat16) and
allocates the GDN temporal state with the resolved dtype. This script measures
the same configuration under two settings (e.g. auto->float32 vs bfloat16) and
reports capacity tokens, max concurrency, per-layer cache tensor dtypes/bytes,
and an optional greedy generation smoke output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
DEFAULT_PER_LAYER = {
    "23": "auto",
    "3": "int4_per_token_head",
    "7": "int4_per_token_head",
    "11": "int4_per_token_head",
    "15": "int4_per_token_head",
    "19": "int4_per_token_head",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    digest = sha256_file(path)
    hash_path = path.with_suffix(path.suffix + ".sha256")
    hash_path.write_text(f"{digest}\n", encoding="ascii")
    return digest


def summarize_layers(workers: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-worker cache tensors by dtype and by layer."""
    summaries = []
    for worker in workers:
        by_dtype: dict[str, int] = {}
        layers = []
        for layer_name, layer in worker.get("bound_layers", {}).items():
            tensors = [
                {
                    "shape": info["shape"],
                    "dtype": info["dtype"],
                    "logical_nbytes": info["logical_nbytes"],
                }
                for info in layer.get("cache_tensors", [])
            ]
            if not tensors:
                continue
            for info in tensors:
                by_dtype[info["dtype"]] = by_dtype.get(info["dtype"], 0) + info["logical_nbytes"]
            layers.append(
                {
                    "layer": layer_name,
                    "type": layer["layer_type"],
                    "tensors": tensors,
                }
            )
        summaries.append(
            {
                "rank": worker.get("rank"),
                "bytes_by_dtype": by_dtype,
                "total_cache_bytes": sum(by_dtype.values()),
                "layers": layers,
            }
        )
    return {"workers": summaries}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Value passed to LLM(mamba_ssm_cache_dtype=...); 'auto' leaves vLLM default.",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv-cache-dtype", default="int4_per_token_head")
    parser.add_argument(
        "--kv-cache-dtype-per-layer",
        default=json.dumps(DEFAULT_PER_LAYER, separators=(",", ":")),
        help="JSON object string or path to a JSON file; use '{}' for uniform.",
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument(
        "--prompt",
        default="State one reason KV-cache capacity matters for LLM serving.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inspect_kv_config import (
        collect_worker_runtime,
        extract_shared_runtime,
    )

    from vllm import LLM, SamplingParams

    per_layer: dict[str, str] | None = None
    if args.kv_cache_dtype_per_layer:
        raw = args.kv_cache_dtype_per_layer
        path = Path(raw)
        text = path.read_text(encoding="utf-8") if path.is_file() else raw
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise TypeError("per-layer dtype configuration must be a JSON object")
        per_layer = {str(key): str(value) for key, value in parsed.items()}

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "enforce_eager": False,
    }
    if per_layer is not None:
        llm_kwargs["kv_cache_dtype_per_layer"] = per_layer
    if args.dtype != "auto":
        llm_kwargs["mamba_ssm_cache_dtype"] = args.dtype

    started = time.time()
    llm = LLM(**llm_kwargs)
    workers = llm.collective_rpc(collect_worker_runtime)
    shared_runtime = extract_shared_runtime(workers)
    capacity = shared_runtime["capacity"]

    generation: dict[str, Any] | None = None
    if args.generate:
        outputs = llm.generate(
            [args.prompt],
            SamplingParams(temperature=0, max_tokens=args.max_tokens),
        )
        output = outputs[0].outputs[0]
        generation = {
            "prompt": args.prompt,
            "text": output.text,
            "finish_reason": output.finish_reason,
            "output_token_count": len(output.token_ids),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "probe_ssm_state_dtype.py",
        "args": {
            "model": args.model,
            "dtype_arg": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
            "kv_cache_dtype": args.kv_cache_dtype,
            "kv_cache_dtype_per_layer": per_layer,
        },
        "resolved_mamba_ssm_cache_dtype": shared_runtime["cache_config"]["mamba_ssm_cache_dtype"],
        "resolved_mamba_cache_dtype": shared_runtime["cache_config"]["mamba_cache_dtype"],
        "capacity": capacity,
        "cache_config": shared_runtime["cache_config"],
        "kv_cache_config": shared_runtime["kv_cache_config"],
        "cache_tensor_summary": summarize_layers(workers),
        "generation": generation,
        "elapsed_seconds": round(time.time() - started, 2),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    digest = atomic_write_json(args.output, report)
    print(json.dumps(
        {
            "output": str(args.output),
            "sha256": digest,
            "dtype_arg": args.dtype,
            "resolved_mamba_ssm_cache_dtype": report["resolved_mamba_ssm_cache_dtype"],
            "capacity_tokens": capacity["tokens"],
            "max_concurrency": capacity["max_concurrency"],
            "generation_tokens": (generation or {}).get("output_token_count", 0),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
