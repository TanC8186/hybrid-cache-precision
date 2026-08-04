"""Dump effective vLLM KV-cache configuration and runtime tensor views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from enum import Enum
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


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return str(value)


def tensor_info(tensor: Any) -> dict[str, Any]:
    storage = tensor.untyped_storage()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "logical_nbytes": tensor.numel() * tensor.element_size(),
        "storage_offset": tensor.storage_offset(),
        "data_ptr": tensor.data_ptr(),
        "storage_data_ptr": storage.data_ptr(),
        "storage_nbytes": storage.nbytes(),
        "is_contiguous": tensor.is_contiguous(),
    }


def collect_worker_runtime(worker: Any) -> dict[str, Any]:
    """Executed through vLLM collective_rpc on each worker."""
    import torch

    model_runner = worker.model_runner
    forward_context = model_runner.compilation_config.static_forward_context
    layers: dict[str, Any] = {}
    storage_ptrs: set[int] = set()
    for layer_name, layer in forward_context.items():
        cache = getattr(layer, "kv_cache", None)
        if cache is None:
            continue
        cache_tensors = list(cache) if isinstance(cache, tuple) else [cache]
        infos = [tensor_info(tensor) for tensor in cache_tensors]
        storage_ptrs.update(info["storage_data_ptr"] for info in infos)
        layers[layer_name] = {
            "layer_type": type(layer).__name__,
            "cache_tensors": infos,
        }

    raw_caches = [tensor_info(tensor) for tensor in model_runner.kv_caches]
    return {
        "rank": getattr(worker, "rank", None),
        "local_rank": getattr(worker, "local_rank", None),
        "device": str(model_runner.device),
        "torch_cuda_device": torch.cuda.get_device_name(model_runner.device),
        "kernel_block_sizes": list(model_runner._kernel_block_sizes),
        "raw_runner_cache_count": len(raw_caches),
        "raw_runner_caches": raw_caches,
        "bound_layers": layers,
        "unique_backing_storage_count": len(storage_ptrs),
        "unique_backing_storage_ptrs": sorted(storage_ptrs),
    }


def spec_to_dict(spec: Any) -> dict[str, Any]:
    output: dict[str, Any] = {
        "type": type(spec).__name__,
        "block_size": getattr(spec, "block_size", None),
        "page_size_bytes": getattr(spec, "page_size_bytes", None),
    }
    for name in (
        "dtype",
        "kv_quant_mode",
        "num_kv_heads",
        "head_size",
        "head_size_v",
        "use_mla",
        "sliding_window",
        "attention_chunk_size",
        "page_size_padded",
        "mamba_type",
        "mamba_cache_mode",
        "shapes",
        "dtypes",
    ):
        if hasattr(spec, name):
            output[name] = json_value(getattr(spec, name))
    inner = getattr(spec, "kv_cache_specs", None)
    if inner is not None:
        output["per_layer_specs"] = {layer_name: spec_to_dict(layer_spec) for layer_name, layer_spec in inner.items()}
    return output


def parse_per_layer(value: str | None) -> dict[str, str] | None:
    if value is None:
        return None
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.is_file() else value
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise TypeError("per-layer dtype configuration must be a JSON object")
    return {str(key): str(item) for key, item in parsed.items()}


def verify_packed_per_layer(report: Mapping[str, Any]) -> dict[str, Any]:
    groups = report["kv_cache_config"]["groups"]
    tensors = report["kv_cache_config"]["tensors"]
    attention_groups = [group for group in groups if group["spec"]["type"] == "UniformTypeKVCacheSpecs"]
    inner_modes: set[str] = set()
    if len(attention_groups) == 1:
        inner = attention_groups[0]["spec"].get("per_layer_specs", {})
        inner_modes = {str(spec.get("kv_quant_mode")) for spec in inner.values()}
    worker_storage_counts = [worker["unique_backing_storage_count"] for worker in report["workers"]]
    checks = {
        "flag_enabled": bool(report["cache_config"].get("enable_per_layer_page_groups")),
        "mamba_ssm_cache_dtype_is_float32": (str(report["cache_config"]["mamba_ssm_cache_dtype"]) == "float32"),
        "one_packed_attention_group": len(attention_groups) == 1,
        "attention_group_is_mixed_precision": (
            any("INT4" in mode.upper() for mode in inner_modes)
            and any(token in mode.upper() for mode in inner_modes for token in ("NONE", "AUTO"))
        ),
        "all_layout_entries_are_packed": bool(tensors) and all(int(tensor["block_stride"]) > 0 for tensor in tensors),
        "one_backing_storage_per_worker": bool(worker_storage_counts)
        and all(count == 1 for count in worker_storage_counts),
        "positive_capacity": report["capacity"]["tokens"] > 0,
        "generation_completed": (report["generation"] is None or report["generation"].get("output_token_count", 0) > 0),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "attention_kv_quant_modes": sorted(inner_modes),
        "worker_backing_storage_counts": worker_storage_counts,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv-cache-dtype", default="int4_per_token_head")
    parser.add_argument(
        "--kv-cache-dtype-per-layer",
        default=json.dumps(DEFAULT_PER_LAYER, separators=(",", ":")),
        help="JSON object or path to a JSON file",
    )
    parser.add_argument("--enable-per-layer-page-groups", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument(
        "--prompt",
        default="State one reason KV-cache capacity matters for LLM serving.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--expect-packed-per-layer", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    per_layer = parse_per_layer(args.kv_cache_dtype_per_layer)

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "disable_log_stats": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
    }
    if per_layer is not None:
        llm_kwargs["kv_cache_dtype_per_layer"] = per_layer
    if args.enable_per_layer_page_groups:
        llm_kwargs["enable_per_layer_page_groups"] = True

    llm = LLM(**llm_kwargs)
    engine = llm.llm_engine
    vllm_config = engine.vllm_config
    cache_config = vllm_config.cache_config
    kv_cache_config = engine.model_executor.kv_cache_config
    capacity_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)

    groups = [
        {
            "group_id": group_id,
            "layer_names": list(group.layer_names),
            "spec": spec_to_dict(group.kv_cache_spec),
        }
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
    ]
    tensors = [
        {
            "tensor_id": index,
            "size": tensor.size,
            "shared_by": list(tensor.shared_by),
            "offset": tensor.offset,
            "block_stride": tensor.block_stride,
        }
        for index, tensor in enumerate(kv_cache_config.kv_cache_tensors)
    ]
    workers = engine.model_executor.collective_rpc(collect_worker_runtime)

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
            "token_ids": list(output.token_ids),
            "output_token_count": len(output.token_ids),
            "finish_reason": output.finish_reason,
        }

    model_config_path = Path(args.model) / "config.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": utc_timestamp(),
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "vllm_file": vllm.__file__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "model_path": str(Path(args.model).resolve()),
            "model_config_sha256": (sha256_file(model_config_path) if model_config_path.exists() else None),
        },
        "requested": {
            "kv_cache_dtype": args.kv_cache_dtype,
            "kv_cache_dtype_per_layer": per_layer,
            "enable_per_layer_page_groups": args.enable_per_layer_page_groups,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
        },
        "cache_config": {
            "cache_dtype": json_value(cache_config.cache_dtype),
            "kv_cache_dtype_per_layer": json_value(cache_config.kv_cache_dtype_per_layer),
            "enable_per_layer_page_groups": getattr(cache_config, "enable_per_layer_page_groups", None),
            "num_gpu_blocks": cache_config.num_gpu_blocks,
            "block_size": cache_config.block_size,
            "mamba_block_size": cache_config.mamba_block_size,
            "mamba_page_size_padded": cache_config.mamba_page_size_padded,
            "mamba_cache_dtype": json_value(cache_config.mamba_cache_dtype),
            "mamba_ssm_cache_dtype": json_value(cache_config.mamba_ssm_cache_dtype),
        },
        "kv_cache_config": {
            "num_blocks": kv_cache_config.num_blocks,
            "groups": groups,
            "tensors": tensors,
        },
        "capacity": {
            "tokens": capacity_tokens,
            "max_concurrency": max_concurrency,
            "max_model_len": vllm_config.model_config.max_model_len,
        },
        "workers": workers,
        "generation": generation,
    }
    report["verification"] = (
        verify_packed_per_layer(report) if args.expect_packed_per_layer else {"passed": None, "checks": {}}
    )
    digest = atomic_write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                "capacity_tokens": capacity_tokens,
                "max_concurrency": max_concurrency,
                "verification": report["verification"],
            },
            indent=2,
        )
    )
    if args.expect_packed_per_layer and not report["verification"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
