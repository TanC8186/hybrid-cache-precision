"""NIAH (needle-in-a-haystack) retrieval quality via vLLM offline greedy generation (R4).

Allocations: fp16 / uniform_int4 / packed_per_layer (same engine args as serving).
Each sample: one (allocation, seed, depth_pct, max_len) cell with num_needles needles;
writes one atomic JSON + .sha256 and is independently resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
import uuid
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL_DEFAULT = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
PER_LAYER = {
    "23": "auto",
    "3": "int4_per_token_head",
    "7": "int4_per_token_head",
    "11": "int4_per_token_head",
    "15": "int4_per_token_head",
    "19": "int4_per_token_head",
}

WORDS = (
    "alpine basket candle drizzle ember falcon garden harbor island jungle kettle lantern meadow "
    "nebula orchard pebble quartz river silver thunder umbrella valley willow yellow zebra "
    "anchor beacon compass dolphin equator glacier horizon ivory journey kernel lagoon marble "
    "needle oxygen prairie rocket signal tracer underwood vapor whisper xylem".split()
)


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
    Path(str(path) + ".sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def make_code(rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(alphabet) for _ in range(8))


def build_prompt(rng: random.Random, length: int, depth_pct: int, code: str) -> str:
    pre_words = max(1, int(length * depth_pct / 100))
    post_words = max(1, length - pre_words)
    pre = " ".join(rng.choice(WORDS) for _ in range(pre_words))
    post = " ".join(rng.choice(WORDS) for _ in range(post_words))
    needle = f"The secret code is {code}."
    return f"{pre} {needle} {post}\n\nQuestion: What is the secret code mentioned in the text? Answer with the exact code.\nAnswer:"


def engine_kwargs(allocation: str, args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "seed": args.seed,
        "disable_log_stats": True,
    }
    if allocation == "uniform_int4":
        kwargs["kv_cache_dtype"] = "int4_per_token_head"
    elif allocation == "packed_per_layer":
        kwargs["kv_cache_dtype"] = "int4_per_token_head"
        kwargs["kv_cache_dtype_per_layer"] = dict(PER_LAYER)
        kwargs["enable_per_layer_page_groups"] = True
    return kwargs


def verify_config_effect(llm, allocation: str) -> dict[str, Any]:
    checks: dict[str, Any] = {"ok": False, "allocation": allocation, "detail": {}}
    eng = getattr(llm, "llm_engine", None)
    if eng is None:
        checks["reason"] = "no llm_engine"
        return checks
    cc = None
    try:
        cc = eng.vllm_config.cache_config
    except Exception:
        cc = getattr(eng, "cache_config", None)
    if cc is not None:
        checks["detail"]["cache_dtype"] = str(getattr(cc, "cache_dtype", None))
        checks["detail"]["per_layer"] = getattr(cc, "kv_cache_dtype_per_layer", None)
        checks["detail"]["a2_flag"] = bool(getattr(cc, "enable_per_layer_page_groups", False))
    kv = None
    try:
        core = getattr(getattr(eng, "engine_core", None), "core_engine", None)
        kv = core.model_executor.driver_worker.model_runner.kv_cache_config
    except Exception:
        kv = None
    if kv is not None:
        checks["detail"]["num_blocks"] = getattr(kv, "num_blocks", None)
        checks["detail"]["groups"] = [
            {
                "type": type(g.kv_cache_spec).__name__,
                "layers": list(g.layer_names),
            }
            for g in getattr(kv, "kv_cache_groups", [])
        ]
    dtype = checks["detail"].get("cache_dtype")
    per_layer = checks["detail"].get("per_layer")
    a2 = checks["detail"].get("a2_flag")
    if allocation == "fp16":
        checks["ok"] = dtype in ("auto", "fp16", "bf16") and not per_layer and not a2
    elif allocation == "uniform_int4":
        checks["ok"] = dtype == "int4_per_token_head" and not per_layer and not a2
    elif allocation == "packed_per_layer":
        checks["ok"] = dtype == "int4_per_token_head" and per_layer == PER_LAYER and a2 is True
        if "groups" in checks["detail"]:
            checks["ok"] = checks["ok"] and any(
                g["type"] == "UniformTypeKVCacheSpecs" for g in checks["detail"]["groups"]
            )
    if not checks["ok"]:
        checks["reason"] = "allocation did not take effect (config mismatch)"
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allocation", required=True, choices=["fp16", "uniform_int4", "packed_per_layer"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--depth-pct", type=int, required=True, choices=[25, 50, 75])
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--num-needles", type=int, default=3)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out-dir", default="results/quality/r4-niah")
    ap.add_argument("--attempt-id", default=f"r4-niah-{time.strftime('%Y%m%d')}")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.attempt_id
    out_path = out_dir / f"{args.allocation}__seed{args.seed}__d{args.depth_pct}__l{args.max_len}.json"
    if args.resume and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed_validated":
                print(f"resume: skip {out_path}")
                return 0
        except Exception:
            pass

    from vllm import LLM, SamplingParams

    kwargs = engine_kwargs(args.allocation, args)
    t0 = time.time()
    llm = LLM(**kwargs)
    effect = verify_config_effect(llm, args.allocation)
    if not effect.get("ok"):
        print(f"config effect FAILED: {json.dumps(effect, ensure_ascii=False)}", file=__import__("sys").stderr)
        return 3

    rng = random.Random((args.seed * 1000003 + args.depth_pct * 31 + args.max_len * 17) & 0xFFFFFFFF)
    cases = []
    for _ in range(args.num_needles):
        code = make_code(rng)
        prompt = build_prompt(rng, args.max_len, args.depth_pct, code)
        outputs = llm.generate([prompt], SamplingParams(max_tokens=32, temperature=0.0), use_tqdm=False)
        answer = outputs[0].outputs[0].text
        hit = code in answer.upper().replace(" ", "")
        cases.append({"code": code, "answer": answer, "hit": hit})

    from vllm import __version__ as vllm_version

    record = {
        "schema_version": 1,
        "attempt_id": args.attempt_id,
        "status": "completed_validated",
        "allocation": args.allocation,
        "seed": args.seed,
        "depth_pct": args.depth_pct,
        "max_len": args.max_len,
        "num_needles": args.num_needles,
        "accuracy": sum(1 for c in cases if c["hit"]) / len(cases),
        "cases": cases,
        "engine": {"model": args.model, "kwargs": {k: v for k, v in kwargs.items() if k != "model"}, "vllm_version": vllm_version},
        "config_effect": effect,
        "elapsed_s": round(time.time() - t0, 1),
        "created_at_utc": utc_timestamp(),
        "host": platform.node(),
    }
    digest = atomic_write_json(out_path, record)
    print(f"NIAH({args.allocation}, seed={args.seed}, d={args.depth_pct}, L={args.max_len}) acc={record['accuracy']:.2f}")
    print(f"→ {out_path} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
