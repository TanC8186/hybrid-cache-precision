"""vLLM-offline Wikitext-2 PPL for Qwen3.5 hybrid allocations (R4 quality closure).

Allocations:
  fp16            -> kv_cache_dtype=auto
  uniform_int4    -> kv_cache_dtype=int4_per_token_head
  packed_per_layer-> kv_cache_dtype=int4_per_token_head + per-layer map (L23 auto)
                     + enable_per_layer_page_groups=True

Protocol (matches canonical PPL file byte_budget_3seed.csv):
  Wikitext-2 test, 5 sequences x 2048 tokens, chunked prefill via vLLM,
  seed-controlled start positions, greedy prompt logprobs.

Each sample writes one atomic JSON (+ .sha256) and is independently resumable:
  --resume skips samples whose JSON exists and passes self-validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

MODEL_DEFAULT = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
PER_LAYER = {
    "23": "auto",
    "3": "int4_per_token_head",
    "7": "int4_per_token_head",
    "11": "int4_per_token_head",
    "15": "int4_per_token_head",
    "19": "int4_per_token_head",
}


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def sample_sequences(tokenizer, corpus_path: Path, max_len: int, num_seqs: int, seed: int) -> list[list[int]]:
    text = corpus_path.read_text(encoding="utf-8")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    total = len(ids)
    import torch

    if seed is not None:
        rng = torch.Generator().manual_seed(seed)
        hi = max(1, total - max_len)
        starts = torch.randint(0, hi, (num_seqs,), generator=rng).tolist()
    else:
        starts = list(range(0, max(1, total - 1), max_len))[:num_seqs]
    seqs = []
    for s in starts:
        chunk = ids[s : s + max_len]
        if chunk.shape[0] >= 2:
            seqs.append(chunk.tolist())
    if not seqs:
        raise ValueError("corpus too short")
    return seqs


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
    """Best-effort verification that the requested allocation actually took effect."""
    checks: dict[str, Any] = {"ok": False, "allocation": allocation, "detail": {}}
    eng = getattr(llm, "llm_engine", None)
    if eng is None:
        checks["reason"] = "no llm_engine"
        return checks
    cc = getattr(eng, "cache_config", None)
    if cc is not None:
        checks["detail"]["cache_dtype"] = str(getattr(cc, "cache_dtype", None))
        checks["detail"]["per_layer"] = getattr(cc, "kv_cache_dtype_per_layer", None)
        checks["detail"]["a2_flag"] = bool(getattr(cc, "enable_per_layer_page_groups", False))
    # Deeper: kv cache groups from the worker model runner (if reachable).
    kv = None
    try:
        runner = eng.model_executor.driver_worker.model_runner
        kv = runner.kv_cache_config
    except Exception:
        kv = None
    if kv is not None:
        checks["detail"]["num_blocks"] = getattr(kv, "num_blocks", None)
        groups = []
        for group in getattr(kv, "kv_cache_groups", []):
            groups.append(
                {
                    "type": type(group.kv_cache_spec).__name__,
                    "layers": list(group.layer_names),
                }
            )
        checks["detail"]["groups"] = groups
    dtype = checks["detail"].get("cache_dtype")
    per_layer = checks["detail"].get("per_layer")
    a2 = checks["detail"].get("a2_flag")
    if allocation == "fp16":
        checks["ok"] = dtype in ("auto", "fp16", "bf16") and not per_layer and not a2
    elif allocation == "uniform_int4":
        checks["ok"] = dtype == "int4_per_token_head" and not per_layer and not a2
    elif allocation == "packed_per_layer":
        checks["ok"] = (
            dtype == "int4_per_token_head"
            and per_layer == PER_LAYER
            and a2 is True
        )
        if "groups" in checks["detail"]:
            types = [g["type"] for g in checks["detail"]["groups"]]
            checks["ok"] = checks["ok"] and any(t == "UniformTypeKVCacheSpecs" for t in types)
    if not checks["ok"]:
        checks["reason"] = "allocation did not take effect (config mismatch)"
    return checks


def compute_ppl(llm, tokenizer, seqs: list[list[int]]) -> tuple[float, list[float]]:
    """Greedy teacher-forced PPL from vLLM prompt logprobs."""
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=1, prompt_logprobs=1)
    requests = [{"prompt_token_ids": seq} for seq in seqs]
    outputs = llm.generate(requests, sp, use_tqdm=False)
    per_seq: list[float] = []
    for out, seq in zip(outputs, seqs):
        logprobs = getattr(out, "prompt_logprobs", None)
        if not logprobs:
            raise RuntimeError("prompt_logprobs missing in output")
        lp_sum = 0.0
        n = 0
        for i, lp in enumerate(logprobs):
            if i == 0 or lp is None:
                continue
            tok = seq[i]
            if tok in lp:
                lp_sum += lp[tok].logprob
                n += 1
        if n == 0:
            raise RuntimeError("no measurable prompt logprobs")
        per_seq.append(math.exp(-lp_sum / n))
    mean = math.exp(sum(math.log(p) for p in per_seq) / len(per_seq))
    return mean, per_seq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allocation", required=True, choices=["fp16", "uniform_int4", "packed_per_layer"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--corpus", default="data/wikitext2_test.txt")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--num-seqs", type=int, default=5)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out-dir", default="results/quality/r4-ppl")
    ap.add_argument("--attempt-id", default=f"r4-ppl-{time.strftime('%Y%m%d')}")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.attempt_id
    out_path = out_dir / f"{args.allocation}__seed{args.seed}.json"
    if args.resume and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed_validated":
                print(f"resume: skip {out_path}")
                return 0
        except Exception:
            pass

    from transformers import AutoTokenizer

    from vllm import LLM

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"corpus missing: {corpus_path}", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    seqs = sample_sequences(tokenizer, corpus_path, args.max_len, args.num_seqs, args.seed)

    kwargs = engine_kwargs(args.allocation, args)
    t0 = time.time()
    llm = LLM(**kwargs)
    effect = verify_config_effect(llm, args.allocation)
    if not effect.get("ok"):
        print(f"config effect FAILED: {json.dumps(effect, ensure_ascii=False)}", file=sys.stderr)
        return 3

    ppl, per_seq = compute_ppl(llm, tokenizer, seqs)
    elapsed = time.time() - t0

    from vllm import __version__ as vllm_version

    record = {
        "schema_version": 1,
        "attempt_id": args.attempt_id,
        "status": "completed_validated",
        "allocation": args.allocation,
        "seed": args.seed,
        "ppl": round(ppl, 6),
        "per_seq_ppl": [round(p, 6) for p in per_seq],
        "num_seqs": len(seqs),
        "max_len": args.max_len,
        "corpus": {
            "path": str(corpus_path),
            "sha256": sha256_text(corpus_path.read_text(encoding="utf-8")),
        },
        "engine": {
            "model": args.model,
            "kwargs": {k: v for k, v in kwargs.items() if k != "model"},
            "vllm_version": vllm_version,
        },
        "config_effect": effect,
        "elapsed_s": round(elapsed, 1),
        "created_at_utc": utc_timestamp(),
        "host": platform.node(),
    }
    digest = atomic_write_json(out_path, record)
    print(f"PPL({args.allocation}, seed={args.seed}) = {ppl:.6f}")
    print(f"→ {out_path} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
