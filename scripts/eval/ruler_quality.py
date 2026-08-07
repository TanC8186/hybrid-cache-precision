"""RULER-subset quality eval via vLLM offline greedy generation.

Cell = (task, length, allocation, seed). Loads the fixed dataset
(data/ruler/{task}_L{length}/validation.jsonl), appends the official
answer_prefix, generates with the same vLLM engine kwargs as the NIAH
quality matrix, and scores with the official RULER `string_match_all`
metric. Output is one atomic JSON + .sha256 per cell (resumable).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path


RULER_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "ruler"
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
MODEL_DEFAULT = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
ALLOCATIONS = ["fp16", "uniform_int4", "packed_per_layer", "turboquant_k8v4", "turboquant_4bit_nc"]
TASK_TYPE = {
    "ruler_niah_single": "niah",
    "ruler_niah_multikey": "niah",
    "ruler_niah_multivalue": "niah",
    "ruler_niah_multiquery": "niah",
    "ruler_vt": "variable_tracking",
    "ruler_cwe": "common_words_extraction",
    "ruler_fwe": "freq_words_extraction",
}


def load_tasks_base() -> dict:
    spec = importlib.util.spec_from_file_location(
        "ruler_tasks_base", RULER_ROOT / "scripts" / "data" / "synthetic" / "constants.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASKS


def load_metrics() -> dict:
    spec = importlib.util.spec_from_file_location(
        "ruler_metrics", RULER_ROOT / "scripts" / "eval" / "synthetic" / "constants.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASKS


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict) -> str:
    import os
    import uuid

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


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    score = sum(
        sum(1.0 if reference.lower() in pred.lower() else 0.0 for reference in reference_list)
        / len(reference_list)
        for pred, reference_list in zip(preds, refs)
    ) / len(preds) * 100
    return round(score, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--length", type=int, required=True, choices=[4096, 8192])
    ap.add_argument(
        "--allocation",
        required=True,
        choices=ALLOCATIONS,
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--data-root", default="data/ruler")
    ap.add_argument("--out-dir", default="results/quality/ruler-subset")
    ap.add_argument("--attempt-id", default="ruler-subset-20260807")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="generation budget override; 0 = official tokens_to_generate",
    )
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="wrap prompts with the model chat template using enable_thinking=False",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data_file = Path(args.data_root) / f"{args.task}_L{args.length}" / "validation.jsonl"
    if not data_file.exists():
        raise SystemExit(f"dataset missing: {data_file}")
    data_sha = sha256_file(data_file)
    tasks_base = load_tasks_base()
    task_type = TASK_TYPE.get(args.task)
    if task_type is None:
        raise SystemExit(f"unknown ruler task: {args.task}")
    config = tasks_base[task_type]
    effective_max_tokens = args.max_tokens or int(config["tokens_to_generate"])
    thinking_mode = "disabled" if args.disable_thinking else "default"

    out_path = (
        Path(args.out_dir)
        / args.attempt_id
        / f"{args.task}__L{args.length}__{args.allocation}__s{args.seed}.json"
    )
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed_validated"
            and existing.get("data_sha256") == data_sha
            and existing.get("ruler_commit") == RULER_COMMIT
            and existing.get("max_tokens") == effective_max_tokens
            and existing.get("thinking") == thinking_mode
        ):
            print(f"resume: skip {out_path}")
            return 0

    rows = [
        json.loads(line)
        for line in data_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing_prefix = [row.get("index") for row in rows if "answer_prefix" not in row]
    if missing_prefix:
        raise SystemExit(f"fail-closed: {len(missing_prefix)} rows lack answer_prefix")

    metrics = load_metrics()
    metric_fn = metrics[task_type]["metric_fn"]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kv_quality_retrieval import engine_kwargs, verify_config_effect  # noqa: PLC0415

    kwargs = engine_kwargs(args.allocation, argparse.Namespace(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    ))
    from vllm import LLM, SamplingParams, __version__ as vllm_version  # noqa: PLC0415

    t0 = time.time()
    llm = LLM(**kwargs)
    effect = verify_config_effect(llm, args.allocation)
    if not effect.get("ok"):
        print(f"config effect FAILED: {json.dumps(effect, ensure_ascii=False)}", file=sys.stderr)
        return 3

    raw_prompts = [row["input"] + row.get("answer_prefix", "") for row in rows]
    if args.disable_thinking:
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in raw_prompts
        ]
    else:
        prompts = raw_prompts
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=effective_max_tokens, temperature=0.0),
        use_tqdm=True,
    )
    preds = [output.outputs[0].text for output in outputs]
    refs = [list(row["outputs"]) for row in rows]
    score = metric_fn(preds, refs)
    cases = []
    for row, pred, output in zip(rows, preds, outputs):
        hits = [reference.lower() in pred.lower() for reference in row["outputs"]]
        cases.append(
            {
                "index": row["index"],
                "references": row["outputs"],
                "prediction": pred,
                "hits": hits,
                "prompt_tokens": len(output.prompt_token_ids),
                "output_tokens": len(output.outputs[0].token_ids),
            }
        )

    record = {
        "schema_version": 1,
        "attempt_id": args.attempt_id,
        "status": "completed_validated",
        "task": args.task,
        "task_type": task_type,
        "length": args.length,
        "allocation": args.allocation,
        "seed": args.seed,
        "num_samples": len(rows),
        "tokens_to_generate": config["tokens_to_generate"],
        "max_tokens": effective_max_tokens,
        "thinking": thinking_mode,
        "accuracy": score,
        "metric": "string_match_all",
        "ruler_commit": RULER_COMMIT,
        "data_file": str(data_file),
        "data_sha256": sha256_file(data_file),
        "cases": cases,
        "engine": {
            "model": args.model,
            "kwargs": {k: v for k, v in kwargs.items() if k != "model"},
            "vllm_version": vllm_version,
        },
        "config_effect": effect,
        "sampling_params": {
            "max_tokens": effective_max_tokens,
            "temperature": 0.0,
            "thinking": thinking_mode,
        },
        "elapsed_s": round(time.time() - t0, 1),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
    }
    digest = atomic_write_json(out_path, record)
    print(f"RULER({args.task}, L{args.length}, {args.allocation}, seed={args.seed}) acc={score}")
    print(f"→ {out_path} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
