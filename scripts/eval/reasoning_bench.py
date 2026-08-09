"""Reasoning/downstream benchmark eval via vLLM offline greedy generation.

Benches (disclosed subsets):
- gsm8k: openai/gsm8k main/test, first 200 rows (deterministic head).
- mmlu: cais/mmlu all/test, first 500 rows (deterministic head).
- aime25: opencompass/AIME2025, all 30 problems (I + II).

Cell = (bench, allocation, seed). Atomic JSON + sha256 per cell, resumable.
Scoring is deterministic extraction (documented per bench). Main protocol:
--disable-thinking (chat template, enable_thinking=False) and generous budgets
(gsm8k 1024 / mmlu 512 / aime25 4096) so the model can state a final answer.
Extraction prefers the last "answer"/"result" marker (strict final answer);
if no marker exists it falls back to the last candidate token and records the
case as fallback so artifact cases remain auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import uuid
from pathlib import Path


MODEL_DEFAULT = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
DATA_ROOT = Path("/root/autodl-tmp/caches/datasets")
ALLOCATIONS = [
    "fp16",
    "fp16_statebf16",
    "uniform_int4",
    "uniform_int4_statebf16",
    "packed_per_layer",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
]

BENCH_CONFIG = {
    "gsm8k": {"max_tokens": 1024, "max_samples": 200},
    "mmlu": {"max_tokens": 512, "max_samples": 500},
    "aime25": {"max_tokens": 4096, "max_samples": 30},
}

ANSWER_MARKER_RE = re.compile(r"(?i)\b(?:answer|result)(?:\s*:\s*|\s+|$)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict) -> str:
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


def load_rows(bench: str, max_samples: int, seed: int) -> tuple[list[dict], list[str], list[int]]:
    """Load a deterministic per-seed sample of rows.

    For gsm8k the rows are sampled WITHOUT replacement from the full test set
    using ``random_state=seed``, so different seeds produce different question
    subsets while the same seed produces the identical subset across
    allocations (this is what makes the paired CI meaningful). Decoding stays
    greedy (temperature=0.0), so the engine ``seed`` does not add randomness.
    """
    if bench == "gsm8k":
        import pandas as pd

        df = pd.read_parquet(DATA_ROOT / "gsm8k" / "main" / "test-00000-of-00001.parquet")
        sampled = df.sample(n=max_samples, random_state=seed)
        sampled_indices = [int(i) for i in sampled.index]
        rows = []
        for _, row in sampled.iterrows():
            expected = str(row["answer"]).split("####")[-1].strip()
            rows.append({"question": str(row["question"]), "expected": expected})
        prompts = [f"Question: {r['question']}\nAnswer:" for r in rows]
        return rows, prompts, sampled_indices

    if bench == "mmlu":
        import pandas as pd

        df = pd.read_parquet(DATA_ROOT / "mmlu" / "all" / "test-00000-of-00001.parquet")
        sampled = df.head(max_samples)
        sampled_indices = list(range(max_samples))
        rows = []
        for _, row in sampled.iterrows():
            choices = list(row["choices"])
            rows.append(
                {
                    "question": str(row["question"]),
                    "subject": str(row["subject"]),
                    "choices": [str(c) for c in choices],
                    "expected": ["A", "B", "C", "D"][int(row["answer"])],
                }
            )
        prompts = []
        for r in rows:
            letters = ["A", "B", "C", "D"]
            body = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, r["choices"]))
            prompts.append(f"Question: {r['question']}\n{body}\nAnswer:")
        return rows, prompts, sampled_indices

    if bench == "aime25":
        rows = []
        for split in ("aime2025-I", "aime2025-II"):
            with (DATA_ROOT / "aime2025" / f"{split}.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    rows.append({"question": item["question"], "expected": str(item["answer"]), "split": split})
        rows = rows[:max_samples]
        prompts = [f"Problem: {r['question']}\nAnswer:" for r in rows]
        return rows, prompts, list(range(len(rows)))

    raise SystemExit(f"unknown bench: {bench}")


def extract_answer(bench: str, prediction: str) -> tuple[str | None, str]:
    markers = list(ANSWER_MARKER_RE.finditer(prediction))
    if markers:
        segment = prediction[markers[-1].end():]
        source = "final_marker"
    else:
        segment = prediction
        source = "last_token_fallback"
    if bench == "gsm8k":
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", segment)
        if not numbers:
            return None, source
        return numbers[-1].replace(",", ""), source
    if bench == "mmlu":
        letters = re.findall(r"\b([A-D])\b", segment.upper())
        return (letters[-1] if letters else None), source
    if bench == "aime25":
        numbers = re.findall(r"\d+", segment)
        return (numbers[-1] if numbers else None), source
    raise SystemExit(f"unknown bench: {bench}")


def normalize_expected(bench: str, expected: str) -> str:
    if bench == "gsm8k":
        return expected.replace(",", "")
    if bench == "aime25":
        # AIME2025 stores some answers as LaTeX/unit strings (e.g. "336^\\circ").
        # AIME answers are integers, so canonicalize to the integer.
        text = expected.strip().replace("$", "").replace(",", "")
        frac = re.fullmatch(r"\\frac\{([^}]+)\}\{([^}]+)\}", text)
        if frac:
            from fractions import Fraction

            value = Fraction(frac.group(1).strip(), frac.group(2).strip())
            return str(value) if value.denominator == 1 else str(value)
        text = re.sub(r"\\circ|\\degree|\\text\{[^}]*\}", "", text).strip()
        numbers = re.findall(r"\d+", text)
        return numbers[-1] if numbers else text
    return expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["gsm8k", "mmlu", "aime25"])
    ap.add_argument("--allocation", required=True, choices=ALLOCATIONS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out-dir", default="results/quality/reasoning")
    ap.add_argument("--attempt-id", default="reasoning-20260807")
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="wrap prompts with the model chat template using enable_thinking=False",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    max_samples = args.max_samples or BENCH_CONFIG[args.bench]["max_samples"]
    max_tokens = BENCH_CONFIG[args.bench]["max_tokens"]
    thinking = "disabled" if args.disable_thinking else "default"
    out_path = (
        Path(args.out_dir)
        / args.attempt_id
        / f"{args.bench}__{args.allocation}__s{args.seed}.json"
    )
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed_validated"
            and existing.get("thinking") == thinking
        ):
            print(f"resume: skip {out_path}")
            return 0

    rows, prompts, sampled_indices = load_rows(args.bench, max_samples, args.seed)
    if len(rows) != max_samples:
        raise SystemExit(f"rows mismatch: {len(rows)} != {max_samples}")
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
            for prompt in prompts
        ]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kv_quality_retrieval import engine_kwargs, verify_config_effect  # noqa: PLC0415

    kwargs = engine_kwargs(
        args.allocation,
        argparse.Namespace(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
        ),
    )
    from vllm import LLM, SamplingParams, __version__ as vllm_version  # noqa: PLC0415

    t0 = time.time()
    llm = LLM(**kwargs)
    effect = verify_config_effect(llm, args.allocation)
    if not effect.get("ok"):
        print(f"config effect FAILED: {json.dumps(effect, ensure_ascii=False)}", file=sys.stderr)
        return 3

    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=max_tokens, temperature=0.0),
        use_tqdm=True,
    )
    cases = []
    hits = 0
    strict_hits = 0
    for row, prompt, output in zip(rows, prompts, outputs):
        prediction = output.outputs[0].text
        predicted, extraction_source = extract_answer(args.bench, prediction)
        expected_norm = normalize_expected(args.bench, row["expected"])
        hit = predicted is not None and predicted == expected_norm
        strict_hit = hit and extraction_source == "final_marker"
        hits += int(hit)
        strict_hits += int(strict_hit)
        cases.append(
            {
                "question": row["question"],
                "expected": row["expected"],
                "prediction": prediction,
                "predicted": predicted,
                "hit": hit,
                "hit_strict_final": strict_hit,
                "extraction_source": extraction_source,
                "prompt_tokens": len(output.prompt_token_ids),
                "output_tokens": len(output.outputs[0].token_ids),
            }
        )

    record = {
        "schema_version": 1,
        "attempt_id": args.attempt_id,
        "status": "completed_validated",
        "bench": args.bench,
        "allocation": args.allocation,
        "seed": args.seed,
        "num_samples": len(cases),
        "accuracy": round(hits / len(cases), 4),
        "accuracy_strict_final": round(strict_hits / len(cases), 4),
        "thinking": thinking,
        "extraction": {
            "gsm8k": "last numeric token after final answer marker (commas stripped)",
            "mmlu": "last A-D letter token after final answer marker",
            "aime25": "last integer token after final answer marker",
        }[args.bench],
        "seed_semantics": (
            "gsm8k rows sampled without replacement using random_state=seed; "
            "mmlu/aime25 fixed deterministic head; decode greedy temperature=0.0"
        ),
        "sampled_indices": sampled_indices if args.bench == "gsm8k" else None,
        "cases": cases,
        "engine": {
            "model": args.model,
            "kwargs": {k: v for k, v in kwargs.items() if k != "model"},
            "vllm_version": vllm_version,
        },
        "config_effect": effect,
        "sampling_params": {"max_tokens": max_tokens, "temperature": 0.0},
        "elapsed_s": round(time.time() - t0, 1),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
    }
    digest = atomic_write_json(out_path, record)
    print(f"REASONING({args.bench}, {args.allocation}, seed={args.seed}) acc={record['accuracy']:.4f}")
    print(f"→ {out_path} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
