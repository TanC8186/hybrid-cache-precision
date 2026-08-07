"""LongBench v1 subset eval via vLLM offline greedy generation.

Tasks (English, config-faithful "sampling + summarization" categories):
  TREC, TriviaQA, SAMSum, LCC, RepoBench-P   (few-shot / code completion)
  GovReport, QMSum, MultiNews                (summarization)

Data: Xnhyacinth/LongBench parquet mirror of THUDM/LongBench v1
(original JSONL revision no longer served by THUDM). Prompts are reassembled
from the mirror's context/question/answer_prefix fields and verified against
the official LongBench v1 prompt templates (GitHub commit 4c4b985bcf).
Metrics follow the official eval.py/metrics.py at that commit:
  trec -> classification_score; triviaqa -> qa_f1_score;
  samsum/gov_report/qmsum/multi_news -> rouge_score (ROUGE-L F);
  lcc/repobench-p -> code_sim_score (fuzzywuzzy ratio).
Per-row max_new_tokens comes from the mirror's per-sample field.

Main protocol mirrors the reasoning benchmarks: --disable-thinking (chat
template, enable_thinking=False), greedy, single seed, atomic JSON + sha256
per cell, resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import sys
import time
import uuid
from pathlib import Path


MODEL_2B = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
MODEL_9B = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-9B/snapshots/master"
DATA_DIR = Path("/root/autodl-tmp/MLSys_Research/data/longbench")
TASKS = ["trec", "triviaqa", "samsum", "lcc", "repobench-p", "gov_report", "qmsum", "multi_news"]
ALLOCATIONS = ["fp16", "uniform_int4", "packed_per_layer", "turboquant_k8v4", "turboquant_4bit_nc"]
OFFICIAL_LONGBENCH_COMMIT = "4c4b985bcf"


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


def build_prompt(task: str, row: dict) -> str:
    context = str(row["context"])
    question = str(row["question"])
    prefix = str(row["answer_prefix"])
    if question.strip():
        if question.rstrip().endswith(prefix.strip()):
            prompt = context.rstrip("\n") + "\n" + question
        else:
            prompt = (
                context.rstrip("\n") + "\n" + question.rstrip("\n") + "\n" + prefix
            )
    else:
        prompt = context.rstrip("\n") + "\n" + prefix
    return prompt


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    from collections import Counter

    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def classification_score(prediction: str, ground_truth: str, all_classes: list[str]) -> float:
    matched = [c for c in all_classes if c in prediction]
    for term in list(matched):
        if term in ground_truth and term != ground_truth:
            matched.remove(term)
    if ground_truth in matched:
        return 1.0 / len(matched)
    return 0.0


def rouge_l_f(prediction: str, ground_truth: str) -> float:
    from rouge import Rouge

    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return float(scores["rouge-l"]["f"])


def code_sim_score(prediction: str, ground_truth: str) -> float:
    from fuzzywuzzy import fuzz

    for line in prediction.lstrip("\n").split("\n"):
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100.0


TASK_METRIC = {
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l",
    "lcc": "code_sim",
    "repobench-p": "code_sim",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
}


def score_case(task: str, prediction: str, row: dict) -> float:
    metric = TASK_METRIC[task]
    if task in {"trec", "triviaqa", "samsum"}:
        prediction = prediction.lstrip("\n").split("\n")[0]
    best = 0.0
    for truth in row["answers"]:
        truth = str(truth)
        if metric == "classification":
            score = classification_score(prediction, truth, [str(c) for c in row["all_classes"]])
        elif metric == "qa_f1":
            score = qa_f1_score(prediction, truth)
        elif metric == "rouge_l":
            score = rouge_l_f(prediction, truth)
        elif metric == "code_sim":
            score = code_sim_score(prediction, truth)
        else:
            raise SystemExit(f"unknown metric: {metric}")
        best = max(best, score)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--allocation", required=True, choices=ALLOCATIONS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default=MODEL_2B)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--out-dir", default="results/quality/longbench")
    ap.add_argument("--attempt-id", default="longbench-20260807")
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data_path = DATA_DIR / f"{args.task}.parquet"
    if not data_path.exists():
        raise SystemExit(f"data missing: {data_path}")
    data_sha = sha256_file(data_path)
    thinking = "disabled" if args.disable_thinking else "default"
    model_tag = "9b" if "9B" in args.model else "2b"
    out_path = (
        Path(args.out_dir)
        / args.attempt_id
        / f"{args.task}__{args.allocation}__s{args.seed}__{model_tag}.json"
    )
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed_validated"
            and existing.get("data_sha256") == data_sha
            and existing.get("thinking") == thinking
            and existing.get("max_samples") == args.max_samples
            and existing.get("model") == args.model
            and existing.get("engine", {}).get("kwargs", {}).get("max_model_len") == args.max_model_len
        ):
            print(f"resume: skip {out_path}")
            return 0

    import pandas as pd

    df = pd.read_parquet(data_path)
    rows = df.head(args.max_samples).to_dict(orient="records")
    if len(rows) != args.max_samples:
        raise SystemExit(f"rows mismatch: {len(rows)} != {args.max_samples}")

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

    tokenizer = None
    if args.disable_thinking:
        from transformers import AutoTokenizer  # noqa: PLC0415

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = []
    for row in rows:
        prompt = build_prompt(args.task, row)
        if args.disable_thinking:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        prompts.append(prompt)

    # Middle truncation, official LongBench style, bounded by max_model_len - output budget.
    tokenizer_for_trunc = tokenizer or llm.get_tokenizer()
    max_out = int(max(int(row["max_new_tokens"]) for row in rows))
    max_input = args.max_model_len - max_out
    tokenized = tokenizer_for_trunc(prompts, truncation=False, add_special_tokens=False)
    input_ids = tokenized.input_ids
    final_prompts = []
    trunc_counts = 0
    for prompt, ids in zip(prompts, input_ids):
        if len(ids) > max_input:
            half = max_input // 2
            head = tokenizer_for_trunc.decode(ids[:half], skip_special_tokens=True)
            tail = tokenizer_for_trunc.decode(ids[-half:], skip_special_tokens=True)
            final_prompts.append(head + tail)
            trunc_counts += 1
        else:
            final_prompts.append(prompt)

    outputs = llm.generate(
        final_prompts,
        SamplingParams(max_tokens=max_out, temperature=0.0),
        use_tqdm=True,
    )
    cases = []
    total_score = 0.0
    for row, prompt, output in zip(rows, final_prompts, outputs):
        prediction = output.outputs[0].text
        case_score = score_case(args.task, prediction, row)
        total_score += case_score
        cases.append(
            {
                "_id": str(row["_id"]),
                "prediction": prediction,
                "answers": [str(a) for a in row["answers"]],
                "all_classes": [str(c) for c in row["all_classes"]] if row["all_classes"] is not None else None,
                "length": int(row["length"]),
                "max_new_tokens": int(row["max_new_tokens"]),
                "prompt_tokens": len(output.prompt_token_ids),
                "output_tokens": len(output.outputs[0].token_ids),
                "score": round(case_score, 4),
            }
        )

    record = {
        "schema_version": 1,
        "attempt_id": args.attempt_id,
        "status": "completed_validated",
        "task": args.task,
        "allocation": args.allocation,
        "seed": args.seed,
        "model": args.model,
        "num_samples": len(cases),
        "max_samples": args.max_samples,
        "score": round(100.0 * total_score / len(cases), 4),
        "metric": TASK_METRIC[args.task],
        "data_source": "Xnhyacinth/LongBench parquet mirror (THUDM/LongBench v1)",
        "data_sha256": data_sha,
        "official_longbench_commit": OFFICIAL_LONGBENCH_COMMIT,
        "thinking": thinking,
        "truncated_prompts": trunc_counts,
        "max_input_tokens": max_input,
        "max_output_tokens": max_out,
        "cases": cases,
        "engine": {
            "model": args.model,
            "kwargs": {k: v for k, v in kwargs.items() if k != "model"},
            "vllm_version": vllm_version,
        },
        "config_effect": effect,
        "sampling_params": {"max_tokens": max_out, "temperature": 0.0},
        "elapsed_s": round(time.time() - t0, 1),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": "remote_5090",
    }
    digest = atomic_write_json(out_path, record)
    print(f"LONGBENCH({args.task}, {args.allocation}, seed={args.seed}) score={record['score']:.4f}")
    print(f"→ {out_path} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
