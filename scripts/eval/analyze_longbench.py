"""Aggregate LongBench cells: per (model, task, allocation) scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = ["trec", "triviaqa", "samsum", "lcc", "repobench-p", "gov_report", "qmsum", "multi_news"]
ALLOCATIONS = ["fp16", "uniform_int4", "packed_per_layer", "turboquant_k8v4", "turboquant_4bit_nc"]
TASK_LABEL = {
    "trec": "TREC",
    "triviaqa": "TriviaQA",
    "samsum": "SAMSum",
    "lcc": "LCC",
    "repobench-p": "RepoBench-P",
    "gov_report": "GovReport",
    "qmsum": "QMSum",
    "multi_news": "MultiNews",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/longbench")
    ap.add_argument("--attempt", default="longbench-20260807")
    ap.add_argument("--out", default="results/quality/longbench-analysis-20260807.json")
    args = ap.parse_args()

    cells: dict[tuple[str, str, str], dict] = {}
    for path in sorted((Path(args.dir) / args.attempt).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        model = "9B" if "9B" in rec["model"] else "2B"
        cells[(model, rec["task"], rec["allocation"])] = rec

    missing = [
        (m, t, a)
        for m in ("2B", "9B")
        for t in TASKS
        for a in ALLOCATIONS
        if (m, t, a) not in cells and not (m == "9B" and a in ("turboquant_k8v4", "turboquant_4bit_nc"))
    ]
    if missing:
        raise SystemExit(f"incomplete cells: {len(missing)} missing, e.g. {missing[:8]}")

    models = ["2B", "9B"]
    tables = {}
    for model in models:
        rows = []
        for task in TASKS:
            row = {
                "task": TASK_LABEL[task],
                "metric": cells[(model, task, "fp16")]["metric"],
                "n": cells[(model, task, "fp16")]["num_samples"],
            }
            for alloc in ALLOCATIONS:
                if (model, task, alloc) in cells:
                    row[alloc] = cells[(model, task, alloc)]["score"]
                    row[f"trunc_{alloc}"] = cells[(model, task, alloc)]["truncated_prompts"]
            rows.append(row)
        tables[model] = rows

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "tables": tables,
        "notes": {
            "data": "Xnhyacinth/LongBench parquet mirror (THUDM/LongBench v1)",
            "official_commit": "4c4b985bcf",
            "protocol": "no-think, greedy, seed 7, first 50 samples/task, max_model_len 16384",
            "metric_legend": {
                "classification": "official classification_score (TREC)",
                "qa_f1": "official qa_f1_score (TriviaQA)",
                "rouge_l": "official rouge_score ROUGE-L F (SAMSum/GovReport/QMSum/MultiNews)",
                "code_sim": "official code_sim_score fuzzywuzzy ratio (LCC/RepoBench-P)",
            },
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
