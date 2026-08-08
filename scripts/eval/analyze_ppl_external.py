"""Aggregate the KIVI-style external-baseline PPL cells (paired 3-seed CI)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def t_half(n: int, sd: float) -> float:
    df = n - 1
    table = {1: 12.706, 2: 4.303, 3: 3.182}
    return table.get(df, 1.96) * sd / math.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/ppl-external")
    ap.add_argument("--attempt", default="ppl-external-20260808")
    ap.add_argument("--out", default="results/quality/ppl-external-analysis-20260808.json")
    args = ap.parse_args()

    cells: dict[tuple[str, str], list[float]] = {}
    for path in (Path(args.dir) / args.attempt).glob("*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        cells[(rec["corpus"], rec["backend"])] = [r["ppl"] for r in rec["rows"]]

    corpora = ["wikitext2", "c4", "pg19"]
    missing = [(c, b) for c in corpora for b in ("fp16", "kivi4_hqq") if (c, b) not in cells]
    if missing:
        raise SystemExit(f"incomplete: {missing}")

    rows = []
    for corpus in corpora:
        fp16 = cells[(corpus, "fp16")]
        kivi = cells[(corpus, "kivi4_hqq")]
        diffs = [k - f for k, f in zip(kivi, fp16)]
        mean_d = statistics.mean(diffs)
        half = t_half(len(diffs), statistics.stdev(diffs))
        rows.append(
            {
                "corpus": corpus,
                "fp16_per_seed": [round(x, 4) for x in fp16],
                "kivi4_per_seed": [round(x, 4) for x in kivi],
                "fp16_mean": round(statistics.mean(fp16), 4),
                "kivi4_mean": round(statistics.mean(kivi), 4),
                "delta_kivi4_vs_fp16": round(mean_d, 4),
                "ci95": [round(mean_d - half, 4), round(mean_d + half, 4)],
            }
        )

    result = {
        "schema_version": 1,
        "attempt": args.attempt,
        "method": "KIVI-style 4-bit KV quantization (K per-channel group32, V per-token, residual 128) via transformers HQQ backend",
        "protocol": "seeds 7/42/2026, num_seqs 5, max_len 2048, chunk 128; same-harness fp16 reference",
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
