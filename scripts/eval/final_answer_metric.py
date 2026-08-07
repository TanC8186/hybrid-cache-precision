"""Post-hoc 'final answer' metric for NIAH cells.

For each stored case, the final answer section is the text after the LAST
case-insensitive "Answer:" marker in the generated text (fallback: whole
text). `hit_last_section` requires the needle code to appear there. This
handles the Qwen3.5-9B repetition pattern (code first, then repeated
Question/Answer blocks) without rerunning anything; original JSONs are
immutable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CODE_RE = re.compile(r"[A-HJ-NP-Z2-9]{8}")


def final_answer_section(answer: str) -> str:
    marker = "Answer:"
    lower = answer.lower()
    positions = [i for i in range(len(lower)) if lower.startswith(marker, i)]
    if not positions:
        return answer
    return answer[positions[-1] + len(marker) :]


def hit_in_section(section: str, code: str) -> bool:
    return code in section.upper().replace(" ", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/quality/niah-fixed")
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records: dict[str, list[dict]] = {}
    for path in sorted((Path(args.dir) / args.attempt).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            continue
        alloc = rec["allocation"]
        cases = []
        for case in rec["cases"]:
            section = final_answer_section(case["answer"])
            cases.append(
                {
                    "code": case["code"],
                    "hit": bool(case["hit"]),
                    "hit_final": bool(case.get("hit_final", False)),
                    "hit_last_section": hit_in_section(section, case["code"]),
                    "last_section_len": len(section),
                    "answer_len": len(case["answer"]),
                    "output_tokens": case.get("output_tokens"),
                }
            )
        records.setdefault(alloc, []).extend(cases)

    rows = []
    for alloc in sorted(records):
        cases = records[alloc]
        n = len(cases)
        rows.append(
            {
                "allocation": alloc,
                "n_needles": n,
                "hit_anywhere": round(sum(c["hit"] for c in cases) / n, 4),
                "hit_final": round(sum(c["hit_final"] for c in cases) / n, 4),
                "hit_last_section": round(sum(c["hit_last_section"] for c in cases) / n, 4),
                "disagreements_hit_vs_last": sum(c["hit"] != c["hit_last_section"] for c in cases),
            }
        )
        print(
            f"{alloc:<20} n={n:<3} hit={rows[-1]['hit_anywhere']:<7} "
            f"hit_final={rows[-1]['hit_final']:<7} hit_last_section={rows[-1]['hit_last_section']:<7} "
            f"disagree={rows[-1]['disagreements_hit_vs_last']}"
        )

    result = {"schema_version": 1, "attempt": args.attempt, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
