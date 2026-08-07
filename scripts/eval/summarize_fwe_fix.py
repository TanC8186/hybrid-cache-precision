"""Summarize the FWE rerun (attempt ruler-fwe-fixed-20260807, max_tokens=256)."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    base = Path("results/quality/ruler-subset/ruler-fwe-fixed-20260807")
    rows = []
    for path in sorted(base.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("status") != "completed_validated":
            raise SystemExit(f"not completed: {path}")
        rows.append(
            {
                "allocation": rec["allocation"],
                "length": rec["length"],
                "accuracy": rec["accuracy"],
                "max_tokens": rec["max_tokens"],
                "num_samples": rec["num_samples"],
                "elapsed_s": rec["elapsed_s"],
            }
        )
    rows.sort(key=lambda r: (r["allocation"], r["length"]))
    for row in rows:
        print(
            f"{row['allocation']:<16} L{row['length']:<5} "
            f"acc={row['accuracy']:<7} max_tokens={row['max_tokens']} "
            f"n={row['num_samples']} elapsed={row['elapsed_s']}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
