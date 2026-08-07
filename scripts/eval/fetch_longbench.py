"""Fetch the LongBench v1 parquet mirror used by the paper subset.

THUDM/LongBench's current HF repo only serves LongBench-v2 (data.zip), and the
original v1 JSONL revision is no longer exposed, so we pin the widely mirrored
Xnhyacinth/LongBench parquet files (same official v1 content, reformatted into
context/question/answer_prefix rows; see data README). Sizes are verified
against the repo tree listing to catch truncated downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


MIRROR = "https://hf-mirror.com/datasets/Xnhyacinth/LongBench/resolve/main"
TASKS = {
    "trec": 2_671_201,
    "triviaqa": 6_304_565,
    "samsum": 4_118_162,
    "lcc": 2_352_136,
    "repobench-p": 7_768_093,
    "gov_report": 5_496_673,
    "qmsum": 972_703,
    "multi_news": 1_497_033,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/longbench")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for task, expected in TASKS.items():
        target = out_dir / f"{task}.parquet"
        if not (target.exists() and target.stat().st_size == expected):
            subprocess.run(
                [
                    "curl",
                    "-sL",
                    "--max-time",
                    "300",
                    "-o",
                    str(target),
                    f"{MIRROR}/{task}/test-00000-of-00001.parquet",
                ],
                check=True,
            )
        actual = target.stat().st_size
        if actual != expected:
            print(f"size mismatch {task}: {actual} != {expected}")
            return 1
        manifest[task] = {
            "size": actual,
            "sha256": sha256_file(target),
            "source": f"{MIRROR}/{task}/test-00000-of-00001.parquet",
        }
        print(f"{task}: {actual} bytes sha256={manifest[task]['sha256']}")
    manifest_path = out_dir / "longbench_data_manifest.json"
    import json

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest → {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
