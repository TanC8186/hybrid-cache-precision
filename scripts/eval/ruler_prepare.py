"""Generate RULER-subset datasets with the official RULER generators.

Replicates the command construction of the official `scripts/data/prepare.py`
but loads task definitions from `vendor/ruler/ruler_subset.yaml` and only
uses noise-haystack tasks (no nltk punkt corpus required). The official task
scripts, templates, and constants are used verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import yaml


RULER_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "ruler"
DATA_SCRIPTS = RULER_ROOT / "scripts" / "data"
SYNTHETIC_DIR = DATA_SCRIPTS / "synthetic"
SUBSET_YAML = RULER_ROOT / "ruler_subset.yaml"
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tasks_base() -> dict:
    spec = importlib.util.spec_from_file_location(
        "ruler_tasks_base", SYNTHETIC_DIR / "constants.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASKS


def load_templates() -> dict:
    sys.path.insert(0, str(DATA_SCRIPTS))
    from template import Templates  # noqa: PLC0415

    return Templates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--length", type=int, required=True, choices=[4096, 8192])
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument(
        "--tokenizer-path",
        default="/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master",
    )
    ap.add_argument("--save-dir", default="data/ruler")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with SUBSET_YAML.open(encoding="utf-8") as handle:
        subset = yaml.safe_load(handle)
    if args.task not in subset:
        raise SystemExit(f"task {args.task!r} not found in {SUBSET_YAML}")

    tasks_base = load_tasks_base()
    config = dict(subset[args.task])
    base = tasks_base[config["task"]]
    config.update(base)
    templates = load_templates()
    template = templates["base"].format(task_template=config["template"]) + config["answer_prefix"]

    save_dir = Path(args.save_dir) / f"{args.task}_L{args.length}"
    data_file = save_dir / "validation.jsonl"
    if data_file.exists() and not args.force:
        print(f"skip (exists): {data_file} sha256={sha256_file(data_file)}")
        return 0

    script = SYNTHETIC_DIR / f"{config['task']}.py"
    if not script.exists():
        raise SystemExit(f"missing official task script: {script}")

    cmd = [
        sys.executable,
        str(script),
        "--save_dir",
        str(save_dir),
        "--save_name",
        f"{args.task}_L{args.length}",
        "--subset",
        "validation",
        "--tokenizer_path",
        args.tokenizer_path,
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(args.length),
        "--tokens_to_generate",
        str(config["tokens_to_generate"]),
        "--num_samples",
        str(args.num_samples),
        "--random_seed",
        str(args.random_seed),
    ]
    for key, value in config["args"].items():
        cmd += [f"--{key}", str(value)]
    cmd += ["--template", template]

    print("generating:", " ".join(cmd))
    import subprocess

    completed = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout[-4000:])
    if completed.returncode != 0:
        raise SystemExit(f"generator failed rc={completed.returncode}")
    if not data_file.exists():
        raise SystemExit(f"generator exited 0 but no dataset at {data_file}")

    rows = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != args.num_samples:
        raise SystemExit(f"row count mismatch: {len(rows)} != {args.num_samples}")
    digest = sha256_file(data_file)
    Path(str(data_file) + ".sha256").write_text(f"{digest}\n", encoding="ascii")
    manifest = {
        "task": args.task,
        "length": args.length,
        "num_samples": args.num_samples,
        "random_seed": args.random_seed,
        "tokenizer_path": args.tokenizer_path,
        "ruler_commit": RULER_COMMIT,
        "template": template,
        "data_file": str(data_file),
        "sha256": digest,
    }
    manifest_path = save_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
