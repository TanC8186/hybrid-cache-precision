"""Per-layer GDN state-dtype sensitivity scan (decision gate).

Protocol (matches the committed PPL matrix, `ppl-state-20260808`):
  Qwen3.5-2B, C4/PG19 slices, seeds 7/42/2026, 5 seqs x 2048 tokens,
  chunk 128, attention KV fp16 (bits=16). Configs per seed:
    - fp32    : no cast (reference baseline)
    - bf16_Li : only GDN layer i's recurrent state is cast to bf16
    - bf16_all: every GDN layer's recurrent state is cast to bf16

Artifact guards (fail-closed):
  1. Per-config audit records which layers were actually cast and the written
     dtype; the run aborts if the intended layer(s) were not cast or if any
     non-target GDN layer changed dtype.
  2. fp32 and bf16_all per-seed PPL must bit-match the committed
     `ppl-state-20260808` 2B seeds CSV (same model/corpus/seeds/params).
  3. Atomic JSON + sha256 output, resumable per corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL_DEFAULT = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
REFERENCE_ROOT = REPO_ROOT / "results" / "quality" / "ppl-state-dtype"


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


def t_half(n: int, sd: float) -> float:
    df = n - 1
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    }
    return table.get(df, 1.96) * sd / math.sqrt(n)


def load_reference_seeds(corpus: str, state_tag: str) -> dict[int, float]:
    path = REFERENCE_ROOT / f"ppl-state-20260808__{corpus}__state{state_tag}__2b.csv.seeds.csv"
    if not path.exists():
        raise SystemExit(f"reference missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen: dict[tuple[str, str], float] = {}
    for r in rows:
        seen.setdefault((r["bits"], r["seed"]), float(r["ppl"]))
    return {int(seed): ppl for (bits, seed), ppl in seen.items() if bits == "16"}


def assert_audit(audit: dict, gdn_indices: list[int], layer_ids: set[int] | None) -> None:
    errors: list[str] = []
    if layer_ids is None:
        for idx in gdn_indices:
            if audit["cast_calls"].get(idx, 0) <= 0:
                errors.append(f"L{idx} was not cast (bf16_all)")
            if audit["written_dtypes"].get(idx) != "torch.bfloat16":
                errors.append(f"L{idx} written dtype {audit['written_dtypes'].get(idx)} != bfloat16")
    else:
        for idx in gdn_indices:
            if idx in layer_ids:
                if audit["cast_calls"].get(idx, 0) <= 0:
                    errors.append(f"L{idx} was not cast (target)")
                if audit["written_dtypes"].get(idx) != "torch.bfloat16":
                    errors.append(f"L{idx} written dtype {audit['written_dtypes'].get(idx)} != bfloat16")
            else:
                if audit["cast_calls"].get(idx, 0):
                    errors.append(f"L{idx} cast unexpectedly")
                if audit["written_dtypes"].get(idx) != "torch.float32":
                    errors.append(f"L{idx} written dtype {audit['written_dtypes'].get(idx)} != float32")
    if errors:
        raise SystemExit("state-cast audit failed: " + "; ".join(errors[:8]))


def audit_summary(audit: dict) -> dict:
    return {
        "n_layers_cast": len(audit["cast_calls"]),
        "min_cast_calls": min(audit["cast_calls"].values()) if audit["cast_calls"] else 0,
        "written_dtype_set": sorted(set(audit["written_dtypes"].values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--corpora", default="c4,pg19")
    ap.add_argument("--seeds", default="7,42,2026")
    ap.add_argument("--num-seqs", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out-dir", default="results/quality/state-sensitivity")
    ap.add_argument("--attempt-id", default="state-sensitivity-20260809")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hybrid_premise import (
        _SMOKE_CORPUS,
        attention_layer_indices,
        chunked_ppl,
        make_cache,
        tokenize_corpus,
    )

    harness_path = Path(__file__).resolve().parent / "hybrid_premise.py"
    harness_sha256 = sha256_file(harness_path)

    print(f"loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="eager"
    ).cuda()
    model.eval()

    attn_indices = attention_layer_indices(model)
    gdn_indices = [
        i for i in range(model.config.num_hidden_layers) if i not in set(attn_indices)
    ]
    print(f"attention layers: {attn_indices}; GDN layers: {gdn_indices}")

    corpora = ["c4", "pg19"] if not args.smoke else ["c4"]
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",")]
    max_len = 512 if args.smoke else args.max_len
    chunk = 64 if args.smoke else args.chunk
    num_seqs = 1 if args.smoke else args.num_seqs
    out_root = Path(args.out_dir) / args.attempt_id

    configs: list[tuple[str, str | None, set[int] | None]] = [
        ("fp32", None, None),
        ("bf16_all", "bfloat16", None),
    ] + [(f"bf16_L{i}", "bfloat16", {i}) for i in gdn_indices]

    for corpus in corpora:
        out_path = out_root / f"{corpus}.json"
        if args.resume and out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed_validated":
                print(f"resume: skip {out_path}")
                continue

        corpus_text = (_SMOKE_CORPUS * 3) if args.smoke else (
            (REPO_ROOT / "data" / f"{corpus}_slice.txt").read_text(encoding="utf-8")
        )
        ref_fp32 = None if args.smoke else load_reference_seeds(corpus, "fp32")
        ref_bf16 = None if args.smoke else load_reference_seeds(corpus, "bf16")

        per_seed: dict[str, dict[int, float]] = {name: {} for name, _, _ in configs}
        audit_summaries: dict[str, dict[int, dict]] = {}
        ref_diffs: dict[str, dict[int, float]] = {}
        t0 = time.time()

        for seed in seeds:
            ids_list = tokenize_corpus(tokenizer, corpus_text, max_len, num_seqs, seed=seed)
            for config_name, dtype, layer_ids in configs:
                audit: dict | None = {} if dtype is not None else None
                total_loss = 0.0
                total_tokens = 0
                for ids in ids_list:
                    ppl, _qb, _fb = chunked_ppl(
                        model,
                        tokenizer,
                        ids,
                        lambda: make_cache(16, None, attn_indices, model),
                        chunk,
                        attn_indices=attn_indices,
                        state_dtype=dtype,
                        state_layer_ids=layer_ids,
                        state_audit=audit,
                    )
                    total_loss += math.log(ppl) * (ids.shape[1] - 1)
                    total_tokens += ids.shape[1] - 1
                ppl = math.exp(total_loss / total_tokens)
                per_seed[config_name][seed] = ppl
                if audit is not None:
                    assert_audit(audit, gdn_indices, layer_ids)
                    audit_summaries.setdefault(config_name, {})[seed] = audit_summary(audit)
                print(f"[{corpus}] seed={seed} {config_name}: PPL={ppl:.6f}")

            if not args.smoke:
                for tag, ref, key in (("fp32", ref_fp32, "fp32"), ("bf16", ref_bf16, "bf16_all")):
                    diff = abs(per_seed[key][seed] - ref[seed])
                    ref_diffs.setdefault(tag, {})[seed] = diff
                    if diff > 1e-9:
                        raise SystemExit(
                            f"reference mismatch {corpus} seed={seed} {tag}: "
                            f"ours={per_seed[key][seed]:.12f} ref={ref[seed]:.12f} diff={diff:.3e}"
                        )

        rows = []
        for config_name, _dtype, _layer_ids in configs:
            vals = [per_seed[config_name][s] for s in seeds]
            diffs = [per_seed[config_name][s] - per_seed["fp32"][s] for s in seeds]
            mean_d = statistics.mean(diffs)
            half = t_half(len(diffs), statistics.stdev(diffs)) if len(diffs) > 1 else 0.0
            rows.append(
                {
                    "config": config_name,
                    "ppl_mean": round(statistics.mean(vals), 6),
                    "ppl_std": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
                    "per_seed": {str(s): round(per_seed[config_name][s], 6) for s in seeds},
                    "delta_vs_fp32_mean": round(mean_d, 6),
                    "ci95_delta": [round(mean_d - half, 6), round(mean_d + half, 6)],
                    "audit": audit_summaries.get(config_name, {}),
                }
            )

        record = {
            "schema_version": 1,
            "attempt_id": args.attempt_id,
            "status": "completed_validated",
            "corpus": corpus,
            "model": args.model,
            "harness_sha256": harness_sha256,
            "protocol": {
                "seeds": seeds,
                "num_seqs": num_seqs,
                "max_len": max_len,
                "chunk": chunk,
                "attention_kv": "fp16 (bits=16)",
                "configs": [c[0] for c in configs],
                "gdn_indices": gdn_indices,
            },
            "reference_match": {
                "note": "fp32/bf16_all per-seed PPL must bit-match ppl-state-20260808 2B seeds CSV",
                "max_abs_diff": {tag: max(diffs.values()) for tag, diffs in ref_diffs.items()},
            },
            "rows": rows,
            "elapsed_s": round(time.time() - t0, 1),
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        digest = atomic_write_json(out_path, record)
        print(f"-> {out_path} (sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
