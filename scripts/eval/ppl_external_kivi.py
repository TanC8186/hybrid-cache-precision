"""External KV-quantization PPL baseline: KIVI-style 4-bit via transformers HQQ.

Protocol mirrors the canonical PPL harness (hybrid_premise.py):
  --seeds 7,42,2026 --num-seqs 5 --max-len 2048 --chunk 128
Scheme (KIVI-inspired, ICML'24):
  - keys: per-channel (head-dim) quantization, group size 32, 4-bit;
  - values: per-token quantization, 4-bit;
  - a 128-token residual fp16 window; older tokens are quantized once.
Implementation: subclass transformers 5.x DynamicCache (keeps GDN recurrent
state methods intact) and override attention-layer update() only.
--backend fp16 computes the same-harness reference without quantization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from transformers.cache_utils import DynamicCache


MODEL_2B = "/root/autodl-tmp/caches/modelscope/models/Qwen--Qwen3.5-2B/snapshots/master"
DATA_FILES = {
    "wikitext2": "/root/autodl-tmp/MLSys_Research/data/wikitext2_test.txt",
    "c4": "/root/autodl-tmp/MLSys_Research/data/c4_slice.txt",
    "pg19": "/root/autodl-tmp/MLSys_Research/data/pg19_slice.txt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_corpus(tokenizer, corpus: str, max_len: int, num_seqs: int, seed: int | None = None):
    ids = tokenizer(corpus, return_tensors="pt").input_ids[0]
    total = ids.numel()
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        lo = 0
        hi = max(lo + 1, total - max_len)
        starts = torch.randint(lo, hi, (num_seqs,), generator=generator).tolist()
    else:
        starts = list(range(0, total - 1, max_len))[:num_seqs]
    return [ids[s : s + max_len] for s in starts]


class KiviKVStore:
    """Per-layer KIVI-style store: residual fp16 window + HQQ-quantized blocks."""

    def __init__(self, residual_length: int = 128, group_size: int = 32, nbits: int = 4):
        self.residual_length = residual_length
        self.group_size = group_size
        self.nbits = nbits
        self.raw_k: list[torch.Tensor] = []
        self.raw_v: list[torch.Tensor] = []
        self.quant_blocks: list[tuple] = []

    @staticmethod
    def _quantize(tensor: torch.Tensor, group_size: int, per_token: bool) -> tuple:
        from transformers.cache_utils import HQQQuantizer

        flat = tensor.to(torch.float16).reshape(-1, tensor.shape[-1])
        # per-token: each row is one token -> group covers the full head dim
        gs = tensor.shape[-1] if per_token else group_size
        return HQQQuantizer.quantize(
            flat,
            nbits=4,
            channel_wise=True,
            group_size=gs,
            optimize=False,
            axis=0,
            bitpack=False,
            device=str(tensor.device),
        )

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.raw_k.append(key_states.detach())
        self.raw_v.append(value_states.detach())
        while sum(t.shape[2] for t in self.raw_k) > self.residual_length:
            block_k = self.raw_k.pop(0)
            block_v = self.raw_v.pop(0)
            self.quant_blocks.append(
                (
                    (self._quantize(block_k, self.group_size, False), block_k.shape),
                    (self._quantize(block_v, self.group_size, True), block_v.shape),
                )
            )
        return self.get_full()

    def get_full(self) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers.cache_utils import HQQQuantizer

        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []
        for (qk, shape_k), (qv, shape_v) in self.quant_blocks:
            parts_k.append(HQQQuantizer.dequantize(*qk).to(dtype=torch.bfloat16).view(shape_k))
            parts_v.append(HQQQuantizer.dequantize(*qv).to(dtype=torch.bfloat16).view(shape_v))
        parts_k.extend(t for t in self.raw_k)
        parts_v.extend(t for t in self.raw_v)
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)


class KiviHybridCache(DynamicCache):
    """DynamicCache subclass: quantizes attention KV (KIVI-style), keeps GDN state."""

    def __init__(self, config, attention_layers: set[int], residual_length: int = 128):
        super().__init__(config=config)
        self.attention_layers = attention_layers
        self.first_attention_idx = min(attention_layers)
        self.stores: dict[int, KiviKVStore] = {}
        self.residual_length = residual_length
        self._tokens = 0

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        n = int(key_states.shape[2])
        if layer_idx in self.attention_layers:
            if layer_idx == self.first_attention_idx:
                self._tokens += n
            store = self.stores.setdefault(layer_idx, KiviKVStore(self.residual_length))
            return store.append(key_states, value_states)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        return self._tokens

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        # Attention layers bypass DynamicCache's internal layer storage, so the
        # inherited implementation would see an empty layer and assume no past.
        return self._tokens + query_length, 0

    def reset(self):
        super().reset()
        self._tokens = 0
        self.stores.clear()


def attention_layer_indices(model) -> set[int]:
    types = getattr(model.config, "layer_types", None)
    if types is None:
        return set(range(model.config.num_hidden_layers))
    return {i for i, t in enumerate(types) if t == "full_attention"}


def compute_ppl(model, tokenizer, ids_list, chunk: int, attention_layers: set[int], quantize: bool) -> float:
    total_loss = 0.0
    total_tokens = 0
    for ids in ids_list:
        if quantize:
            cache = KiviHybridCache(model.config, attention_layers)
        else:
            cache = DynamicCache(config=model.config)
        prev_last_logits = None
        for start in range(0, ids.numel() - 1, chunk):
            end = min(start + chunk, ids.numel())
            chunk_ids = ids[start:end].unsqueeze(0).to(model.device)
            with torch.no_grad():
                outputs = model(chunk_ids, past_key_values=cache, use_cache=True)
            logits = outputs.logits[0, :-1, :]  # [L-1, V]
            targets = chunk_ids[0, 1:]
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
            if prev_last_logits is not None:
                loss = loss + torch.nn.functional.cross_entropy(
                    prev_last_logits.unsqueeze(0), chunk_ids[0, :1]
                )
                total_tokens += 1
            total_loss += loss.item()
            total_tokens += logits.shape[0]
            prev_last_logits = outputs.logits[0, -1, :]
    return math.exp(total_loss / total_tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=list(DATA_FILES), required=True)
    ap.add_argument("--backend", choices=["fp16", "kivi4_hqq"], required=True)
    ap.add_argument("--model", default=MODEL_2B)
    ap.add_argument("--seeds", default="7,42,2026")
    ap.add_argument("--num-seqs", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out-dir", default="results/quality/ppl-external")
    ap.add_argument("--attempt-id", default="ppl-external-kivi-20260808")
    args = ap.parse_args()

    corpus_path = Path(DATA_FILES[args.corpus])
    corpus = corpus_path.read_text(encoding="utf-8")
    corpus_sha = sha256_file(corpus_path)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    attention_layers = attention_layer_indices(model)

    rows: list[tuple[int, float]] = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        ids_list = tokenize_corpus(tokenizer, corpus, args.max_len, args.num_seqs, seed=seed)
        t0 = time.time()
        ppl = compute_ppl(model, tokenizer, ids_list, args.chunk, attention_layers, args.backend != "fp16")
        rows.append((seed, ppl))
        print(f"seed={seed} backend={args.backend} corpus={args.corpus} PPL={ppl:.4f} [{time.time()-t0:.0f}s]")

    out_dir = Path(args.out_dir) / args.attempt_id
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{args.corpus}__{args.backend}.csv"
    with base.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backend", "corpus", "ppl_mean", "ppl_std", "num_seeds"])
        values = [ppl for _, ppl in rows]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        writer.writerow([args.backend, args.corpus, f"{statistics.mean(values):.4f}", f"{std:.4f}", len(values)])
    seeds_path = Path(str(base) + ".seeds.csv")
    with seeds_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["backend", "corpus", "seed", "ppl"])
        writer.writerows([args.backend, args.corpus, seed, ppl] for seed, ppl in rows)
    manifest = {
        "backend": args.backend,
        "corpus": args.corpus,
        "corpus_sha256": corpus_sha,
        "model": args.model,
        "protocol": f"seeds={args.seeds}, num_seqs={args.num_seqs}, max_len={args.max_len}, chunk={args.chunk}",
        "scheme": "KIVI-style: K per-channel group32 4-bit, V per-token 4-bit, residual 128 (HQQ backend)" if args.backend != "fp16" else "fp16 reference (same harness)",
        "rows": [{"seed": s, "ppl": p} for s, p in rows],
        "ppl_mean": round(statistics.mean(values), 4),
        "ppl_std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
    }
    manifest_path = Path(str(base) + ".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"→ {base} (+seeds.csv, +manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
