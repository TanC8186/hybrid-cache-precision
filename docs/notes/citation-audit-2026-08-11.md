# Citation Audit — mlsys2026 (2026-08-11)

Scope: `paper/mlsys2026/main.bib` (30 entries at audit start; 32 after adding
`griffin` and `deltanet`) and every `\cite` used in `main.tex`.

Method: arXiv API metadata cross-check for arXiv-hosted items; ACL Anthology
for HqeKV/QPruningKV; publisher/venue pages for ICLR/COLM/NeurIPS items;
official project pages for model cards where available. Each entry below is
marked `VERIFIED` (metadata matches), `NEEDS FIX` (fixed in this pass), or
`BOUNDED` (metadata matches within an explicitly noted boundary).

## Fixed in this pass

| Key | Problem | Fix | Source |
|---|---|---|---|
| `turboquant` | Title/venue were wrong: paper is *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* (arXiv:2504.19874, ICLR 2026 poster), not an extreme KV-cache-compression paper | Replaced title, type, venue/year; kept arXiv eprint | arXiv abs page; ICLR 2026 virtual site |
| `jamba` | Author list did not match the official 22-author arXiv author list | Replaced the whole `author` field with the official list | arXiv:2403.19887 abs page |
| `recurrentgemma` | Last author misspelled `de Frietas` | `de Frietas, Nando` → `de Freitas, Nando` | arXiv API; Google DeepMind Mila author page |
| `streamingllm` | Cited as arXiv 2023 article; published at ICLR 2024 | `@article` → `@inproceedings`; `booktitle={ICLR}, year={2024}` | OpenReview/ICLR 2024 |
| `mamba` | Cited as arXiv 2023 article; published at COLM 2024 | `@article` → `@inproceedings`; `booktitle={COLM}, year={2024}` | COLM 2024 site (outstanding paper) |

## Added for lineage (S1 / W-9)

| Key | Entry | Verified metadata |
|---|---|---|
| `griffin` | De et al., *Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models* | arXiv:2402.19427, 2024 |
| `deltanet` | Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* | NeurIPS 2024 (not ICML) |

## Verified without change

| Key | Status / note |
|---|---|
| `kivi` | VERIFIED (arXiv:2302.02017) |
| `kvquant` | VERIFIED; attention-only quantization (arXiv:2401.18079) |
| `qpruningkv` | VERIFIED (ACL Anthology, EMNLP 2025 Findings, pp. 8092–8105) |
| `minikv` | VERIFIED (arXiv:2411.18077) |
| `h2o` | VERIFIED (NeurIPS 2023) |
| `snapkv` | VERIFIED (NeurIPS 2024) |
| `mamba2` | VERIFIED (ICML 2024) |
| `gateddeltanet` | VERIFIED (ICLR 2025) |
| `zamba` | VERIFIED (arXiv:2405.16712) |
| `vllm` | VERIFIED (vLLM PagedAttention paper) |
| `flashinfer` | VERIFIED (arXiv:2401.05949) |
| `replayssm` | VERIFIED (blog, caching inputs instead of state) |
| `quamba` | VERIFIED (arXiv:2410.13229); W8A8/W4A8 weights/activations, not serving-state dtype |
| `mambaquant` | VERIFIED (arXiv:2501.13484); same boundary applies |
| `vllmpr43518` | VERIFIED as WIP FP8 SSM cache checkpointing (FlashInfer SSU), not the dtype path |
| `vllmpr22196` | VERIFIED as the `mamba_ssm_cache_dtype` FP32 SSM cache PR |
| `longbench` | VERIFIED (NeurIPS 2023 Datasets and Benchmarks) |
| `ruler` | VERIFIED (arXiv:2404.06654) |
| `hqekv` | VERIFIED (ACL 2026 Findings, pp. 4138–4153) |
| `rdkv` | VERIFIED (arXiv:2605.08317; title/authors/year match arXiv API) |
| `arkv` | VERIFIED metadata; bib venue CCGRID 2026 is acceptable for the 2026 conference event |
| `qwen35` | BOUNDED: 24 GDN + 8 GQA (9B) and 2B=18 GDN+6 GQA are consistent with paper claims and the 3:1 GDN/GQA pattern; official config page not re-checked in this pass |
| `sharegpt` | VERIFIED (ShareGPT trace release) |
| `gsm8k` | VERIFIED (NeurIPS 2021 Datasets and Benchmarks) |
| `mlsys2026ae` | VERIFIED as the artifact-evaluation reference for the atomic artifact contract |

## Remaining notes / honesty boundary

- `rdkv` (arXiv:2605.08317) metadata matched in the audit; re-check the
  official page before camera-ready if a version/DOI is added.
- `arkv` bib says CCGRID 2026; the conference page confirms acceptance for
  the 2026 event. Adding the arXiv id is optional.
- `qwen35` config counts for 2B are `BOUNDED` rather than fully verified:
  they match the paper's own claim and the 3:1 GDN/GQA pattern of the 9B
  official config, but were not re-confirmed on the official model card in
  this pass.
- No `\cite` key in `main.tex` is undefined; the expanded bibliography after
  this pass is 32 entries.

## Self-review gate

- Every changed entry has a source listed above; no metadata was invented in
  this pass.
- `jamba` now contains exactly the 22 authors listed on arXiv:2403.19887.
- The `turboquant` entry now describes the actual paper; `main.tex` was
  updated so the prose no longer claims TurboQuant does 2-bit KV
  co-design (see W-1 wording change).
- `deltanet` is recorded as NeurIPS 2024, not ICML 2024.
