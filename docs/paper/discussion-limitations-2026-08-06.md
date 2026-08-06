# Discussion, Limitations, and Data Availability (Draft, 2026-08-06)

> Companion to `paper-mainline-2026-08-03.md` and
> `serving-evaluation-2026-08-03.md` (both updated 2026-08-06). Language: English (MLSys).

## 9. Discussion

**Workload-dependent SLO behavior.** The protocol-v2 steady-state matrix shows that uniform int4
KV quantization buys capacity (2.245× at 4K, 3.155× at 16K) but does not universally improve SLO
boundaries: on synthetic Random traffic the sustainable boundary gains 0–14.3% depending on the
TTFT threshold, while on the ShareGPT trace it drops by 17.6%. This reversal has a plausible
mechanism — the per-step dequantization overhead (TPOT p50 +8–10%) matters more under real
variable-length traffic than the capacity headroom does — but the causal decomposition is not
established by our measurements. We therefore report both workloads separately and explicitly do
not pool them.

**Why uniform int4 is the serving mainline.** Sensitivity-guided per-layer allocation is better at
equal bytes on the quality side, but the legacy vLLM V1 manager penalizes mixed dtypes
catastrophically (×0.258). Our packed per-layer page groups remove that penalty: capacity recovers
to 0.833× of uniform int4, and on ShareGPT the packed configuration sustains a boundary at least
as high as uniform int4 (40 vs 35 req/s at 250 ms). This makes the quality-motivated protection
deployable, but two gates remain before the serving boundaries are headline-grade: quality closure
(packed vs. uniform PPL/retrieval) and an independent reproduction of the 108-sample matrix
(currently ANALYZED).

**Long context.** The dilution model predicts — and the 16K capacity probes confirm — that the
system-level compression ratio improves with context length (2.245× → 3.155× at 2B; 2.19× →
3.167× at 9B). We have not yet run serving SLO or quality evaluations beyond 16K; the
"long-context advantage" is therefore a capacity-level claim, not an end-to-end serving claim.

**What we did not do.** We do not introduce a new quantizer or attention kernel; we study a stock
vLLM dtype and a configuration/layout mechanism. We do not compress the GDN recurrent state, which
is the largest single per-sequence item (≈60% of the KV budget at int4 peak concurrency); its
compressibility is an open question that bounds the headroom of attention-only KV quantization.
We provide executable TurboQuant baselines (`turboquant_k8v4` and `turboquant_4bit_nc`) in this
fork: engine startup, greedy generation, and an 18-cell NIAH quality matrix completed with zero
failures (mean accuracy 0.8519 / 0.8889 vs. fp16 0.9074; paired 95% CIs include zero; ANALYZED).
The same-protocol serving SLO matrix for TurboQuant is pending; KIVI/KVQuant remain
transformers-path-only. Until the serving matrix is complete, the capacity ratios should be read
as "vs. fp16/bf16 serving on this stack", not as "vs. state-of-the-art KV quantizers" in serving.

## 10. Limitations

1. **Scope of hardware and software.** All serving results are from one RTX 5090 (sm_120), one
   vLLM fork commit, and one model family (Qwen3.5-2B for the full matrix; Qwen3.5-9B for capacity
   and a single-run E2 check). No multi-GPU, other hybrid family, or upstream vLLM validation has
   been run. The A2 flag is opt-in and not upstreamed.
2. **Statistical uncertainty.** E3 boundaries use n = 3 seeds with an all-seeds-satisfy criterion
   and a 5 req/s rate grid; ShareGPT intervals are wide (CAUTION). PPL claims use paired t-CIs
   over n = 3 seeds; 4-bit's +1.7% is nonzero but small, and the 3-bit/2-bit CIs are wide.
   Sensitivity and heterogeneous-budget PPL tables are single deterministic protocol runs; their
   3-seed replication is pending.
3. **Evidence status asymmetry.** E1 capacity and E3 protocol-v2 boundaries are VERIFIED
   (independent reproduction). A2 runtime/capacity is VERIFIED; A2 serving boundaries are
   ANALYZED (samples and audits complete, independent reproduction pending). The historical
   single-run E2 matrix is background only. Any table in the paper carries its status explicitly.
4. **Workload coverage.** Random (1024/128) and ShareGPT cover one synthetic and one real trace;
   multi-turn, retrieval-heavy, >16K, and multi-tenant workloads are not covered. Prefix-cache
   hit rates and block-alignment effects of the packed layout are not yet measured.
5. **Quality evaluation.** Wikitext-2 PPL is the only quality metric; no retrieval/long-context
   benchmarks (LongBench/RULER) are reported yet. The equal-byte comparisons are not byte-exact
   (≤5.4% tolerance at ≈3.2 MB); the ordering is robust within the tolerance in every seed.
6. **GDN state as a fixed cost.** We treat the 18.63 MiB/request recurrent state as non-quantizable
   and fixed. We have not explored fp32→bf16/int8 state compression or state sharing; if such
   compression is viable, the dilution model's upper-bound framing changes.

## 11. Data Availability

We commit to releasing the following with the camera-ready/artifact submission (MLSys 2026
Artifact Evaluation):

- **Code**: the vLLM fork at frozen commits (root `c7379f0`/`3108650` for A2, vLLM `55f47685`),
  the per-layer dtype patch and the A2 page-group patch as diffs
  (`vendor/vllm-patches/`), and the experiment pipeline (`Makefile`, `scripts/`).
- **Reproduction**: `make reproduce` entry point; frozen configs under `configs/` and
  `experiments/configs/`; `--resume` slices with immutable attempt IDs.
- **Evidence**: per-attempt SHA-256 manifests, contract/result/analysis sidecars, server startup
  logs, and raw benchmark JSONs (large raw dumps are served from the archive; small verified
  copies are in `results/verified/`).
- **Failure semantics**: failed/quarantined attempts are retained with distinct statuses
  (`QUARANTINED`, `FAILED_RUNTIME_COLLECTION`, `PILOT_FAILED_*`) and are never pooled into
  efficacy denominators.
- **Data**: Wikitext-2 (public), ShareGPT_Vicuna_unfiltered via hf-mirror, model checkpoints via
  ModelScope (`Qwen/Qwen3.5-2B`, `Qwen3.5-9B`), pinned by config hash.
- **Hardware**: single RTX 5090 32 GB (sm_120); server environment pinned in
  `configs/env/remote_5090.yaml`.

Status summary for reviewers: capacity E1 and E3 protocol-v2 = VERIFIED; A2 capacity =
VERIFIED; A2 serving = ANALYZED; quality PPL = canonical 3-seed with paired CIs (sensitivity
tables single-run); external baselines = not yet available.
