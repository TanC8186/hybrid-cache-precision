# Uniform 4-bit KV Cache Quantization for Hybrid Linear-Attention LLMs

> **Working draft — Abstract / Introduction / Related Work / Method** (updated 2026-08-06).
> These chapters connect to the existing Evaluation draft
> (`docs/paper/serving-evaluation-2026-08-03.md`, referenced below as **Eval**; its §1–§8
> map to the paper's Experiments/Limitations sections and should be renumbered when merged).
> All numbers are copied from the archived JSONs / server logs and cross-validated notes; items
> derived from code or carrying residual uncertainty are marked **[VERIFY]**. No number is invented;
> where a value is not yet measured we say so explicitly. Language: English (MLSys submission).
>
> **Canonical-data update (2026-08-06).** The serving mainline is **uniform 4-bit** KV quantization
> (`kv_cache_dtype=int4_per_token_head` applied to every GQA attention layer). The earlier
> "+25% SLO" headline is **withdrawn**: it was an overloaded-transient artifact of the first
> single-run matrix. The canonical SLO evidence is the protocol-v2 steady-state matrix
> (`results/verified/2026-08-04/e3/validation_report.md`, VERIFIED): uniform int4's sustainable
> boundary is workload-dependent (Random: +0% at 250 ms, +4.8% at 500 ms, +14.3% at 1000–3000 ms;
> ShareGPT: −17.6%). Genuine per-layer mixed-dtype allocation *collapses* capacity under the
> legacy uniform-page KV manager (Eval §8), which we address with **packed per-layer page groups**
> (§4): capacity recovers to 0.833× of uniform int4 (3.232× vs. legacy per-layer), and ShareGPT
> serving boundaries are at least as high as uniform int4 (Eval §8, ANALYZED).

---

## Abstract

KV caches are the dominant memory bottleneck in LLM serving, and 4-bit KV quantization has become
the primary lever for fitting more context and higher concurrency into a fixed GPU memory budget.
Existing KV-quantization and cache-compression systems are, however, designed for and validated on
purely attention-based Transformers. Hybrid architectures — which interleave linear-attention
layers (Gated DeltaNet, GDN) with a small number of standard GQA attention layers — are rapidly
proliferating (e.g., the Qwen3.5 family), and their structural properties break a central
assumption of prior work: a per-sequence recurrent state that the KV scheme cannot quantize and
that shares the same GPU memory pool as the quantizable attention KV.

We present a system study of int4 KV-cache quantization on a hybrid architecture, using
Qwen3.5-2B (18 GDN + 6 GQA layers) served with vLLM on an RTX 5090. Uniform int4 per-token-head
quantization increases end-to-end KV capacity **2.245× at 4K context** (growing to **3.155× at
16K**). Under a fixed 60 s steady-state Poisson protocol (protocol-v2, VERIFIED), the sustainable
SLO boundary is workload-dependent: uniform int4 gains **+14.3%** on synthetic random traffic at
TTFT thresholds 1000–3000 ms, no gain at 250 ms, and loses **−17.6%** on the real ShareGPT trace.
We separate the mechanism-level compression (**3.88×** on the attention KV alone) from the
system-level ratio, exposing a dilution effect unique to hybrid models: the **18.63 MiB
per-request GDN state** is non-quantizable and consumes ≈60% of the KV budget at the int4 server's
maximum concurrency (code-derived estimate). Under an equal byte budget we find that keeping high
precision and evicting tokens dominates dropping below 4 bits (3-seed Wikitext-2 PPL 11.85 ± 1.66
vs 19.00 ± 3.19 at ≈3.2 MB). To make per-layer protection deployable, we implement **packed
per-layer page groups** (§4): mixed-precision attention layers share a single packed backing
storage, recovering capacity from **0.258× to 0.833× of uniform int4** (3.232× vs. legacy
per-layer), with ShareGPT SLO boundaries at least as high as uniform int4 (Eval §8, ANALYZED).

---

## 1. Introduction

**The problem.** In LLM serving the KV cache is the resource that scales with users and context,
and it is the first to exhaust GPU memory. For a 7B-class model with 4K–16K context, KV caches
routinely exceed the weight footprint; KV quantization is therefore the standard production lever
for longer context, larger batches, and higher concurrency under a fixed memory budget
(§2.1). The standard recipe is simple: store K/V activations in 4 bits per token with per-token
scales, and accept a small quality cost in exchange for a roughly 4× compression of the cache.

**The blind spot.** Almost all of this work assumes a purely attention-based Transformer, where
*every* layer contributes a growing KV cache and where the cache is homogeneous and quantizable. A
new model family breaks both assumptions and is spreading quickly: **hybrid linear-attention
LLMs**, which interleave Gated DeltaNet (GDN) linear-attention layers with a small number of
full-attention layers. The Qwen3.5 family — the subject of this paper — is a concrete instance:
Qwen3.5-2B has **18 GDN layers and only 6 GQA attention layers** (at indices {3,7,11,15,19,23}),
and Qwen3.5-9B has 24 GDN + 8 GQA layers. Because the GDN layers keep no growing key/value cache,
the *quantizable* attention KV is a small fraction of the architecture — but the *recurrent state*
of the GDN layers is stored per sequence, is **not quantizable** by any KV scheme, and is drawn
from the **same GPU memory pool** as the attention KV.

This hybrid structure invalidates the mental model behind existing KV-quantization results in three
ways, which we exploit and, in one case, are blocked by:

1. **Only a fraction of layers carry a KV cache.** Quantization can only compress the 6 (or 8) GQA
   layers; the 18 (or 24) GDN layers contribute no KV but a fixed per-sequence state instead.
2. **The fixed per-sequence state dilutes the compression ratio end to end.** We measure a
   mechanism-level 3.88× on the attention KV, but only 2.245× of end-to-end KV capacity at 4K
   context — a dilution caused by the 18.63 MiB/request GDN state that no KV bit-width can remove.
3. **The few attention layers are highly heterogeneous.** Forcing a single attention layer (the
   final GQA layer, index 23) to 2 bits costs +28.7% of the 2-bit sensitivity range, while layer 3
   is effectively free (−5.9%), motivating per-layer allocation — which the legacy uniform-page KV
   manager cannot host without a catastrophic capacity penalty, and which we make deployable with
   packed per-layer page groups (§4; Eval §8).

**Our approach.** We study the hybrid case end to end. We quantize all GQA attention layers to
uniform int4 (`int4_per_token_head`: 4 bits per token with a per-token scale), deploy it in the
vLLM serving stack on a single RTX 5090, and measure (i) KV-cache capacity under a fixed GPU memory
budget, (ii) throughput–latency curves under Poisson load, and (iii) SLO-constrained capacity. We
report **two calibers** throughout — the mechanism-level attention-only compression (3.88×) and the
system-level end-to-end capacity ratio (2.245× @4096, 3.155× @16384, 2.19× for Qwen3.5-9B @4096) —
because reporting only one would either over- or under-sell the method on a hybrid model.

**Contributions.**

**(a) A system study of KV quantization on a hybrid linear-attention LLM, with real-load serving
evidence.** Uniform int4 KV quantization on Qwen3.5-2B increases end-to-end KV capacity
**2.245×** under a fixed GPU budget at 4K context (1,203,106 → 2,701,721 tokens), growing to
**3.155×** at 16K context; on Qwen3.5-9B it gives **2.19×** @4096. Under the protocol-v2
steady-state SLO protocol (TTFT thresholds 250–3000 ms, TPOT 200 ms, 60 s arrival window,
3 seeds, VERIFIED), the sustainable boundary is workload-dependent: **+14.3%** for synthetic
random at 1000–3000 ms thresholds, **+4.8%** at 500 ms, **0%** at 250 ms, and **−17.6%** on the
ShareGPT trace. The 2.245× larger cache converts into higher saturation goodput (≈+5%, 3-seed).

**(b) A structural insight unique to hybrid models: recurrent-state dilution.** We identify and
quantify that the GDN per-sequence state (18 layers × 1,085,440 B = **18.63 MiB/request**,
derived from `MambaStateShapeCalculator.gated_delta_net_state_shape` in the vLLM fork) is not
quantizable, grows linearly with concurrency, and is drawn from the same KV memory pool. It
dilutes the attention-only 3.88× compression to a 2.245× end-to-end ratio and caps the total
capacity gain of "quantize-the-KV-only" approaches in hybrid models. The dilution weakens with
context length (2.245× @4096 → 3.155× @16384), giving hybrid KV quantization a *long-context
advantage* that pure-attention models do not exhibit.

**(c) An equal-byte-budget ordering in the sub-4-bit region.** On this hybrid model, 4-bit KV is
near-lossless (paired 3-seed Wikitext-2 PPL **+1.7%** with 95% CI [0.8%, 2.6%] vs. FP16) and 8-bit
is lossless; the tension zone is below 4 bits (3-bit +17.8%, 2-bit +65.5%). Under a fixed byte
budget, *keeping 4-bit and evicting tokens dominates dropping below 4 bits*: at ≈3.2 MB, 4-bit +
eviction scores PPL 11.85 vs 19.00 for 2-bit full retention (paired Δ −7.15, 95% CI [−10.97,
−3.34]). This ordering, established by QPruningKV on standard Transformers, is verified here on a
hybrid architecture.

**(d) A deployable system contribution: packed per-layer page groups.** Sensitivity-guided
per-layer allocation is attractive at the quality level (it beats uniform 3-bit at equal bytes;
Eval §6), but in the legacy vLLM V1 KV manager, mixing dtypes forces uniform-page-size unification
and collapses capacity to **×0.258 of uniform int4** — below even the fp16 baseline. We implement
**packed per-layer page groups** (§4): five int4 GQA layers and one bf16-protected layer share a
single packed backing storage, recovering capacity to **0.833× of uniform int4** (2,280,448 vs
2,736,947 tokens at 4K; 3.232× vs. legacy per-layer), with the runtime layout and capacity ratios
reproduced on an independent host (VERIFIED). In ShareGPT serving, packed sustains a boundary at
least as high as uniform int4 (40 vs 35 req/s at the 250 ms TTFT threshold; ANALYZED, Eval §8).

**(e) Honest disclosure of a serving-system limitation.** We report the quantified legacy
regression (×0.258) that motivated (d), and we separately report the workload reversal in (a):
uniform int4 does not universally improve SLO capacity, and Random/ShareGPT results must not be
pooled.

**Organization.** §2 reviews related work. §3 describes the model, the int4 scheme, the capacity
model that explains the 3.88×→2.245× dilution, and why the serving mainline is uniform. §4
presents the packed per-layer page-group mechanism. The experimental setup and results (E1
capacity, E2 throughput–latency, E3 protocol-v2 steady-state SLO, per-token cost, byte-budget
quality) and the quantified legacy limitation are in the companion Evaluation draft
(`docs/paper/serving-evaluation-2026-08-03.md`).

---

## 2. Related Work

### 2.1 KV-cache quantization

Quantized KV caches reduce the bytes stored per token per layer. Practical schemes span int8 and
fp8 KV caches used in production serving systems, and sub-4-bit methods. KIVI [kivi] introduced
grouped 2-bit quantization with a small FP16 residual; KVQuant [kvquant] applied per-channel /
per-token scales with non-uniform quantization and showed 3-bit KV within ≈1 PPL point. TurboQuant
[turboquant] demonstrated int3/int4 KV backends on recent models and explicitly identified the
2-bit *value* cache as the quality bottleneck. For the Qwen3.5 family specifically, RotorQuant and
llama.cpp's q4_0 path report near-lossless 4-bit KV [rotorquant, llamacpp] (community sources, not
peer-reviewed) — consistent with our own measurement that 4-bit is within +1.7% PPL on
Qwen3.5-2B (Eval §6). In the serving stack, vLLM [vllm] ships `int4_per_token_head` /
`int8_per_token_head` and TurboQuant backends natively; we build on the vLLM mechanism directly
rather than introducing a new quantizer. All of these methods quantize *attention* KV and assume
every layer participates; none addresses the hybrid case where most layers carry no quantizable KV
and instead hold a per-sequence recurrent state.

### 2.2 Eviction, compression, and joint byte-budget allocation

A complementary line reduces the *number* of cached tokens. H2O [h2o] retains heavy-hitter tokens
by attention score; StreamingLLM [streamingllm] and SnapKV [snapkv] keep attention sinks and
recent windows / clustered tokens. More recent work fuses quantization and eviction into a single
byte-budget allocation: QPruningKV [qpruningkv] argues via
a budget-equivalence protocol (1×16 vs 2×8 vs 4×4) that "keep more tokens at lower precision"
dominates both pure eviction and pure quantization; RDKV [rdkv] treats eviction as
0-bit quantization under rate-distortion water-filling; ARKV [arkv] allocates bytes
across FP16 / low-bit / evicted states on Qwen3/LLaMA3; MiniKV [minikv] and HqeKV [hqekv]
push joint schemes further (additional joint-scheme works reviewed in
`docs/notes/lit-due-diligence-2026-08-02.md` are omitted here until their metadata is
verifiable). The literature due-diligence we performed
(`docs/notes/lit-due-diligence-2026-08-02.md`) found that **every one of these joint schemes is
validated only on standard Transformers** (LLaMA3, Qwen3, Mistral). None exploits the hybrid
architecture's structural properties: a minority of quantizable attention layers, linear layers that
absorb quantization noise, and per-layer heterogeneity.

### 2.3 Linear-attention and hybrid serving

Linear-attention models replace (a subset of) full attention with recurrent state-space layers:
Mamba and Mamba-2 [mamba, mamba2], Gated DeltaNet [gateddeltanet], and the DeltaNet
ecosystem; hybrids such as Jamba [jamba], Zamba [zamba], and RecurrentGemma [recurrentgemma]
interleave such layers with full attention. These models advertise very long contexts
(Qwen3.5-2B advertises 262K tokens, per its model card [qwen35]), which makes their KV behavior
*more* not less consequential — yet hybrid serving
work has focused on fusing the recurrent kernel into the scheduler (e.g., vLLM's
`LinearAttentionBackend` and per-sequence state management), not on compressing the attention KV
that remains. The only hybrid-specific KV-quantization observations we found are community
measurements that q4 KV is effectively lossless on Qwen3.5 (consistent with our 4-bit result) and
TurboQuant's note [turboquant] that decode must dequantize the full history — a bandwidth cost our serving
evaluation measures directly as a +8–10% TPOT p50 overhead (Eval §5).

### 2.4 Positioning

To our knowledge this is the first report of an end-to-end **capacity × real-load SLO** study of
KV quantization on a hybrid linear-attention LLM in the vLLM serving stack, and the first to
quantify the **recurrent-state dilution** that separates the mechanism-level compression ratio
(3.88×) from the end-to-end system ratio (2.245×). Community measurements of 4-bit KV quality on
Qwen3.5 (e.g., llama.cpp q4_0 and RotorQuant) and TurboQuant's sub-4-bit vLLM backends are
complementary; none reports the hybrid-specific capacity/SLO system behavior studied here. Our
equal-byte-budget ordering (evict before dropping below 4 bits) matches the direction of
QPruningKV/RDKV/ARKV but is established here on a hybrid architecture where the quantizable cache
is a small minority of layers — and where the fixed per-sequence state changes the byte accounting.
We also contribute a serving-system negative result that constrains heterogeneous-KV design in
hybrid models: mixed-dtype per-layer allocation collapses capacity under uniform-page KV managers
(Eval §8), and a positive system mechanism (packed per-layer page groups, §4) that removes the
collapse.

---

## 3. Method

### 3.1 Model and KV-cache composition

**Hybrid architecture.** We study the Qwen3.5 family of hybrid linear-attention LLMs
(`qwen3_5` model type). Qwen3.5-2B has 24 layers: **18 Gated DeltaNet (GDN) linear-attention
layers** and **6 standard GQA full-attention layers** at indices {3,7,11,15,19,23} (attention
layers every 4 layers). Each GQA layer uses 2 KV heads with head_dim 256. Qwen3.5-9B is the same
family at larger scale: 32 layers = 24 GDN + 8 GQA attention layers at indices
{3,7,11,15,19,23,27,31}, hidden size 4096, 4 KV heads, GDN linear KV heads 16 with dim 128, and
`mamba_ssm_dtype=float32`. In both models the attention-layer indices follow a fixed interval-4
`layer_types` pattern; we use this pattern in the capacity model below.

**What the KV cache actually contains.** Two disjoint objects live in the KV memory pool:

- **Attention KV (quantizable).** Per GQA layer, per token: 2 KV heads × 256 dim × 2 B × 2 (K and
  V) = **2048 B** in fp16. Across the 6 layers this is **12,288 B/token** (2B model). Under
  int4-per-token-per-head (including the per-token scale) it drops to ≈**528 B** per layer per
  token (≈3,168 B/token across the 6 layers), for a mechanism-level ratio of 2048/528 = **3.878×**.
- **GDN recurrent state (not quantizable).** Each GDN layer keeps a per-sequence recurrent state.
  From `MambaStateShapeCalculator.gated_delta_net_state_shape`
  (`vendor/vllm/.../mamba/mamba_utils.py`): the temporal state is
  (num_v_heads=16, head_v_dim=128, head_k_dim=128) = 262,144 elements stored in fp32 =
  1,048,576 B; the conv state is (conv_dim=6144, conv_kernel−1=3) = 18,432 elements in bf16 =
  36,864 B. Per layer this is **1,085,440 B**; × 18 GDN layers = 19,537,920 B ≈ **18.63 MiB per
  sequence**. This state is drawn from the same KV-cache pool as the attention blocks, grows
  linearly with the number of concurrent sequences, and is **independent of context length**. No
  KV bit-width touches it.

### 3.2 Uniform int4 per-token-head quantization

We quantize **every GQA attention layer** with vLLM's `int4_per_token_head` dtype
(`kv_cache_dtype=int4_per_token_head`): each token–head block is stored as 4-bit values with a
**per-token scale** (per-token-head layout), values packed 2-per-byte. The write path quantizes
inside the cache-store kernel (vLLM's `CopyWithScaleOp` in `cache_kernels.cu`), and the read path
uses **lazy dequantization**: the attention kernel gathers and dequantizes quantized blocks back to
fp16 on the fly (vLLM's `gather_and_maybe_dequant_cache` semantics; for int4 the serving kernel is
the Triton attention backend with fused dequant). Quantization is symmetric with no calibration
data and no per-channel offline statistics — the per-token scale makes it data-independent, which
is what keeps the serving path deterministic. The mainline is **uniform**: the same dtype on all 6
GQA layers. (Why we do *not* serve the sensitivity-guided per-layer variant is the subject of §3.4.)

The honest per-step cost of lazy dequantization is a per-token latency increase. Offline 3-seed
measurements (warmup protocol, 4096 context, eager mode) give **TPOT p50 +8.0%** (33.8 vs 31.3 ms)
and throughput −5.5% for uniform int4; across warmup protocols the cost spans **−6~8% throughput
and +8~10% TPOT p50** (Eval §5), and it appears as a +7.7% TTFT p99 / +8.0% TPOT p50 gap at low
load in the serving matrix (Eval §3). We also disclose that int4 forces the Triton attention
backend, whose 2D-decode kernel specialization is compiled by Triton JIT at serving time in cold
runs — this produces a large TPOT p99 tail (up to 894 ms) that is **not an inherent int4 cost**:
with `--warmup-n 120` the int4 p99 returns to ≈144.7 ms vs fp16 145.5 ms (Eval §5). The CUDA-graph
serving matrix itself keeps TPOT p99 ≤ 49.3 ms throughout (Eval §3–§4).

### 3.3 Capacity model: from 3.88× to 2.245×

Let $A_f$ and $A_q$ be the attention-KV bytes per token across the 6 GQA layers in fp16 and int4
($A_f = 12{,}288$ B, $A_q \approx 3{,}168$ B for the 2B model), and let $G$ be the per-sequence GDN
state ($G = 19{,}537{,}920$ B). For a context length $L$ and a concurrency $C$, the fixed KV pool
holds $C$ sequences whose total footprint is $C\,(A\,L + G)$. Equating this to the pool budget in
fp16 and in int4 gives the **end-to-end capacity ratio**:

$$r_s(L) = \frac{A_f\,L + G}{A_q\,L + G} , \qquad
  r_s(\infty) = \frac{A_f}{A_q} = 3.878 .$$

Two structural consequences follow directly:

1. **The mechanism-level 3.88× is diluted by the fixed per-sequence state.** The dilution is
   $r_s / r_m < 1$ whenever $G > 0$, and it is *strongest at short context*: at $L = 4096$ the model
   predicts $r_s = 2.149$; the measured server value is **2.2456×** (2,701,721 vs 1,203,106 tokens,
   Eval §2). At $L = 16384$ the model predicts 3.091; the measured value is **3.155×** (4,910,731 vs
   1,556,961). The residual gap is page alignment: vLLM rounds the attention `block_size` up so that
   a page covers at least the GDN page (1,085,440 B), giving fp16 blocks of 544 tokens and int4
   blocks of 2064 tokens — at 4096 context fp16 therefore rounds 7.53→8 blocks (6% slack) while
   int4 rounds 1.98→2 (0.8% slack), which favors int4 and explains the ~4% excess over the model.
2. **The dilution weakens as context grows.** $r_s(L)$ increases monotonically toward 3.878,
   because $G$ is amortized over more tokens and, at the server level, over lower concurrency (Eval
   §2: at 16K the max concurrency drops to 299.7, so the fixed GDN state consumes a smaller fraction
   of the budget). Hybrid KV quantization therefore becomes *more* attractive exactly in the
   long-context regime these models target.

The same $G$ also sets an upper bound on what any attention-only KV quantizer can buy on a hybrid:
at the int4 server's maximum concurrency (659.6 sequences) the aggregate GDN state is ≈12.0 GiB ≈
**60% of the 20.08 GiB KV budget** **[VERIFY — provenance estimate from max concurrency ×
code-derived state size, see Eval §7]**; at the 400 sequences observed in flight at the highest
offered rate it is ≈36%. Because $G$ is identical for both allocations, it is a pure diluent:
further capacity gains in a hybrid require compressing the GDN state itself, not just the attention
KV. We report both calibers everywhere so the claim is never overstated.

**Across scales and family.** The same dilution explains the Qwen3.5-9B result: uniform int4 gives
**2.19×** @4096 (328,499 vs 150,062 tokens, server logs). The 9B family member has more attention
layers (8 vs 6) but also more GDN layers (24 vs 18), so the per-sequence GDN state is larger in
absolute terms; the system-level ratio lands close to the 2B value.

### 3.4 Layer sensitivity, and why the serving mainline is uniform

**Per-layer heterogeneity (quality side).** The 6 GQA layers are far from interchangeable. With the
per-layer knob exposed on the transformers path, we force each layer to 2-bit (others 8-bit) and
measure Wikitext-2 PPL (Eval §6; deterministic single-protocol sweep with anchors 8-bit 13.63 and
2-bit 21.07; 3-seed replication pending): layer 23 is the most sensitive, costing +28.7% of the
2-bit sensitivity range (PPL 15.76 vs 13.63), while layer 3 is essentially free (−5.9%, PPL
13.20). Sensitivity-guided allocations then dominate uniform ones at equal bytes: `sens_guided`
{3:2-bit, middle 3-bit, 23:4-bit} → **PPL 14.63 @ 4.87 MB** vs uniform 3-bit 15.87 @ 4.85 MB, and
`only_layer3_2bit` {3:2-bit, others 4-bit} → **PPL 13.39 @ 5.92 MB** — better quality *and* fewer
bytes than uniform 4-bit (13.86 @ 6.44 MB). This is the evidence that per-layer protection is
valuable in principle.

**Why the serving mainline is uniform.** In the current vLLM V1 KV manager, mixing dtypes is not
deployable without a catastrophic capacity penalty. The KV pool requires a uniform page size; a
mixed `{layer 23: bf16, others: int4}` config triggers `unify_kv_cache_spec_page_size`, which
unifies every layer to the **largest** page — a bf16 layer needs 4× the bytes of an int4 layer, so
the int4 `block_size` inflates from 2064 to 8256 tokens and the number of KV groups explodes. The
measured result is a capacity of **×0.258 of uniform int4** — 696,456 vs 2,701,721 tokens (2B) and
84,787 vs 328,499 (9B) — *below* the fp16 baseline (1,203,106 / 150,062). The identical 0.258 ratio
across 2B and 9B confirms a deterministic mechanism, not model-specific noise (Eval §8). True
per-layer also costs throughput (−8.2%) and TPOT p50 (+13.8%) in 3-seed offline benches, though it
improves TPOT p99 because the bf16-protected layer bypasses the int4 Triton JIT (Eval §8).

We therefore report **uniform int4 as the serving mainline**, and treat sensitivity-guided per-layer
allocation as (i) a valid quality-side result on the transformers path and (ii) a design target that
requires **independent per-dtype page groups**. We implement this as **packed per-layer page
groups** (§4): a packed-slab layout that recovers 0.833× of uniform-int4 capacity while keeping
layer 23 protected, and that we validate end to end in the serving stack. The design rationale is
documented in `docs/notes/per-layer-page-group-design-2026-08-03.md`.

---

## 4. Packed Per-Layer Page Groups (A2)

### 4.1 Problem restated

In the legacy vLLM V1 KV manager, one global block pool is shared by all KV groups, and the
configuration layer requires a single uniform page size (`get_uniform_page_size`). Mixing dtypes
— protecting the most sensitive GQA layer (index 23) in bf16 while quantizing the other five GQA
layers to int4 — triggers `unify_kv_cache_spec_page_size`, which inflates the int4 `block_size`
from 16 to 64 tokens (page 2,064 → 8,256 B) and then splits every attention layer into its own
group (`group_size=1`). The result is a deterministic capacity collapse to **×0.258 of uniform
int4** on both Qwen3.5-2B (696,456 vs 2,701,721 tokens) and Qwen3.5-9B (84,787 vs 328,499),
below even the fp16 baseline (Eval §8). The collapse is an artifact of the uniform-page
configuration layer, not of mixed-precision KV itself: vLLM V1 already ships a packed layout
(`_get_packed_kv_cache_layout`, `KVCacheTensor(offset, block_stride)`) that lays out multiple
groups with different page sizes into one backing storage (used upstream for DeepSeek V4).

### 4.2 Mechanism

We add a flag, `--enable-per-layer-page-groups`, that changes two things on the configuration
path only (the scheduler, block pool, and coordinator are untouched):

1. **Grouping.** All full-attention layers are extracted before the uniform-page unification step
   and merged into a single `UniformTypeKVCacheSpecs` group with mixed precision: five int4 GQA
   layers plus the bf16-protected layer 23 share one block table and one `block_size` (16 tokens),
   while each layer keeps its own physical page size (int4 2,064 B, bf16 8,256 B). The 18 GDN
   Mamba groups are unchanged.
2. **Packed backing storage.** The mixed group and the Mamba groups are laid out through the
   existing packed path (`_get_packed_kv_cache_layout`): one backing tensor per worker with a
   `block_stride` equal to the maximum per-block byte footprint across groups, and per-layer
   `(offset, block_stride)` views. The Mamba reshape on the worker is fixed to honor
   `(offset, block_stride)` instead of assuming offset-0 contiguous pages, and
   `_max_memory_usage_bytes_from_groups` gains a packed branch so startup memory accounting
   matches the runtime layout.

The resulting configuration for Qwen3.5-2B is: one mixed-precision attention group
(`UniformTypeKVCacheSpecs`, 6 layers) + two 9-layer Mamba groups. The runtime MVEx confirms a
single backing storage per worker, mixed `INT4_PER_TOKEN_HEAD`/`NONE` attention quantization
modes, GDN temporal state in fp32, and a correct non-empty generation (8/8 checks).

**Relationship to existing vLLM infrastructure.** The packed `offset/block_stride` tensor layout
is the mechanism already used for DeepSeek V4; our contribution is (i) extending its trigger from
the DSV4/cross-layer paths to a mixed-precision full-attention group coexisting with Mamba
groups, (ii) the grouping branch that avoids page-size unification for GQA layers, (iii) the Mamba
reshape fix, and (iv) memory-accounting alignment. The flag is opt-in; the default path is
unchanged.

### 4.3 Verification

**Runtime and capacity (VERIFIED).** Three independent probes at `gpu_memory_utilization=0.85`,
`max_model_len=4096`, Qwen3.5-2B:

| Configuration | Capacity (tokens) | Max concurrency | Ratio vs. uniform |
|---|---:|---:|---:|
| Legacy per-layer (L23 bf16, others int4) | 705,604 | 172.3 | ×0.258 |
| Uniform int4 | 2,736,947 | 668.2 | 1.000 |
| Packed per-layer (A2) | 2,280,448 | 556.8 | **0.833** |

**Table 3 (mainline).** A2 capacity gate: packed/legacy = **3.232×** (threshold ≥ 3×),
packed/uniform = **0.833** (preset [0.80, 0.92]); gate `PASSED`
(`results/verified/2026-08-04/a2/a2_capacity_gate_c7379f0_v2.json`). On an independently
deployed host (westd-03, frozen root commit + vLLM commit, SHA-verified overlay), the three
probes reproduce within ≤0.1353% absolute capacity and ≤0.0150% ratio difference; the
runtime/capacity sub-scope is VERIFIED (REPRODUCIBLE).

**Serving (ANALYZED, independent reproduction pending).** We compared fp16, uniform int4, and
packed per-layer under the protocol-v3 steady-state matrix (3 allocations × Random60/ShareGPT300
workloads × 3 seeds; 300 s ShareGPT window, `ignore_eos` pairing; 108/108 samples
`completed_validated`, zero failed requests; slice audits PASSED). The maximum sustainable
offered rate (all 3 seeds ≥ 0.95 goodput/offered):

| Workload | TTFT threshold | fp16 | int4 | packed per-layer |
|---|---:|---:|---:|---:|
| ShareGPT300 | 250 ms | 45 | 35 | **40** |
| ShareGPT300 | 500–3000 ms | 45 | 40 | **40** |
| Random60 | 250 ms | 30 | none* | 30 |
| Random60 | 500/1000 ms | 35 | 35 | 35 |
| Random60 | 2000–3000 ms | 35 | 40 | **40** |

**Table 4 (mainline).** *None = no 3-seed fully sustainable point in the tested grid. These
boundaries are ANALYZED: the formal samples and slice audits completed, but the independent
reproduction gate has not yet run, so they must not be treated as VERIFIED paper headline
numbers.

**Reading.** On ShareGPT, packed per-layer **always meets or exceeds uniform int4** (250 ms: 40 vs
35 req/s) — the quality benefit of protecting layer 23 translates into an SLO boundary advantage —
while restoring capacity to 0.833× of uniform int4 (3.232× vs. legacy). On Random60 the packed
boundary is at least as high as uniform int4 at every threshold where uniform int4 has a point.
FP16 remains the highest-boundary allocation on ShareGPT (45), consistent with the honest
workload-specific E3 result in §3/Eval §4: quantization trades SLO boundary on real traffic for
capacity.

### 4.4 Scope and status

The A2 mechanism is validated on Qwen3.5-2B (runtime/capacity) and Qwen3.5-9B (legacy-collapse
ratio only; packed 9B capacity not yet probed), on a single RTX 5090, in a vLLM fork. The flag is
opt-in and not yet upstreamed. Quality closure (packed vs. uniform PPL and retrieval/long-context)
and the independent serving reproduction are required before the serving boundaries may be used
as headline claims (currently ANALYZED).

---

> **Next sections (see Evaluation draft `docs/paper/serving-evaluation-2026-08-03.md`).**
> Experiment setup and platform (Eval §1); E1 KV-cache capacity incl. the 16K / 9B probes (Eval §2);
> E2 throughput–latency matrix (Eval §3); E3 SLO-constrained capacity (Eval §4); TPOT per-token cost
> and warmup protocol (Eval §5); quality under a byte budget (Eval §6); honesty statements and
> provenance (Eval §7); legacy mixed-dtype capacity collapse + A2 packed per-layer page groups,
> capacity gate, and serving boundaries (Eval §8).
