# MLSys 2026 Figure Contract

## Claim boundary

The figures support a Qwen3.5/vLLM/RTX 5090 allocator characterization. They
do not claim demonstrated scheduler concurrency, stable end-to-end benefit,
lossless quality, or independent reproduction.

## Figure-level contract

Core conclusion: reducing recurrent temporal-state precision changes the fixed
per-sequence cache term and consistently increases measured allocator token
capacity, but quality and serving evidence require a fail-closed deployment
interpretation.

Target: MLSys 2026, two-column PDF, editable SVG/PDF sources, all rendered
glyphs at least 5 pt. Python is the quantitative plotting backend. Next AI
Draw.io is used for the system and mechanism schematics requested by the
author.

Visual semantics are fixed across all main figures: orange = attention KV,
blue = recurrent state or state-bf16, dark neutral = full precision/measured
reference, teal = exact observed equality, and red = adverse or failed
evidence. White backgrounds, direct labels, thin connectors, and one visual
hero per figure are required. Gradients, shadows, 3D, decorative card grids,
and arbitrary panel palettes are prohibited.

## Main-paper panel map

- Figure 1, schematic-led composite: (a) hybrid layer topology, two cache terms,
  shared GPU byte ledger, discrete allocator, and scope boundary; (b) measured
  capacity evidence across the frozen matrix plus continuous-model residuals.
- Figure 2, mechanism-led composite: (a) byte-to-page-to-block pipeline for the
  2B/int4/4K worked example; (b) discrete packing terms across the seven
  fixed-utilization cells. Values 657.4 and 904.3 are allocator-equivalent
  sequence-slot counts, never concurrent requests.
- Figure 3, guardrail composite: (a) dependence-aware GSM8K intervals; (b)
  complete fail-closed selector outcome. Only the strict request is feasible,
  and it selects full precision.
- Figure 4, evidence-boundary composite: (a) chunk-level PPL intervals; (b)
  formal and temporal-rerun RULER observations. Exact observed equality is not
  an equivalence test.
- Figure 5, serving-boundary composite: (a) same-contract formal and temporal
  goodput contrasts; (b) workload boundary changes; (c) failed stability/FDR
  gates. The rerun is not an independent replication.

## Source-file mapping

- `drawio/fig1_hybrid_allocator.drawio` provides Figure 1a.
- `drawio/fig2_discrete_allocator.drawio` provides Figure 2a.
- `vector_redesign/make_vector_figures.py` reads frozen artifacts for all
  quantitative panels and final composites.
- `fig6_sensitivity`: exploratory per-layer effects after multiplicity correction.
- `fig7_harness`: chunk-size sensitivity and state/KV stacking cost.
- `fig8_gsm8k_per_seed`: descriptive dataset-seed summaries, not iid replicates.

## Statistical semantics

- GSM8K primary estimand: mean paired accuracy difference over 1,800 observed
  seed-item draws.
- GSM8K uncertainty: intercept-only OLS with two-way CR1 clustering by 1,017
  unique items and nine dataset seeds; Student-t reference with 8 df.
- Square GSM8K markers indicate a 95% interval that excludes zero; color is
  redundant and is not the sole significance encoding.
- PPL and RULER limitations remain explicit in captions and manuscript text.
- Serving effects are in requests per second. The second run is a temporal
  rerun under the same contract, not an independent replication.

## Capacity semantics

- The capacity source is
  `results/verified/2026-08-14/capacity-2x2-analysis-corrected.json`.
- Recurrent state contains a bf16 convolution state plus a temporal state.
  Changing the temporal state from fp32 to bf16 reduces per-GDN-layer state
  bytes from 1,085,440 to 561,152 (48.30%), not by exactly 50%.
- Int4 attention storage is 528 bytes per token per attention layer: two times
  256 packed payload bytes plus 8 metadata bytes.
- Signed model residuals describe approximation error. They are neither lower
  nor upper bounds.
- Any sequence count derived from token capacity is labeled an
  allocator-equivalent sequence-slot count.

## Frozen sources

- `results/verified/2026-08-14/capacity-phase-formal-corrected.analysis.json`
- `results/verified/2026-08-14/capacity-2x2-analysis-corrected.json`
- `results/quality/gsm8k-state9seed-v2-dependence-aware-20260814.json`
- `results/quality/gsm8k-9b-state9seed-v2-dependence-aware-20260814.json`
- `results/quality/ppl-stacking-analysis-20260809.json`
- `results/reproduction/2026-08-13/ruler-nothink/ruler-nothink-5cell-gate4-20260813/gate4_validation.json`
- `results/verified/2026-08-09/statebf16-serving-{formal,repro}-analysis.json`
- `results/quality/serving-direction/serving-direction-agreement-20260811.json`
- `results/reproduction/2026-08-13/m4-four-config/gate4-r3/m4_gate4_validation.json`
- `results/quality/state-sensitivity-analysis-20260809-bonf.json`
- `results/quality/chunk-ablation/*.csv`

`verify_figure_data.py` pins the expected SHA-256 digest and headline values
for every source used by the vector plotting script and checks the complete
52-row supplementary capacity table against the formal analysis.
