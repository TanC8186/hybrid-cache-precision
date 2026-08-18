# MLSys Figure-Style Audit (2026-08-14)

## Scope

The audit inspected 39 open papers: 27 MLSys papers and 12 papers from OSDI,
SOSP, ASPLOS, ICML, ISCA, and USENIX ATC. The reproducible index, source URLs,
PDF hashes, extracted captions, and five contact sheets are under
`paper/mlsys2026/build/style-corpus/`. Source figures were used only for visual
study; no external panel is copied into this manuscript.

## Corpus signals

- 34/39 papers use at least one system-, architecture-, workflow-, or
  design-oriented caption.
- 37/39 papers use quantitative figure captions.
- Figure 1 appears on page 1 or 2 in 29/39 papers.
- Frequent caption concepts include throughput (54 mentions), latency (46),
  comparison (43), memory (37), cache (32), and overview (26).

The most relevant visual references for this paper were FlashInfer, Marconi,
FlexInfer, GMLake, ExeGPT, Splitwise, Alpa, Prompt Cache, Atom, Llumnix, and the
PagedAttention paper. These papers repeatedly establish memory organization,
dataflow, or scheduling structure before presenting compact validation plots.

## Adopted visual grammar

1. Give one panel clear visual priority. A system or mechanism panel may occupy
   45--60% of a full-width figure; supporting plots validate it rather than
   compete with it.
2. Use alignment, repeated glyphs, and semantic color to create hierarchy.
   Avoid shadows, gradients, three-dimensional effects, decorative cards, and
   oversized arrows.
3. Reuse one semantic palette throughout the paper: orange for attention/KV,
   blue for recurrent state, dark neutral for full precision or measured
   reference, teal for observed equality, and red only for failed or adverse
   evidence.
4. Prefer direct labels and a single shared legend. Keep panel letters and
   typography consistent at final publication size.
5. Show null and failed results as first-class evidence. Missing cells stay
   visible, confidence intervals crossing zero remain visible, and failed
   selector or serving gates are not hidden.
6. Separate three evidence layers visually: allocator capacity, quality
   guardrails, and operational serving. Do not imply that one substitutes for
   another.
7. Keep quantitative geometry proportional to the frozen values. Diagrams may
   explain mechanism, but they may not invent concurrency, scheduler admission,
   throughput, latency, or quality effects.

## Redesign decisions

- Figure 1 becomes a schematic-led composite: hybrid layer topology and a
  shared byte ledger lead into the allocator; the measured 52-pair capacity
  result and model residuals are the validation evidence.
- Figure 2 becomes a byte-to-page-to-block mechanism figure, centered on the
  2B/int4/4K worked example and the integer packing terms in the allocator.
- Figure 3 joins the dependence-aware GSM8K intervals to the fail-closed
  selector outcome, making the guardrail consequence visible.
- Figure 4 distinguishes weak PPL evidence from exact observed RULER equality
  without turning either into an equivalence claim.
- Figure 5 leads with the failed serving-stability conclusion, then shows the
  run-level evidence and the 0/60 BH-FDR result.

## Non-negotiable claim boundaries

- Scope is Qwen3.5-2B/9B with vLLM on one RTX 5090.
- The work characterizes an allocator; it does not demonstrate scheduler
  concurrency or a stable end-to-end serving benefit.
- The 657.4 and 904.3 values are allocator-equivalent sequence-slot counts,
  not concurrent requests.
- The second serving run is a temporal rerun, not independent replication.
- Signed continuous-model residuals are approximation errors, not bounds.

