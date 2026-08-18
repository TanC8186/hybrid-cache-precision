# MLSys vector figure package

This directory contains the quantitative vector package for the five
main-paper figures and three supplementary figures. The system and allocator
mechanism panels are editable Next AI Draw.io sources under `../drawio/`.

## Main-paper mapping

| Paper figure | Vector source | Role |
|---|---|---|
| Figure 1 | `../drawio/fig1_hybrid_allocator.drawio` + `fig1_capacity.svg` | Hybrid byte ledger, all 52 capacity pairs, and model residuals |
| Figure 2 | `../drawio/fig2_discrete_allocator.drawio` + `fig5_block_granularity.svg` | Byte-to-page-to-block mechanism and seven-cell generalization |
| Figure 3 | `fig2_gsm8k.svg` | Dependence-aware GSM8K contrasts and fail-closed selector matrix |
| Figure 4 | `fig3_ppl_ruler.svg` | PPL intervals, observed RULER equality, and exact temporal rerun |
| Figure 5 | `fig4_serving.svg` | Serving run-stability evidence and the failed four-configuration M4 primary gate |

## Supplementary-PDF mapping

| Supplementary figure | Vector source | Role |
|---|---|---|
| Figure 1 | `fig8_gsm8k_per_seed.svg` | Descriptive GSM8K seed trajectories |
| Figure 2 | `fig7_harness.svg` | Harness boundary and stacking cost |
| Figure 3 | `fig6_sensitivity.svg` | Exploratory per-layer sensitivity |

Every quantitative figure is exported as editable-text SVG and TrueType-text
PDF. The two Draw.io panels are retained as `.drawio`, SVG, and text-preserving
PDF. No SVG contains an embedded raster `<image>` element.

## Rebuild

```powershell
python paper/mlsys2026/figures/vector_redesign/make_vector_figures.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/mlsys2026/main.tex
```

The legacy quantitative pass is followed automatically by
`make_top_venue_figures.py`, which overwrites the five main-paper quantitative
panels with the claim-led layouts while leaving the supplementary figures
unchanged.

The plotting script reads only repository experiment artifacts and writes
600-dpi TIFF/PNG previews to `tmp/figure-redesign/previews` for QA; raster
previews are not used by the paper.

## Evidence boundaries preserved in the redesign

- Capacity gains are measured and compared with the analytic capacity model.
- GSM8K inference uses 1,800 paired seed-item draws with two-way CR1 clustering
  by item and dataset seed. The 2B state-only interval includes zero; the 2B
  int4-KV interval excludes zero on the negative side.
- The five no-think RULER cells have exact observed zero deltas in both
  temporal runs; the zero-width descriptive intervals are not an equivalence test.
- Per-layer raw significance does not survive Bonferroni or BH-FDR correction.
- Serving is presented as workload-limited, same-contract run-stability
  evidence. No serving cell survives BH-FDR, and the second run is labeled a
  temporal rerun rather than an independent reproduction. In the frozen M4
  audit, 537/720 continuous-goodput comparisons are within 10% and 183/720 are
  outside 10%, so the primary gate fails. The 713/720 exact binary SLO labels
  are secondary evidence and do not override that failure.

The visual grammar was benchmarked against 39 papers from MLSys, OSDI, SOSP,
ASPLOS, ICML, ISCA, and USENIX ATC: restrained color, direct labels, aligned
small multiples, and explicit causal flow in system schematics. The audit is
recorded in `../../../../docs/notes/mlsys-figure-style-audit-2026-08-14.md`.

## QA snapshot

- Nature Figure base-script static preflight: 20 pass, 0 warnings, 0 failures.
- Top-venue wrapper static preflight: 18 pass, 2 reviewed warnings, 0 failures.
  The detected 70.6 mm width is a composite subpanel rather than a final paper
  width, and confidence-interval drawing is delegated to the shared base
  module.
- Data audit: 16 frozen source hashes and all 52 supplementary capacity rows
  pass fail-closed verification.
- PDF text audit: every figure passes a 5 pt glyph floor; minimum observed is
  5.1 pt.
- Assembled-paper physical-size audit: all 308 Arial figure-text spans remain
  at least 5 pt after LaTeX scaling; the minimum is 5.016 pt.
- SVG audit: all 10 SVGs (eight quantitative and two Draw.io) contain editable
  text and no embedded raster images.
- Submission audit: the main paper is an 8-page letter-size PDF and the
  separate supplement is 4 pages. All fonts are embedded. The final logs have
  no overfull boxes, undefined citations or references, duplicate
  destinations, empty anchors, or author-template errors.
