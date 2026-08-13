# MLSys vector figure package

This directory is the vector-first replacement for the eight figures in
`paper/mlsys2026/main.tex`.

## Paper mapping

| Paper figure | Vector source | Role |
|---|---|---|
| Figure 1 | `fig5_block_granularity.svg` | Block granularity and allocated blocks |
| Figure 2 | `fig1_capacity.svg` | Joint KV-state budget and capacity model |
| Figure 3 | `fig2_gsm8k.svg` | GSM8K paired quality deltas |
| Figure 4 | `fig3_ppl_ruler.svg` | PPL stacking and RULER evidence |
| Figure 5 | `fig7_harness.svg` | Harness boundary and stacking cost |
| Figure 6 | `fig8_gsm8k_per_seed.svg` | GSM8K per-seed trajectories |
| Figure 7 | `fig6_sensitivity.svg` | Per-layer precision sensitivity |
| Figure 8 | `fig4_serving.svg` | Serving run-stability evidence |

Every figure is exported as editable-text SVG and TrueType-text PDF. The SVG
files contain no embedded raster `<image>` elements.

## Rebuild

```powershell
python paper/mlsys2026/figures/vector_redesign/make_vector_figures.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/mlsys2026/main.tex
```

The plotting script reads only repository experiment artifacts and writes
600-dpi TIFF/PNG previews to `tmp/figure-redesign/previews` for QA; raster
previews are not used by the paper.

## Evidence boundaries preserved in the redesign

- Capacity gains are measured and compared with the analytic capacity model.
- GSM8K is reported in percentage points. The 2B state result is a significant
  -1.00 pp regression; the figures do not claim lossless quality.
- The five no-think RULER cells have exact observed zero deltas in both
  attempts; the zero-width descriptive intervals are not an equivalence test.
- Per-layer raw significance does not survive Bonferroni or BH-FDR correction.
- Serving is presented as workload-limited, same-contract run-stability
  evidence. No serving cell survives BH-FDR, and the second run is not labeled
  as an independent reproduction.

The visual grammar was benchmarked against official MLSys papers and project
pages for FlashInfer, QServe, Rethinking KV Cache Compression, and MorphServe:
restrained color, direct labels, aligned small multiples, and explicit causal
flow in system schematics.

## QA snapshot

- Nature Figure strict static preflight: 20 pass, 0 warnings, 0 failures.
- PDF text audit: every figure passes a 5 pt glyph floor; minimum observed is
  5.3 pt.
- SVG audit: 8/8 contain editable text and 0/8 contain raster images.
- Manuscript audit: `main.pdf` builds successfully as an 11-page letter-size
  PDF with all fonts embedded and no overfull boxes or undefined references.
