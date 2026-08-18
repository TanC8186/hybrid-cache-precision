# Manuscripts

Both directories are venue-specific versions of the same research artifact.
They share the title, evidence, analysis code, and core figures.

| Directory | Role | Build |
|---|---|---|
| `mlsys2026/` | Anonymous MLSys-format research manuscript and appendix | Use the bundled MLSys style files with a standard LaTeX toolchain |
| `dls2026/` | Six-page IEEE MASS DLS workshop version and internal supplement | `make -C paper/dls2026` |

The manuscript values must be traceable to `results/`. The executable value
ledger is `mlsys2026/figures/verify_figure_data.py`; run it before committing a
paper change:

```bash
python paper/mlsys2026/figures/verify_figure_data.py
```

Editable Draw.io, SVG, PDF, and Python figure sources are retained. Generated
LaTeX auxiliaries, TIFF exports, and visual-QA rasters are ignored.

The manuscripts remain anonymous. Replace placeholders and add citation
metadata only when the applicable venue policy permits de-anonymization.
