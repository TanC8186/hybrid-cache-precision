# DLS 2026 submission

This directory is the IEEE MASS workshop version of the manuscript. The
original MLSys version remains under `../mlsys2026`.

Venue requirements verified from the DLS 2026 website on 2026-08-14:

- IEEE conference proceedings format (`IEEEtran`, conference mode)
- US Letter paper, 10 pt, single-spaced, two columns
- at most six pages, including figures, tables, and references
- single-blind review

The directory bundles `IEEEtran.cls` v1.8b and `IEEEtran.bst` v1.14 from the
TeX Live IEEEtran distribution so the source package is self-contained.

Build the submission PDF with:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`supplement.tex` preserves the former supplementary material for internal use.
The DLS call does not state that supplementary material is accepted, so do not
upload it unless the workshop chairs or EDAS explicitly permit it.

Before submission, replace the clearly marked author placeholders in
`main.tex`. The source manuscript did not contain real author metadata.

Official requirements: <https://bds-sdu.github.io/DLS-2026/#submission-guidelines>
