# DLS 2026 migration QA

Checked on 2026-08-15.

## Venue contract

- Venue: 2nd Workshop on Data Security and LLM Safety in Smart Systems (DLS),
  co-located with IEEE MASS 2026
- Format: IEEE conference proceedings, US Letter, 10 pt, single-spaced,
  two columns
- Limit: six pages including figures, tables, and references
- Review: single-blind
- Submission deadline shown by the website: 2026-08-15

Source: <https://bds-sdu.github.io/DLS-2026/#submission-guidelines>

## Main manuscript

- `main.pdf`: 6 pages, 612 x 792 pt (US Letter)
- Official `IEEEtran` conference class loaded from the bundled v1.8b file
- `IEEEtran.bst` v1.14 bibliography style
- Inter-column separation is explicitly set to 0.20 in, above the 0.12 in
  submission-check threshold; the page-2 body-ink gap measures 12.843 pt
  (0.1784 in) after rendering
- Single-blind author block identifies Junxi Tan, Yinhao Xiao, and Wencheng
  Yang with their respective English-language affiliations and emails
- No overfull boxes, undefined citations/references, or float-too-large errors
- Abstract length: 227 words by the repository's mechanical count, within the
  submission form's 20--250-word range
- One expected underfull-vbox notice on the balanced final page; it was
  visually inspected
- All fonts embedded
- No raster images reported by `pdfimages`; manuscript figures remain vector
- Minimum figure text after LaTeX scaling: 5.100 pt
- Every page inspected in `qa/main-contact-sheet.png`

Technical results, numerical values, captions, and claim boundaries from the
MLSys manuscript were preserved. The capability-boundary table was converted
to prose, leaving no table in the Introduction or Background section. Fig. 1 is
at the top of page 2, with Fig. 2 directly below it to retain the six-page
limit. Other DLS-specific changes include the document class, author block,
keywords, IEEE bibliography formatting, float flushing, and final-page column
balancing.

## Internal supplement

- `supplement.pdf`: 4 pages, US Letter
- Compiles without warnings or undefined references
- All fonts embedded
- Minimum figure text after LaTeX scaling: 6.207 pt
- Every page inspected in `qa/supplement-contact-sheet.png`

The DLS call does not state that supplementary material is accepted. Treat this
file as internal unless EDAS or the workshop chairs explicitly allow it.

## Evidence integrity

- Figure/data verifier: PASS
- Frozen source hashes verified: 16
- SVG/PDF vector pairs verified: 8
- Copied figure sources match the MLSys version by SHA-256, except the
  byte-to-page figure (now Fig. 1) source and exports, whose orange
  stage-transition arrow was corrected to point horizontally right
- `capacity_matrix_table.tex` and `selector_audit_table.tex` match the source
  versions by SHA-256

## Required before upload

- Confirm whether EDAS accepts `supplement.pdf`; do not assume it does.
- Note that the linked venue is the DLS workshop co-located with MASS, not the
  IEEE MASS main track. Safety framing is limited to deployment implications:
  the manuscript does not claim agentic or adversarial safety experiments or
  improved security.
