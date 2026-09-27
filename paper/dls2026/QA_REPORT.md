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

## Camera-ready revision (2026-09-27)

The paper was accepted at DLS 2026 (reviews: Weak Accept 4, Strong Accept 5).
The camera-ready addresses every actionable comment; the per-comment mapping
and the response skeleton live outside the repository (peer-review comments are
not redistributed).

- R1 (generalizability): Limitations paragraph rewritten as "Scope and
  transferability"; the accounting is stated as parameterized by the two byte
  terms so it can be re-instantiated for another hybrid stack.
- R1 (recurrent-state dominance): Section III now defines the state share
  `s(L) = G/(A L + G)` and the crossover length `L* = G/A`, with values
  recomputed by the verifier from the frozen architecture parameters.
- R1 (artifact): the Limitations and supplement artifact statements now cite
  the public repository and its `dls2026-camera-ready` tag. Internal
  review-process notes were removed from the repository in a separate commit.
- R2 (safety framing): the "safety-relevant configuration" claim was removed;
  fail-closed evaluation is presented as a property of the offline selector,
  and the agentic/adversarial-safety boundary sentence is retained.
- R2 (practical implications): the Introduction now lists three practitioner
  implications, and the Abstract states the fail-closed admission rule.
- R2 (method framing): the Abstract, Introduction contributions, and Related
  Work present the byte-exact accounting and discrete reconstruction as the
  main methodological contribution, with the matrix results framed as its
  instantiation.
- R2 (scope): tensor parallelism, prefix caching, and offloading are named as
  unmeasured settings that change the terms or the layout.

Re-verified after the revision:

- `main.pdf`: 6 pages, 612 x 792 pt (US Letter); PDF metadata (title, authors,
  keywords) now set explicitly; abstract 248 words by mechanical count with
  LaTeX commands stripped (the venue submission form allows 20--250)
- No overfull boxes and no content outside the text block: rendered ink extents
  are rows 51--718 and columns 47--563 against a text block of roughly
  45--567 pt at 72 dpi (float pages 3 and 5 end early by design). Three
  underfull notices remain and were inspected: two `\vbox` slack warnings on
  the float and balanced pages, and one mildly loose line (badness 1502) in the
  artifact paragraph where the URL breaks
- Figures 3 and 4 are set at `0.95\linewidth`; `xurl` is loaded so the artifact
  URL can break inside a column
- `supplement.pdf`: 4 pages
- Figure/data verifier: PASS, including the new crossover and state-share
  recomputation (`verify_state_share_crossover`)
- A clean rebuild in an isolated copy reproduces the committed PDF exactly
  (identical extracted text, 6 pages)
