## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: validate
- Origin Date: 2026-08-13T02:38:49.367230+00:00
- Verification Status: VERIFIED
- Version Label: ruler_nothink_gate4_reproduction_v1

## Validation Report

- **Source**: `ruler-statebf16-nothink-repro-20260813-2b` / `ruler-statebf16-nothink-repro-20260813-9b`
- **Overall Confidence**: CAUTION
- **Gate**: Gate 4 PASS
- **Reproducibility**: REPRODUCIBLE (30/30 exact quality and sample-output matches)
- **Review policy**: logical protocol audit; experiment-script hashes were not checked or used as gates

### Statistical Findings

| Cell | n pairs | FP16 mean | State-bf16 mean | Delta (95% t-CI) |
|---|---:|---:|---:|---:|
| 2b ruler_fwe L4096 | 3 | 91.11 | 91.11 | 0.00 [0.00, 0.00] |
| 2b ruler_fwe L8192 | 3 | 98.33 | 98.33 | 0.00 [0.00, 0.00] |
| 9b ruler_niah_multiquery L4096 | 3 | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| 9b ruler_niah_multiquery L8192 | 3 | 100.00 | 100.00 | 0.00 [0.00, 0.00] |
| 9b ruler_fwe L8192 | 3 | 100.00 | 100.00 | 0.00 [0.00, 0.00] |

### Warnings

- The three paired dataset seeds provide low power. Exact observed equality does not establish equivalence or non-inferiority.
- Degenerate zero-width intervals reflect identical observed outputs, not population-level certainty.
- No p-values are used; the five predeclared cells are reported in full, without outcome selection.

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | All five cells are reported separately by task, length, and model; aggregate direction is not substituted for strata. |
| Ecological fallacy | CAUTION | The unit of analysis and inference is the paired dataset seed, not an individual request or token. |
| Berkson's paradox | NOTE | The five cells were fixed by the protocol-repair question before this reproduction; no outcome-based admission filter was applied. |
| Collider bias | NOTE | No post-treatment variable is conditioned on; allocation pairs share the same frozen samples. |
| Base rate neglect | NOTE | Accuracy is reported with its exact 20-sample denominator; this is not a diagnostic sensitivity/specificity claim. |
| Regression to the mean | NOTE | The independent temporal rerun repeats every cell and is not selected from extreme parent outcomes. |
| Survivorship bias | NOTE | The exact 30-cell matrix is required; any missing, failed, or extra cell fails validation. |
| Look-elsewhere effect | CAUTION | Five task/length/model cells, two allocations, and three dataset seeds were frozen before execution; no cell is selected by result. |
| Garden of forking paths | CAUTION | Thinking mode, token budget, seeds, allocations, matrix, comparison rule, and denominator are fixed in the contract. |
| Correlation != causation | NOTE | The report describes observed accuracy agreement and does not attribute a causal mechanism to state dtype. |
| Reverse causality | NOTE | Allocation is experimentally assigned before generation, so outcome-to-allocation reverse direction is not applicable. |

### Reproducibility

The parent and reproduction use the same five-cell no-think matrix, allocations, dataset seeds, engine seed, token budget, model revisions, and evaluator semantics. Host and elapsed-time fields are intentionally excluded from quality comparison.

All 30 primary accuracy values and all sample-level predictions, references, hit vectors, and token counts match exactly. Result JSON sidecars pass; no experiment-script hash was inspected.

### Evidence Boundary

This result supports only empirical agreement between fp32-state and bf16-state for the tested RULER cells. It does not prove general equivalence, cross-task quality preservation, or a serving mechanism.
