## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: validate
- Origin Date: 2026-08-11
- Verification Status: VERIFIED
- Version Label: capacity_phase_r3_gate4_report_v1

## Validation Report

- **Source**: `capacity-phase-repro-r2-20260811` -> `capacity-phase-repro-r3-20260811`
- **Overall Confidence**: SOLID
- **Reproducibility Verdict**: REPRODUCIBLE
- **Promotion Status**: PASS
- **Gate**: Capacity Gate 4 complete

### Completeness And Integrity

| Check | Observed | Status |
|---|---:|---|
| Formal exit code | 0 | PASS |
| Requested/completed cells | 112/112 | PASS |
| `[OK]` / `[FAIL]` / `[DONE]` | 112 / 0 / 1 | PASS |
| `[SKIP]` / `[RETRY]` | 0 / 0 | PASS |
| Cell JSON / SHA-256 sidecars | 112 / 112 | PASS |
| Missing or mismatched R3 artifact hashes | 0 | PASS |
| Core bf16-state direction pairs | 52/52 positive | PASS |
| Local independent analysis replay | Semantic exact match | PASS |
| Local independent validation replay | Semantic exact match | PASS |

### Statistical Findings

| Metric | R2 | R3 | Difference | Status |
|---|---:|---:|---:|---|
| Maximum cell token symmetric relative difference | N/A | N/A | 1.415962% | WITHIN 2% |
| 2B, fp16 KV median state gain | 13.801364% | 13.801364% | 0.000000 pp | WITHIN 2 pp |
| 2B, int4 KV median state gain | 30.042518% | 30.042518% | 0.000000 pp | WITHIN 2 pp |
| 9B, fp16 KV median state gain | 9.303324% | 9.303324% | 0.000000 pp | WITHIN 2 pp |
| 9B, int4 KV median state gain | 27.302925% | 26.374380% | 0.928545 pp | WITHIN 2 pp |

The largest cell difference is the 9B int4-KV, bfloat16-state,
length-32768, utilization-0.8 cell: 528,969 tokens in R2 versus 521,479
in R3. The largest derived per-pair gain shift is 1.984495 percentage
points for 9B int4-KV, length 4096, utilization 0.8. Per-pair gain
difference was not a preregistered threshold; the preregistered pair criterion
was direction and the group-level criterion was median gain. Both pass. This
near-2 pp derived value should remain visible if later work makes a cell-level
effect-size claim.

The unit of analysis is one frozen configuration cell. This is an
environment-sensitive allocator benchmark, so timing fields are excluded.
There are no null-hypothesis tests, p-values, or stochastic replicate means;
confidence intervals and multiple-comparison p-value correction are therefore
not applicable. Effect magnitude is reported directly as paired capacity gain
and reproduction difference over the exhaustive frozen matrix.

### Warnings And Claim Boundaries

| Type | Detail | Affected Evidence |
|---|---|---|
| Environment scope | The clean chain covers the recorded RTX 5090, driver, vLLM runtime, and two Qwen3.5 model sizes. | Cross-GPU and cross-runtime generalization |
| Mechanism scope | Capacity construction was measured without generation; timing metrics were excluded. | Serving latency, goodput, and quality claims |
| Derived pair variability | One 9B int4 pair has a 1.984495 pp gain shift while all frozen primary tolerances pass. | Fine-grained cell-level effect-size claims |
| Historical evidence | The provenance-defective old formal remains quarantined; R2 and R3 form the clean promotion chain. | Evidence lineage only |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked

| Fallacy | Severity | Finding | Recommendation |
|---|---|---|---|
| Simpson's paradox | NOTE | All 52 paired directions are positive, and model-by-KV subgroup medians preserve the aggregate direction. No reversal was observed. | Retain subgroup results with aggregate summaries. |
| Ecological fallacy | NOTE | The inference unit remains the configuration cell; no request-level or user-level inference is made. | Do not generalize capacity cells to serving requests. |
| Berkson's paradox | NOTE | The exhaustive frozen 112-cell matrix completed without post hoc selection. | Preserve the full denominator. |
| Collider bias | NOTE | No adjusted regression or conditioning on a common effect is used. | Not applicable to this paired deterministic comparison. |
| Base-rate neglect | NOTE | No diagnostic classification metric is reported. | Not applicable. |
| Regression to the mean | NOTE | This is not an extreme-selected pre/post design. | Not applicable. |
| Survivorship bias | NOTE | All requested cells completed; no failed, skipped, or retried cell was removed from the denominator. | Continue fail-closed accounting. |
| Look-elsewhere effect | NOTE | The matrix, primary metrics, and tolerances were frozen before R3; all cells and group medians are disclosed. | Keep later exploratory slicing separate. |
| Garden of forking paths | NOTE | R3 was frozen after the validator-entry fix and ran without command, threshold, matrix, or denominator changes. | Preserve the R2 validator failure as historical evidence. |
| Correlation versus causation | NOTE | The controlled result supports the allocator configuration contrast on this runtime, not end-to-end serving or quality causation. | Restrict claims to measured capacity behavior. |
| Reverse causality | NOTE | State dtype is configured before deterministic capacity construction; no reverse temporal path applies. | Not applicable beyond the stated mechanism. |

### Reproducibility

- **Method**: full environment-sensitive re-run under the same hash-frozen
  runtime; structure, model/config inputs, arguments, resolved dtypes, capacity
  directions, per-cell tokens, and group medians compared; timing excluded.
- **Result**: all 112 cells and 52 core pairs passed with zero token,
  direction, or median-gain failures.
- **Verdict**: REPRODUCIBLE.
- **Evidence Status**: VERIFIED.

### Observation, Inference, Recommendation

- **Observation**: R3 is complete and internally valid, and its comparison
  with the clean R2 parent passes every declared tolerance.
- **Inference**: The measured capacity advantage from bfloat16 state storage is
  reproducible for the frozen 2B/9B, fp16/int4-KV matrix on the recorded RTX
  5090 runtime.
- **Recommendation**: Promote the R2-to-R3 capacity chain as `VERIFIED`
  evidence, retain the old formal as quarantined provenance, and use the clean
  capacity result as the dependency for the selector/controller experiment.
