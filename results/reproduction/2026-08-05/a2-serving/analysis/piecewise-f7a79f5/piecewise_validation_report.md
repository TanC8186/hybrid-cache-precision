## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_piecewise_packed_serving_pilot_v1

## Validation Report

- **Diagnostic**: `a2-packed-serving-debug-piecewise-f7a79f5-westd-01`
- **MVEx**: `a2-packed-serving-mvex-piecewise-f7a79f5-westd-01`
- **Pilot**: `a2-packed-serving-pilot-piecewise-f7a79f5-westd-01`
- **Verdict**: `PILOT_PASSED_PIECEWISE_RUNTIME_INTEGRITY`
- **Evidence Status**: `UNVERIFIED`
- **Overall A2 Status**: `PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING`

The PIECEWISE packed pilot completed 7,200/
7,200 measured requests with zero failures. Every attempt
used root commit `f7a79f5`, vLLM `55f47685`, a clean worktree,
and server logs that prove `CUDAGraphMode.PIECEWISE`. Post-benchmark health
checks passed before each sample was published.

### Pilot Results

| Offered req/s | Completed | Failed | Throughput | P99 TTFT ms | P99 TPOT ms | Sustainable TTFT thresholds ms |
|---:|---:|---:|---:|---:|---:|---|
| 30 | 1,800 | 0 | 29.40 | 251.95 | 22.64 | 250, 500, 1000, 2000, 3000 |
| 40 | 2,400 | 0 | 38.22 | 1509.46 | 45.73 | 2000, 3000 |
| 50 | 3,000 | 0 | 38.66 | 16129.63 | 45.89 | none |

Rate 50 is a valid overload point: all 3,000 requests completed, but no tested
TTFT threshold met the 0.95 goodput/offered criterion. This negative result is
retained and is not treated as a failed request or silently excluded sample.

### Evidence Boundary

This audit closes the packed-only runtime-integrity pilot under PIECEWISE CUDA
graphs. It does not compare fp16, uniform int4, and packed allocations under the
same graph mode; it has one seed and one synthetic workload. The result remains
`ANALYZED/UNVERIFIED` and cannot enter paper quantitative claims. The next gate
is a fair three-allocation Random/ShareGPT pilot, followed by an independently
frozen multi-seed formal matrix and reproducibility run.

### Fallacy Scan

- **Coverage**: 11/11
