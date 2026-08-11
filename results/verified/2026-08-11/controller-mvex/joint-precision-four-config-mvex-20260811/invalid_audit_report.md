## Material Passport

- Origin Skill: experiment-skill
- Origin Mode: run
- Origin Date: 2026-08-11
- Verification Status: INVALID
- Version Label: joint_precision_mvex_invalid_audit_v1

## Experiment Result

- **ID**: `joint-precision-four-config-mvex-20260811`
- **Type**: generic serving minimum viable execution
- **Status**: crashed at the Gate 1 log-proof check
- **Working Directory**: `/root/autodl-tmp/MLSys_Controller_1037a6c`
- **Code Revision**: `99c86b48a6577b44d6389e9975ca6e3146a4d77b`
- **Started**: `2026-08-11T19:03:39+08:00`
- **Finished**: `2026-08-11T19:04:57+08:00`
- **Duration**: 78 seconds
- **Exit Code**: 2

## Denominator Audit

| State | Samples | Measurement requests |
|---|---:|---:|
| Planned | 4 | 7,200 |
| Measured | 0 | 0 |
| Completed and validated | 0 | 0 |
| Not started | 4 | 7,200 |
| Silently excluded | 0 | 0 |

All four predeclared sample IDs remain visible in `attempt/summary.json` as
`not_started`. No latency, throughput, goodput, or failure-rate value may be
computed from this attempt.

## Observations

1. The full-precision server command requested `kv_cache_dtype=auto`,
   `mamba_ssm_cache_dtype=float32`, and PIECEWISE CUDA graphs.
2. The server reached `/health` with HTTP 200 after 73.954 seconds.
3. Its log contains allocation-specific resolved proof:
   `'mamba_ssm_cache_dtype': 'float32'`, `kv_cache_dtype=auto`, and
   `CUDAGraphMode.PIECEWISE`.
4. The runner then rejected the server because the frozen config also required
   `Using the user-specified value`, a generic warning absent from this vLLM
   build.
5. The server was terminated before warmup or measurement. Port 8000 and GPU
   memory were clean after termination.

## Integrity Verdict

Gate 1 failed closed as designed, but the proof specification was incorrect.
The attempt is `INVALID`, is retained for provenance, and is excluded from all
paper and calibration denominators. Exit code 0 in the server status describes
the controlled shutdown; it does not change the attempt-level exit code 2 or
the zero-measurement denominator.

## Statistical Scope

No statistical interpretation is possible because no sample entered
measurement. Effect sizes, uncertainty, multiple comparisons, reproducibility,
and the 11 statistical-fallacy checks are not applicable at this stage. They
remain mandatory for later measured evidence.

## Corrective Action

Freeze a new attempt, `joint-precision-four-config-mvex-r2-20260811`, with this
attempt as its parent. Require allocation-specific state dtype, resolved KV
dtype, and PIECEWISE graph-mode substrings; add positive and cross-allocation
negative tests before launching on westb. The failed attempt ID must never be
reused or resumed.
