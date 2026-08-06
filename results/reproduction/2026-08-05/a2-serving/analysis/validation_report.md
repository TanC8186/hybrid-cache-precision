## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_packed_serving_pilot_audit_v1

## Validation Report

- **MVEx**: `a2-packed-serving-mvex-d1d52c4-westd-01`
- **Pilot**: `a2-packed-serving-pilot-d1d52c4-westd-01`
- **Pilot Verdict**: `PILOT_FAILED_RUNTIME_CUDA_ILLEGAL_INSTRUCTION`
- **Evidence Status**: `QUARANTINED`
- **Overall A2 Status**: `PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING`

### MVEx

The MVEx completed 1,800/
1,800 requests with zero failures,
arrival ratio `1.000015`, P99 TTFT
`229.40` ms, and P99 TPOT
`17.03` ms. It remains `UNVERIFIED`.

### Failed Pilot

| Sample | Completed | Failed | Status |
|---|---:|---:|---|
| packed/random/r30/s7 | 611 | 1,189 | completed_validated, then quarantined with the session |
| packed/random/r40/s7 | 0 | 2,400 | failed result validation |
| packed/random/r50/s7 | 0 | 0 | not started |

All issued requests are accounted for; there are no silent exclusions. However,
the shared server session logged both `EngineCore encountered a fatal error` and
`CUDA error: an illegal instruction was encountered`. The rate-40 result had no
successful request, the supervisor exited nonzero, and rate 50 was not started.
The pilot therefore fails Gate 2 and must not be resumed or promoted.

### Next Gate

Run a new parent-linked diagnostic attempt with `CUDA_LAUNCH_BLOCKING=1`. Do not
start packed comparative or formal serving experiments until the failing kernel
is identified and a fresh MVEx plus pilot pass.
