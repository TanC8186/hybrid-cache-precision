## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_comparative_protocol_v3_mvex_v1

## Validation Report

- **FD MVEx**: `a2-comparative-serving-fd-mvex-piecewise-f7a79f5-westd-03`
- **ShareGPT Window MVEx**: `a2-comparative-serving-sharegpt300-mvex-piecewise-37ce9e3-westd-01`
- **Verdict**: `PROTOCOL_V3_MVEX_CHAIN_REVIEW_REQUIRED`
- **Evidence Status**: `QUARANTINED`
- **Overall A2 Status**: `PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING`

The client-FD MVEx completed 3,000/3,000 requests with zero failures after
raising soft `nofile` from 1024 to 65535. The linked ShareGPT MVEx completed
45,000/
45,000 requests with
0 failures under a 300-second arrival
window while preserving the trace completion-length distribution.

### ShareGPT Results

| Allocation | Offered req/s | Completed | Failed | Throughput/offered | P99 TTFT ms | P99 TPOT ms | Drain s | Sustainable TTFT thresholds ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp16 | 20 | 6,000 | 0 | 0.9869 | 124.88 | 14.41 | 3.99 | 250, 500, 1000, 2000, 3000 |
| fp16 | 30 | 9,000 | 0 | 0.9821 | 162.17 | 19.45 | 5.47 | 250, 500, 1000, 2000, 3000 |
| int4 | 20 | 6,000 | 0 | 0.9804 | 134.26 | 19.36 | 5.99 | 250, 500, 1000, 2000, 3000 |
| int4 | 30 | 9,000 | 0 | 0.9746 | 192.94 | 25.87 | 7.82 | 250, 500, 1000, 2000, 3000 |
| packed_per_layer | 20 | 6,000 | 0 | 0.9819 | 130.56 | 18.73 | 5.53 | 250, 500, 1000, 2000, 3000 |
| packed_per_layer | 30 | 9,000 | 0 | 0.9760 | 179.42 | 23.71 | 7.38 | 250, 500, 1000, 2000, 3000 |

### Gate Result

Request conservation and runtime-integrity checks passed, but the predeclared
rate-20/rate-30 ShareGPT bracket did not hold for every allocation. This is a
scientific gate failure, not a missing-artifact failure. The complete MVEx is
retained as a negative bracketing result and does not permit a comparative
pilot.

### Evidence Boundary

The chain is `ANALYZED/QUARANTINED`. Its requests remain available for
protocol diagnosis but contribute zero rows to a pilot or formal efficacy
denominator. A linked upper-neighbor MVEx must use a new attempt ID; the failed
protocol-v2 pilot also remains separately `QUARANTINED`.

### Fallacy Scan

- **Coverage**: 11/11
