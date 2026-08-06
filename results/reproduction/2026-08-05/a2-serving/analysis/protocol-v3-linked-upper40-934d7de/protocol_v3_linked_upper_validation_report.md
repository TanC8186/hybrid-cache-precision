## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_comparative_protocol_v3_linked_upper_mvex_v1

## Validation Report

- **FD MVEx**: `a2-comparative-serving-fd-mvex-piecewise-f7a79f5-westd-03`
- **Lower MVEx**: `a2-comparative-serving-sharegpt300-mvex-piecewise-37ce9e3-westd-01`
- **Upper MVEx**: `a2-comparative-serving-sharegpt300-upper-piecewise-934d7de-westd-01`
- **Verdict**: `PROTOCOL_V3_LINKED_BRACKET_MVEX_REVIEW_REQUIRED`
- **Evidence Status**: `QUARANTINED`
- **Overall A2 Status**: `PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING`

### ShareGPT Results

| Attempt role | Allocation | Offered req/s | Completed | Failed | Throughput/offered | P99 TTFT ms | P99 TPOT ms | Drain s | Sustainable TTFT thresholds ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| lower_mvex | fp16 | 20 | 6,000 | 0 | 0.9869 | 124.88 | 14.41 | 3.99 | 250, 500, 1000, 2000, 3000 |
| lower_mvex | fp16 | 30 | 9,000 | 0 | 0.9821 | 162.17 | 19.45 | 5.47 | 250, 500, 1000, 2000, 3000 |
| lower_mvex | int4 | 20 | 6,000 | 0 | 0.9804 | 134.26 | 19.36 | 5.99 | 250, 500, 1000, 2000, 3000 |
| lower_mvex | int4 | 30 | 9,000 | 0 | 0.9746 | 192.94 | 25.87 | 7.82 | 250, 500, 1000, 2000, 3000 |
| lower_mvex | packed_per_layer | 20 | 6,000 | 0 | 0.9819 | 130.56 | 18.73 | 5.53 | 250, 500, 1000, 2000, 3000 |
| lower_mvex | packed_per_layer | 30 | 9,000 | 0 | 0.9760 | 179.42 | 23.71 | 7.38 | 250, 500, 1000, 2000, 3000 |
| upper_mvex | fp16 | 40 | 12,000 | 0 | 0.9709 | 218.46 | 26.77 | 9.00 | 250, 500, 1000, 2000, 3000 |
| upper_mvex | int4 | 40 | 12,000 | 0 | 0.9570 | 248.14 | 31.89 | 13.48 | 500, 1000, 2000, 3000 |
| upper_mvex | packed_per_layer | 40 | 12,000 | 0 | 0.9587 | 239.17 | 30.51 | 12.93 | 250, 500, 1000, 2000, 3000 |

### Gate Result

The linked rate-30/rate-40 bracket did not pass every threshold. No comparative pilot is permitted.

### Evidence Boundary

The FD, lower, and upper attempts retain independent denominators. The lower
20/30 attempt remains `QUARANTINED` as a standalone failed bracket; this linked
audit uses its intact rate-30 gate observation without promoting any MVEx row
to an efficacy claim. A passing chain permits a new single-seed pilot only.

### Fallacy Scan

- **Coverage**: 11/11
