## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-05
- Verification Status: ANALYZED
- Version Label: a2_comparative_protocol_v3_rate50_linked_mvex_v1

## Validation Report

- **FD MVEx**: `a2-comparative-serving-fd-mvex-piecewise-f7a79f5-westd-03`
- **Lower MVEx**: `a2-comparative-serving-sharegpt300-mvex-piecewise-37ce9e3-westd-01`
- **Preserved rate-40 MVEx**: `a2-comparative-serving-sharegpt300-upper-piecewise-934d7de-westd-01`
- **Rate-50 MVEx**: `a2-comparative-serving-sharegpt300-upper50-piecewise-b3bd79a-westd-01`
- **Verdict**: `PROTOCOL_V3_RATE50_LINKED_BRACKET_MVEX_PASSED`
- **Evidence Status**: `UNVERIFIED`
- **Overall A2 Status**: `PASSED_NOT_VERIFIED_SERVING_QUALITY_PENDING`

### ShareGPT Bracket Evidence

| Attempt role | Allocation | Offered req/s | Completed | Failed | Throughput/offered | P99 TTFT ms | P99 TPOT ms | Drain s | Sustainable TTFT thresholds ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| lower_mvex | fp16 | 30 | 9,000 | 0 | 0.9821 | 162.17 | 19.45 | 5.47 | 250, 500, 1000, 2000, 3000 |
| lower_mvex | int4 | 30 | 9,000 | 0 | 0.9746 | 192.94 | 25.87 | 7.82 | 250, 500, 1000, 2000, 3000 |
| lower_mvex | packed_per_layer | 30 | 9,000 | 0 | 0.9760 | 179.42 | 23.71 | 7.38 | 250, 500, 1000, 2000, 3000 |
| upper40_mvex | fp16 | 40 | 12,000 | 0 | 0.9709 | 218.46 | 26.77 | 9.00 | 250, 500, 1000, 2000, 3000 |
| upper40_mvex | int4 | 40 | 12,000 | 0 | 0.9570 | 248.14 | 31.89 | 13.48 | 500, 1000, 2000, 3000 |
| upper40_mvex | packed_per_layer | 40 | 12,000 | 0 | 0.9587 | 239.17 | 30.51 | 12.93 | 250, 500, 1000, 2000, 3000 |
| upper50_mvex | fp16 | 50 | 15,000 | 0 | 0.9333 | 11941.69 | 32.05 | 21.44 | none |
| upper50_mvex | int4 | 50 | 15,000 | 0 | 0.8263 | 49206.68 | 36.61 | 63.04 | none |
| upper50_mvex | packed_per_layer | 50 | 15,000 | 0 | 0.8809 | 27845.80 | 33.88 | 40.55 | none |

### Gate Result

Rate 30 is sustainable and rate 50 is unsustainable for every allocation at all five TTFT thresholds. The linked protocol-v3 MVEx gate passes without pooling request denominators.

### Evidence Boundary

All four attempts retain independent denominators. The rate-20/30 and rate-40
attempts remain `QUARANTINED` as standalone failed brackets. Their intact gate
observations are linked for protocol validation only; no MVEx request row is a
pilot, formal, or paper efficacy observation. A passing chain permits a new
single-seed comparative pilot.

### Fallacy Scan

- **Coverage**: 11/11
