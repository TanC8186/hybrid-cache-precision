# Verified Evidence Snapshot (2026-08-04)

This directory contains small, hash-checked copies of the remote A2 and E3
evidence. Raw sample JSON and logs remain on the server and are indexed by the
E3 artifact manifest.

## A2

- `a2/a2_runtime.json`: packed runtime MVEx, 8/8 checks passed.
- `a2/a2_capacity_gate_c7379f0_v2.json`: capacity gate `PASSED`.
- Evidence scope: root commit `c7379f0`, vLLM commit `55f47685`.
- Status boundary: engineering/runtime/capacity gate passed; independent
  reproduction and packed serving/quality formal runs are still required before
  `VERIFIED`.

## E3

- `e3/formal_gate.json`: protocol-v2 formal gate, 72/72 samples and
  160,200/160,200 requests.
- `e3/reproducibility_report.json`: 80/80 cell comparisons within the declared
  10% tolerance and 60/60 exact boundary matches.
- `e3/validation_report.md`: human-readable validation and 11/11 fallacy scan.
- `e3/artifact_sha256_manifest.json`: per-file SHA-256 inventory for the formal,
  reproduction, upper-neighbor, and launch artifacts.
- `e3/verification_link.json`: link-style promotion from the preserved
  `ANALYZED` formal artifacts to E3 scope `VERIFIED`.
- Evidence scope: root commit `d1d52c4`, vLLM commit `55f47685`.

The two protocol-v1 formal attempts remain `QUARANTINED`. Formal,
reproduction, and upper-neighbor denominators remain independent and must not be
pooled.
