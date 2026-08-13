# M4 Four-Configuration Serving Formal Audit

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-13
- Verification Status: ANALYZED
- Version Label: m4_four_config_final_checkpoint_v1

## Verdict

The M4 required experiment package is execution-complete and has passed the
Gate 3 integrity audit, but it did not pass the frozen Gate 4 run-stability
criterion. Its quantitative results remain `ANALYZED` and are not authorized
as `VERIFIED` paper evidence.

- Reproducibility verdict: `NOT_REPRODUCIBLE`
- Overall confidence: `CAUTION`
- No threshold was changed after observing the results.
- No third formal run is scheduled to seek a favorable result.
- The rerun uses the same seeds and is run stability, not an independent
  replication or additional sample size.

## Attempts And Environment

- Original: `m4-four-config-serving-formal-39739e0-20260813-r2`
- Rerun: `m4-four-config-serving-repro-39739e0-20260813-r3`
- Frozen root commit: `39739e03012747bd622175b9a37996297c1b1519`
- Local archive ref: `refs/archive/m4-frozen-39739e0`
- Frozen vLLM commit: `e2fa28594f7baad142a426b0b6a2cfe2c79201c7`
- Hardware: one NVIDIA RTX 5090
- Matrix: `full`, `kv_only`, `state_only`, and `joint`; Random and
  ShareGPT workloads; seeds 11, 23, and 47; 144 cells per attempt
- Raw attempts remain on the data disk under
  `/root/autodl-tmp/m4-four-config-formal-20260813/`.
- Script hash verification was skipped per user instruction; logic and
  realized-configuration evidence were reviewed instead.

## Integrity

Both attempts contain 144/144 validated cells and 320,400/320,400 completed
measurement requests with zero failed requests. The rerun has zero residual
`.partial` artifacts, complete launcher provenance, exit code 0, and four
cleanly stopped server sessions. Logs confirm the intended KV/state precision
for all four configurations and `CUDAGraphMode.PIECEWISE`.

The r3 launcher ran from `2026-08-13T17:50:01+08:00` to
`2026-08-13T21:42:43+08:00` (13,962 seconds). The final analyzer fails closed
on missing cells, request denominator drift, failed requests, realized
precision drift, nonzero server/launcher exits, and residual partial files.

## Statistical Results

The inferential unit is one allocation-workload-rate-seed cell; individual
requests are not treated as independent repeats. Each attempt has 180 primary
paired goodput comparisons:

`3 alternatives x (5 Random + 7 ShareGPT rates) x 5 TTFT thresholds`.

- r2 BH-FDR survivors at q < 0.05: 68/180
- r3 BH-FDR survivors at q < 0.05: 73/180
- Fallacy scan coverage: 11/11

These counts differ across attempts and do not establish a universal benefit.
Several low-load comparisons favor full precision, so no one-direction
compression headline is supported.

## Gate 4 Run Stability

The predeclared promotion gate compares all 720 cell-threshold goodput values
using symmetric relative difference strictly below 10%. Wall-clock timing is
excluded from the reproducibility gate.

- Within tolerance: 537/720
- Outside tolerance: 183/720
- Maximum symmetric relative difference: 79.610%
- Exact sustainable point labels: 713/720
- Exact all-seed SLO boundaries: 38/40
- Difference distribution: 346 below 1%, 159 from 1% to 5%, 32 from 5% to
  10%, and 183 at or above 10%

Outside-tolerance comparisons are not confined to near-zero goodput: 180/183
have at least 1 request/s in both attempts and 103/183 have at least 10
requests/s in both attempts. The two non-matching all-seed boundaries are:

- `joint / Random / TTFT 250 ms`: 30 request/s in r2 versus no sustainable
  tested rate in r3.
- `kv_only / Random / TTFT 1000 ms`: 35 request/s in r2 versus 40 request/s
  in r3.

The mostly stable binary SLO labels do not override the failed continuous
goodput promotion gate.

## Archived Evidence

- `analysis-r2/`: complete r2 analysis JSON, comparison CSV, and report
- `analysis-r3/`: complete r3 analysis JSON, comparison CSV, and report
- `gate4-r3/`: all 720 comparisons and the Gate 4 validation report
- `provenance/r2/` and `provenance/r3/`: frozen attempt contracts and summaries
- `provenance/r3-launch/`: exact command, timestamps, PID, timeout, exit code,
  review mode, working directory, and run log

## Scope Closure

M4 is the final required package. E2-E5 enhancement packages and the second
GPU / TP=2/4 package are outside the user-selected scope and were not started.
