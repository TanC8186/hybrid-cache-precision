#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/MLSys_Serving_f7a79f5
BASE=/root/autodl-tmp/a2-serving-20260805-f7a79f5
ATTEMPT=a2-comparative-serving-sharegpt300-formal-v3-piecewise-3108650-westd-01
PARENT=a2-comparative-serving-random60-formal-v3-piecewise-3108650-westd-01
CONFIG="$ROOT/experiments/configs/a2_comparative_piecewise_protocol_v3_sharegpt300_formal.yaml"
CONFIG_SHA=13888902c80bda29ce282969ac3200789006554f4373fd1340ee48d8b991aca8
COMMIT=310865011daf2a9d8d694eddded0411fd956fd95
PY=/root/autodl-tmp/MLSys_Research/.venv/bin/python
ATTEMPT_DIR="$BASE/attempts/$ATTEMPT"
SUPERVISOR_ROOT="$BASE/supervisors/$ATTEMPT"
START_SLICE=${START_SLICE:-3}
END_SLICE=${END_SLICE:-13}
ORCH_NAME=${ORCH_NAME:-sharegpt-formal-slices-003-013}
ORCH="$BASE/orchestrators/$ORCH_NAME"

fail() {
  local message=$1
  printf '%s\n' "$message" > "$ORCH/failure.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ORCH/finished_at.txt"
  exit 1
}

run_gate() {
  local expected_completed=$1
  local output_path=$2
  "$PY" - "$ATTEMPT_DIR" "$expected_completed" "$COMMIT" > "$output_path" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

attempt_dir = Path(sys.argv[1])
expected_completed = int(sys.argv[2])
expected_commit = sys.argv[3]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_sidecar(path: Path):
    expected = path.with_name(path.name + ".sha256").read_text(
        encoding="utf-8"
    ).strip()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"hash mismatch: {path}")


contract = load_json(attempt_dir / "attempt_contract.json")
summary = load_json(attempt_dir / "summary.json")
plan = contract["plan"]
if contract["git_commit"] != expected_commit:
    raise RuntimeError("attempt commit mismatch")
if len(plan) != 63:
    raise RuntimeError("unexpected plan size")

expected_counts = {"completed_validated": expected_completed}
if expected_completed < len(plan):
    expected_counts["not_started"] = len(plan) - expected_completed
if summary["counts"] != expected_counts:
    raise RuntimeError(
        f"summary counts mismatch: {summary['counts']} != {expected_counts}"
    )

expected_items = plan[:expected_completed]
expected_ids = [item["sample_id"] for item in expected_items]
expected_set = set(expected_ids)
sample_root = attempt_dir / "samples"
actual_set = {path.name for path in sample_root.iterdir() if path.is_dir()}
if actual_set != expected_set:
    raise RuntimeError("sample directory set mismatch")

total_expected = 0
total_completed = 0
total_failed = 0
for item in expected_items:
    sample_id = item["sample_id"]
    sample_dir = sample_root / sample_id
    status = load_json(sample_dir / "status.json")
    if status["status"] != "completed_validated" or status["returncode"] != 0:
        raise RuntimeError(f"invalid sample status: {sample_id}")
    for name in ("contract.json", "result.json", "analysis.json"):
        verify_sidecar(sample_dir / name)
    result = load_json(sample_dir / "result.json")
    expected_requests = int(item["num_prompts"])
    if result["sample_id"] != sample_id:
        raise RuntimeError(f"sample ID mismatch: {sample_id}")
    if result["git_commit"] != expected_commit:
        raise RuntimeError(f"sample commit mismatch: {sample_id}")
    if int(result["completed"]) != expected_requests:
        raise RuntimeError(f"completed request mismatch: {sample_id}")
    if int(result["failed"]) != 0:
        raise RuntimeError(f"failed requests: {sample_id}")
    total_expected += expected_requests
    total_completed += int(result["completed"])
    total_failed += int(result["failed"])

for name in ("attempt_contract.json", "environment.json", "summary.json"):
    verify_sidecar(attempt_dir / name)

expected_allocations = []
for start in range(0, expected_completed, 5):
    previous = None
    for item in plan[start : min(start + 5, expected_completed)]:
        allocation = str(item["allocation"])
        if allocation != previous:
            expected_allocations.append(allocation)
            previous = allocation
server_status_paths = sorted((attempt_dir / "servers").glob("*/*/status.json"))
actual_allocations = [path.parents[1].name for path in server_status_paths]
if len(server_status_paths) != len(expected_allocations):
    raise RuntimeError(
        "server session count mismatch: "
        f"{len(server_status_paths)} != {len(expected_allocations)}"
    )
if Counter(actual_allocations) != Counter(expected_allocations):
    raise RuntimeError(
        "server allocation sessions mismatch: "
        f"{Counter(actual_allocations)} != {Counter(expected_allocations)}"
    )
for status_path in server_status_paths:
    status = load_json(status_path)
    log_text = status_path.with_name("server.log").read_text(
        encoding="utf-8",
        errors="replace",
    )
    if status["status"] != "stopped":
        raise RuntimeError(f"server not stopped: {status_path}")
    if status["returncode"] != 0 or status["exception"] is not None:
        raise RuntimeError(f"server failed: {status_path}")
    if "CUDAGraphMode.PIECEWISE" not in log_text:
        raise RuntimeError(f"PIECEWISE graph proof missing: {status_path}")
    if "CUDAGraphMode.FULL_AND_PIECEWISE" in log_text:
        raise RuntimeError(f"forbidden graph mode: {status_path}")
    for signature in (
        "EngineCore encountered a fatal error",
        "CUDA error: an illegal instruction was encountered",
        "server health check failed after benchmark",
    ):
        if signature in log_text:
            raise RuntimeError(f"runtime fault in {status_path}: {signature}")

print(
    json.dumps(
        {
            "completed_samples": expected_completed,
            "planned_samples": len(plan),
            "expected_requests": total_expected,
            "completed_requests": total_completed,
            "failed_requests": total_failed,
            "server_sessions": len(server_status_paths),
            "status": "PASSED",
        },
        indent=2,
        sort_keys=True,
    )
)
PY
}

mkdir -p "$ORCH"
printf '%s\n' "$$" > "$ORCH/orchestrator.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ORCH/started_at.txt"

cd "$ROOT" || fail "cannot enter frozen repository"
test "$(git rev-parse HEAD)" = "$COMMIT" || fail "frozen commit mismatch"
test -z "$(git status --porcelain)" || fail "frozen repository is dirty"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$CONFIG_SHA" \
  || fail "formal config hash mismatch"

for slice_number in $(seq "$START_SLICE" "$END_SLICE"); do
  slice_id=$(printf 'slice-%03d' "$slice_number")
  previous_completed=$(( (slice_number - 1) * 5 ))
  if [ "$previous_completed" -gt 63 ]; then
    previous_completed=63
  fi
  expected_completed=$(( slice_number * 5 ))
  if [ "$expected_completed" -gt 63 ]; then
    expected_completed=63
  fi
  slice_dir="$SUPERVISOR_ROOT/$slice_id"

  test ! -e "$slice_dir" || fail "$slice_id already exists"
  mkdir -p "$slice_dir"
  run_gate "$previous_completed" "$slice_dir/precheck.json" \
    || fail "$slice_id precheck failed"
  sha256sum "$slice_dir/precheck.json" \
    | awk '{print $1}' > "$slice_dir/precheck.json.sha256"

  "$PY" - "$ATTEMPT_DIR/attempt_contract.json" "$slice_number" \
    "$slice_dir/contract.json" <<'PY'
import json
import sys
from pathlib import Path

attempt_contract_path = Path(sys.argv[1])
slice_number = int(sys.argv[2])
output_path = Path(sys.argv[3])
attempt_contract = json.loads(attempt_contract_path.read_text(encoding="utf-8"))
plan = attempt_contract["plan"]
start = (slice_number - 1) * 5
items = plan[start : start + 5]
completed = min(slice_number * 5, len(plan))
slice_requests = sum(int(item["num_prompts"]) for item in items)
cumulative_requests = sum(
    int(item["num_prompts"]) for item in plan[:completed]
)
contract = {
    "schema_version": 1,
    "frozen_at": "2026-08-06",
    "attempt_id": attempt_contract["attempt_id"],
    "slice_id": f"slice-{slice_number:03d}",
    "resume": True,
    "max_samples": 5,
    "expected_samples": [item["sample_id"] for item in items],
    "expected_slice_requests": slice_requests,
    "expected_cumulative_completed_samples": completed,
    "expected_cumulative_requests": cumulative_requests,
    "timeout_s": 5400,
    "code_commit": attempt_contract["git_commit"],
    "config_sha256": attempt_contract["config_sha256"],
    "failure_policy": (
        "Do not resume this attempt if any prior or current sample is "
        "failed or incomplete."
    ),
}
output_path.write_text(
    json.dumps(contract, indent=2) + "\n",
    encoding="utf-8",
)
PY
  test $? -eq 0 || fail "$slice_id contract generation failed"

  cat > "$slice_dir/run.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail

ROOT=$ROOT
BASE=$BASE
ATTEMPT=$ATTEMPT
CONFIG="\$ROOT/experiments/configs/a2_comparative_piecewise_protocol_v3_sharegpt300_formal.yaml"
SUPERVISOR="\$BASE/supervisors/\$ATTEMPT/$slice_id"

cd "\$ROOT"
printf '%s\\n' "\$\$" > "\$SUPERVISOR/supervisor.pid"
date -u +%Y-%m-%dT%H:%M:%SZ > "\$SUPERVISOR/started_at.txt"
ulimit -Sn > "\$SUPERVISOR/soft_nofile_before.txt"
ulimit -n 65535
ulimit -Sn > "\$SUPERVISOR/soft_nofile_after.txt"
timeout --signal=TERM --kill-after=30s 5400s bash -lc \\
  "ulimit -n 65535; PYTHONPATH=. $PY scripts/bench/run_steady_state.py \\
    --config '\$CONFIG' \\
    --phase comparative_sharegpt300_formal_v3 \\
    --attempt-id '\$ATTEMPT' \\
    --parent-attempt $PARENT \\
    --output-root '\$BASE/attempts' \\
    --resume \\
    --max-samples 5" \\
  > "\$SUPERVISOR/stdout.log" \\
  2> "\$SUPERVISOR/stderr.log"
ec=\$?
printf '%s\\n' "\$ec" > "\$SUPERVISOR/exit_code.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "\$SUPERVISOR/finished_at.txt"
exit "\$ec"
EOF
  chmod 700 "$slice_dir/run.sh"
  sha256sum "$slice_dir/contract.json" \
    | awk '{print $1}' > "$slice_dir/contract.json.sha256"
  sha256sum "$slice_dir/run.sh" \
    | awk '{print $1}' > "$slice_dir/run.sh.sha256"

  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$slice_id starting" >> "$ORCH/progress.log"
  bash "$slice_dir/run.sh"
  slice_exit=$?
  if [ "$slice_exit" -ne 0 ]; then
    fail "$slice_id runner failed with exit code $slice_exit"
  fi

  run_gate "$expected_completed" "$slice_dir/postcheck.json" \
    || fail "$slice_id postcheck failed"
  sha256sum "$slice_dir/postcheck.json" \
    | awk '{print $1}' > "$slice_dir/postcheck.json.sha256"
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$slice_id passed" >> "$ORCH/progress.log"
done

printf '%s\n' "PASSED" > "$ORCH/status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ORCH/finished_at.txt"
