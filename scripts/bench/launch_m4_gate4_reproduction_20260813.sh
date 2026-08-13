#!/usr/bin/env bash
set -Eeuo pipefail

WORKTREE=/root/autodl-tmp/MLSys_M2_20260812
PYTHON=/root/autodl-tmp/MLSys_Research/.venv/bin/python
OUTPUT_ROOT=/root/autodl-tmp/m4-four-config-formal-20260813
ATTEMPT_ID=m4-four-config-serving-repro-39739e0-20260813-r3
PARENT_ATTEMPT=m4-four-config-serving-formal-39739e0-20260813-r2
CONFIG=configs/experiments/joint_precision_controller_2b.yaml
LAUNCH_DIR="${OUTPUT_ROOT}/launch/${ATTEMPT_ID}"
HARD_TIMEOUT_S=21600

export PYTHONPATH="${WORKTREE}/src:${WORKTREE}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

guard_resources() {
    local system_avail_kb data_avail_kb gpu_used
    system_avail_kb=$(df -Pk / | awk 'NR == 2 {print $4}')
    data_avail_kb=$(df -Pk /root/autodl-tmp | awk 'NR == 2 {print $4}')
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if (( system_avail_kb < 8 * 1024 * 1024 )); then
        printf 'system disk safety threshold reached: %s KiB available\n' "${system_avail_kb}" >&2
        return 70
    fi
    if (( data_avail_kb < 10 * 1024 * 1024 )); then
        printf 'data disk safety threshold reached: %s KiB available\n' "${data_avail_kb}" >&2
        return 71
    fi
    if (( gpu_used > 512 )); then
        printf 'GPU is not idle: %s MiB used\n' "${gpu_used}" >&2
        return 72
    fi
}

guard_resources
if [[ -n "$(git -C "${WORKTREE}" status --porcelain=v1 --untracked-files=all)" ]]; then
    printf 'remote experiment worktree is dirty\n' >&2
    exit 73
fi
if [[ -e "${OUTPUT_ROOT}/${ATTEMPT_ID}" || -e "${LAUNCH_DIR}" ]]; then
    printf 'refusing to overwrite attempt or launch evidence: %s\n' "${ATTEMPT_ID}" >&2
    exit 64
fi

command=(
    "${PYTHON}" scripts/bench/run_steady_state.py
    --config "${CONFIG}"
    --phase confirmatory
    --attempt-id "${ATTEMPT_ID}"
    --parent-attempt "${PARENT_ATTEMPT}"
    --output-root "${OUTPUT_ROOT}"
)

mkdir -p "${LAUNCH_DIR}"
printf '%s\n' "${PARENT_ATTEMPT}" >"${LAUNCH_DIR}/parent_attempt"
printf '%s\n' "${WORKTREE}" >"${LAUNCH_DIR}/working_directory"
printf '%s\n' "${HARD_TIMEOUT_S}" >"${LAUNCH_DIR}/hard_timeout_s"
printf '%s\n' logic_review_only >"${LAUNCH_DIR}/evidence_review_mode"
printf '%s\n' "$$" >"${LAUNCH_DIR}/pid"
printf '%q ' "${command[@]}" >"${LAUNCH_DIR}/command.txt"
printf '\n' >>"${LAUNCH_DIR}/command.txt"
date -Is >"${LAUNCH_DIR}/started_at"

finalize() {
    local exit_code=$?
    printf '%s\n' "${exit_code}" >"${LAUNCH_DIR}/exit_code.tmp"
    mv "${LAUNCH_DIR}/exit_code.tmp" "${LAUNCH_DIR}/exit_code"
    date -Is >"${LAUNCH_DIR}/finished_at.tmp"
    mv "${LAUNCH_DIR}/finished_at.tmp" "${LAUNCH_DIR}/finished_at"
}
trap finalize EXIT

cd "${WORKTREE}"
timeout --signal=TERM --kill-after=30s "${HARD_TIMEOUT_S}" \
    "${command[@]}" >"${LAUNCH_DIR}/run.log" 2>&1
