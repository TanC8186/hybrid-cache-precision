#!/usr/bin/env bash
set -Eeuo pipefail

WORKTREE=/root/autodl-tmp/MLSys_Controller_r2_20260811
OUTPUT_ROOT=/root/autodl-tmp/controller-calibration-r1-20260811
ATTEMPT_ID=joint-precision-four-config-calibration-r1-20260811
PARENT_ATTEMPT=joint-precision-four-config-mvex-r2-20260811
ATTEMPT_DIR="${OUTPUT_ROOT}/${ATTEMPT_ID}"
LAUNCH_PARENT="${OUTPUT_ROOT}/launch"
LAUNCH_DIR="${LAUNCH_PARENT}/${ATTEMPT_ID}"

mkdir -p "${LAUNCH_PARENT}"
if [[ -e "${ATTEMPT_DIR}" || -e "${LAUNCH_DIR}" ]]; then
    printf 'refusing to overwrite existing attempt or launch directory\n' >&2
    exit 64
fi
mkdir "${LAUNCH_DIR}"

printf '%s\n' "$$" >"${LAUNCH_DIR}/pid"
date -Is >"${LAUNCH_DIR}/started_at"

record_exit() {
    local exit_code=$?
    set +e
    printf '%s\n' "${exit_code}" >"${LAUNCH_DIR}/exit_code.tmp.$$"
    mv "${LAUNCH_DIR}/exit_code.tmp.$$" "${LAUNCH_DIR}/exit_code"
    date -Is >"${LAUNCH_DIR}/finished_at"
}
trap record_exit EXIT

source /etc/profile.d/mlsys-data-disk.sh
cd "${WORKTREE}"

{
    date -Is
    df -hT / /root/autodl-tmp
} >"${LAUNCH_DIR}/run.log"

system_free_kib=$(df --output=avail / | tail -n 1)
data_free_kib=$(df --output=avail /root/autodl-tmp | tail -n 1)
if ((system_free_kib < 8 * 1024 * 1024)); then
    printf 'system disk has less than 8 GiB free\n' >>"${LAUNCH_DIR}/run.log"
    exit 65
fi
if ((data_free_kib < 10 * 1024 * 1024)); then
    printf 'data disk has less than 10 GiB free\n' >>"${LAUNCH_DIR}/run.log"
    exit 66
fi

PYTHONPATH="${WORKTREE}/src:${WORKTREE}" \
PYTHONDONTWRITEBYTECODE=1 \
timeout --signal=TERM --kill-after=30s 28800 \
    /root/autodl-tmp/MLSys_Research/.venv/bin/python \
    scripts/bench/run_steady_state.py \
    --config configs/experiments/joint_precision_controller_2b.yaml \
    --phase calibration \
    --attempt-id "${ATTEMPT_ID}" \
    --parent-attempt "${PARENT_ATTEMPT}" \
    --output-root "${OUTPUT_ROOT}" \
    >>"${LAUNCH_DIR}/run.log" 2>&1
