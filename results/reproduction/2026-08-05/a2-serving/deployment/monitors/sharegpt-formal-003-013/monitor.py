"""Persistently monitor the ShareGPT formal continuation without mutating it."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path

BASE = Path("/root/autodl-tmp/a2-serving-20260805-f7a79f5")
ATTEMPT = "a2-comparative-serving-sharegpt300-formal-v3-piecewise-3108650-westd-01"
ORCHESTRATOR_NAME = os.environ.get(
    "ORCHESTRATOR_NAME",
    "sharegpt-formal-slices-003-013",
)
MONITOR_NAME = os.environ.get(
    "MONITOR_NAME",
    "sharegpt-formal-003-013",
)
ORCHESTRATOR = BASE / "orchestrators" / ORCHESTRATOR_NAME
ATTEMPT_DIR = BASE / "attempts" / ATTEMPT
MONITOR_DIR = BASE / "monitors" / MONITOR_NAME
INTERVAL_S = 60
SAMPLE_PATTERN = re.compile(r"sample_id=([^\s]+)")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def read_pid(path: Path) -> int | None:
    value = read_text(path)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip()


def current_processes() -> tuple[list[dict[str, object]], str | None]:
    output = command_output(["ps", "-eo", "pid,ppid,etimes,rss,args"])
    processes: list[dict[str, object]] = []
    active_sample = None
    for line in output.splitlines()[1:]:
        if not any(
            marker in line
            for marker in (
                "run_steady_state.py",
                "vllm serve",
                "vllm bench serve",
            )
        ):
            continue
        fields = line.strip().split(maxsplit=4)
        if len(fields) != 5:
            continue
        pid, ppid, elapsed_s, rss_kib, args = fields
        processes.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "elapsed_s": int(elapsed_s),
                "rss_kib": int(rss_kib),
                "args": args,
            }
        )
        match = SAMPLE_PATTERN.search(args)
        if match:
            active_sample = match.group(1)
    return processes, active_sample


def summary_counts() -> dict[str, int] | None:
    summary_path = ATTEMPT_DIR / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return {str(key): int(value) for key, value in summary["counts"].items()}


def append_record(record: dict[str, object]) -> None:
    output_path = MONITOR_DIR / "monitor.jsonl"
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def monitor() -> int:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    (MONITOR_DIR / "monitor.pid").write_text(f"{os.getpid()}\n", encoding="ascii")
    (MONITOR_DIR / "started_at.txt").write_text(f"{utc_now()}\n", encoding="ascii")

    orchestrator_pid = read_pid(ORCHESTRATOR / "orchestrator.pid")
    while True:
        processes, active_sample = current_processes()
        failure = read_text(ORCHESTRATOR / "failure.txt")
        status = read_text(ORCHESTRATOR / "status.txt")
        alive = process_alive(orchestrator_pid)
        record = {
            "timestamp": utc_now(),
            "orchestrator_pid": orchestrator_pid,
            "orchestrator_alive": alive,
            "orchestrator_status": status,
            "failure": failure,
            "summary_counts": summary_counts(),
            "active_sample": active_sample,
            "processes": processes,
            "gpu": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.free,"
                    "temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ]
            ),
        }
        append_record(record)

        if failure:
            exit_code = 2
            final_status = "FAILED"
            break
        if status == "PASSED":
            exit_code = 0
            final_status = "PASSED"
            break
        if not alive:
            exit_code = 3
            final_status = "ORCHESTRATOR_STOPPED_UNEXPECTEDLY"
            break
        time.sleep(INTERVAL_S)

    (MONITOR_DIR / "status.txt").write_text(
        f"{final_status}\n",
        encoding="ascii",
    )
    (MONITOR_DIR / "exit_code.txt").write_text(
        f"{exit_code}\n",
        encoding="ascii",
    )
    (MONITOR_DIR / "finished_at.txt").write_text(
        f"{utc_now()}\n",
        encoding="ascii",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(monitor())
