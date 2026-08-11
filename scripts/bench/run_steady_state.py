"""Run resumable steady-state vLLM serving experiments.

Each allocation/workload/rate/seed tuple is an immutable sample. Results are
published only after the benchmark exits successfully and the detailed JSON
passes schema, denominator, and arrival-window checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import yaml

SCHEMA_VERSION = 1
ANALYSIS_SCHEMA_VERSION = 2
VALID_SAMPLE_STATUS = "completed_validated"
REQUEST_FAILURE_POLICY = "count_as_slo_miss"


class ExperimentError(RuntimeError):
    """Raised when an integrity gate fails."""


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compact_timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with tmp.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    atomic_write_bytes(path, payload + b"\n")


def write_json_with_hash(path: Path, value: Any) -> str:
    atomic_write_json(path, value)
    digest = sha256_file(path)
    atomic_write_bytes(path.with_suffix(path.suffix + ".sha256"), f"{digest}\n".encode())
    return digest


def run_capture(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    allow_failure: bool = False,
) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode and not allow_failure:
        raise ExperimentError(
            f"command failed ({completed.returncode}): {shlex.join(command)}\n{completed.stdout[-4000:]}"
        )
    return completed.stdout.strip()


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ExperimentError(f"config must be a mapping: {path}")
    required = {
        "name",
        "environment",
        "server",
        "protocol",
        "allocations",
        "workloads",
        "phases",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ExperimentError(f"config missing keys: {', '.join(missing)}")
    protocol = config["protocol"]
    failure_policy = protocol.get("request_failure_policy")
    if failure_policy != REQUEST_FAILURE_POLICY:
        raise ExperimentError(
            "protocol.request_failure_policy must be "
            f"{REQUEST_FAILURE_POLICY!r}, got {failure_policy!r}"
        )
    client_keepalive_s = int(protocol["benchmark_client_keepalive_timeout_s"])
    server_keepalive_s = int(
        config["environment"].get("env", {}).get(
            "VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
            5,
        )
    )
    if server_keepalive_s <= client_keepalive_s:
        raise ExperimentError(
            "server HTTP keep-alive must exceed the benchmark client keep-alive: "
            f"server={server_keepalive_s}s client={client_keepalive_s}s"
        )
    return config


def parse_csv(values: str | None, cast=str) -> list[Any] | None:
    if values is None:
        return None
    return [cast(item.strip()) for item in values.split(",") if item.strip()]


def format_rate(rate: float) -> str:
    return f"{rate:g}".replace(".", "p")


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ExperimentError("cannot compute percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def ensure_finite(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ExperimentError(f"{name} is not finite: {parsed!r}")
    return parsed


def get_git_state(repo_root: Path, require_clean: bool) -> dict[str, Any]:
    commit = run_capture(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = run_capture(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
    )
    if require_clean and status:
        raise ExperimentError("commit-before-run gate failed; repository is dirty:\n" + status)
    return {"commit": commit, "status": status, "clean": not bool(status)}


def get_optional_git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    return run_capture(["git", "rev-parse", "HEAD"], cwd=path)


def resolve_phase(
    config: Mapping[str, Any],
    phase_name: str,
    allocation_filter: list[str] | None = None,
    workload_filter: list[str] | None = None,
    seed_filter: list[int] | None = None,
    rate_filter: list[float] | None = None,
) -> dict[str, Any]:
    phases = config["phases"]
    if phase_name not in phases:
        raise ExperimentError(f"unknown phase {phase_name!r}; choose from {', '.join(phases)}")
    phase = phases[phase_name]
    allocations = list(phase["allocations"])
    seeds = [int(seed) for seed in phase["seeds"]]
    workload_rates = {name: [float(rate) for rate in rates] for name, rates in phase["workload_rates"].items()}

    if allocation_filter is not None:
        allocations = [name for name in allocations if name in allocation_filter]
    if seed_filter is not None:
        seeds = [seed for seed in seeds if seed in seed_filter]
    if workload_filter is not None:
        workload_rates = {name: rates for name, rates in workload_rates.items() if name in workload_filter}
    if rate_filter is not None:
        workload_rates = {
            name: [rate for rate in rates if rate in rate_filter] for name, rates in workload_rates.items()
        }

    unknown_allocations = sorted(set(allocations) - set(config["allocations"]))
    unknown_workloads = sorted(set(workload_rates) - set(config["workloads"]))
    if unknown_allocations or unknown_workloads:
        raise ExperimentError(
            f"phase references unknown entries: allocations={unknown_allocations}, workloads={unknown_workloads}"
        )
    if not allocations or not seeds or not workload_rates:
        raise ExperimentError("filters produced an empty experiment plan")
    for workload_name, rates in workload_rates.items():
        if not rates:
            raise ExperimentError(f"workload {workload_name!r} has no selected rates")

    return {
        "name": phase_name,
        "allocations": allocations,
        "seeds": seeds,
        "workload_rates": workload_rates,
    }


def build_sample_plan(
    config: Mapping[str, Any],
    phase: Mapping[str, Any],
) -> list[dict[str, Any]]:
    window_s = float(config["protocol"]["measurement_window_s"])
    samples: list[dict[str, Any]] = []
    for allocation in phase["allocations"]:
        for workload, rates in phase["workload_rates"].items():
            for rate in rates:
                num_prompts_float = float(rate) * window_s
                num_prompts = round(num_prompts_float)
                if not math.isclose(num_prompts, num_prompts_float, abs_tol=1e-9):
                    raise ExperimentError(f"rate {rate} x window {window_s} is not an integer")
                for seed in phase["seeds"]:
                    sample_id = f"{allocation}__{workload}__r{format_rate(rate)}__s{seed}"
                    samples.append(
                        {
                            "sample_id": sample_id,
                            "allocation": allocation,
                            "workload": workload,
                            "request_rate": float(rate),
                            "seed": int(seed),
                            "num_prompts": int(num_prompts),
                        }
                    )
    return samples


def build_server_command(
    config: Mapping[str, Any],
    allocation_name: str,
    repo_root: Path,
) -> list[str]:
    environment = config["environment"]
    server = config["server"]
    allocation = config["allocations"][allocation_name]
    executable = str(resolve_path(environment["vllm_executable"], repo_root))
    model = str(resolve_path(environment["model_path"], repo_root))
    command = [
        executable,
        "serve",
        model,
        "--host",
        str(server["host"]),
        "--port",
        str(server["port"]),
    ]
    command.extend(str(arg) for arg in server.get("args", []))
    command.extend(str(arg) for arg in allocation.get("server_args", []))
    return command


def build_benchmark_command(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    repo_root: Path,
    attempt_id: str,
    result_dir: Path,
    contract_sha256: str,
    git_commit: str,
    vllm_source_commit: str | None,
) -> list[str]:
    environment = config["environment"]
    server = config["server"]
    protocol = config["protocol"]
    workload = config["workloads"][sample["workload"]]
    executable = str(resolve_path(environment["vllm_executable"], repo_root))
    model = str(resolve_path(environment["model_path"], repo_root))
    max_ttft = max(float(value) for value in protocol["ttft_thresholds_ms"])

    command = [
        executable,
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://{server['host']}:{server['port']}",
        "--model",
        model,
        "--dataset-name",
        str(workload["dataset_name"]),
        "--num-prompts",
        str(sample["num_prompts"]),
        "--num-warmups",
        str(protocol["warmup_requests"]),
        "--request-rate",
        f"{sample['request_rate']:g}",
        "--seed",
        str(sample["seed"]),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        "result.json",
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--metric-percentiles",
        "50,95,99",
        "--goodput",
        f"ttft:{max_ttft:g}",
        f"tpot:{float(protocol['tpot_threshold_ms']):g}",
        "--temperature",
        "0",
        "--disable-tqdm",
        "--request-id-prefix",
        f"{sample['sample_id']}-",
    ]
    max_concurrency = protocol.get("max_concurrency")
    if max_concurrency is not None:
        command.extend(["--max-concurrency", str(max_concurrency)])

    if workload.get("dataset_path"):
        dataset_path = resolve_path(workload["dataset_path"], repo_root)
        command.extend(["--dataset-path", str(dataset_path)])
    if workload.get("random_input_len") is not None:
        command.extend(["--random-input-len", str(workload["random_input_len"])])
    if workload.get("random_output_len") is not None:
        command.extend(["--random-output-len", str(workload["random_output_len"])])
    if workload.get("sharegpt_output_len") is not None:
        command.extend(["--sharegpt-output-len", str(workload["sharegpt_output_len"])])
    if workload.get("ignore_eos", False):
        command.append("--ignore-eos")
    if workload.get("no_oversample", False):
        command.append("--no-oversample")

    metadata = {
        "attempt_id": attempt_id,
        "sample_id": sample["sample_id"],
        "allocation": sample["allocation"],
        "workload": sample["workload"],
        "seed": sample["seed"],
        "offered_rate": f"{sample['request_rate']:g}",
        "measurement_window_s": f"{protocol['measurement_window_s']:g}",
        "contract_sha256": contract_sha256,
        "git_commit": git_commit,
        "vllm_source_commit": vllm_source_commit or "unknown",
    }
    command.append("--metadata")
    command.extend(f"{key}={value}" for key, value in metadata.items())
    return command


def per_request_tpot_ms(result: Mapping[str, Any]) -> list[float]:
    output_lens = result["output_lens"]
    itls = result["itls"]
    tpots: list[float] = []
    for output_len, request_itls in zip(output_lens, itls):
        output_len = int(output_len)
        if output_len <= 1:
            tpots.append(0.0)
        else:
            tpots.append(1000.0 * sum(float(value) for value in request_itls) / (output_len - 1))
    return tpots


def analyze_result(
    result: Mapping[str, Any],
    sample: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "duration",
        "completed",
        "failed",
        "request_throughput",
        "request_goodput",
        "p99_ttft_ms",
        "p99_tpot_ms",
        "ttfts",
        "itls",
        "output_lens",
        "start_times",
        "errors",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise ExperimentError(f"result missing required fields: {', '.join(missing)}")

    expected = int(sample["num_prompts"])
    completed = int(result["completed"])
    failed = int(result["failed"])
    if completed + failed != expected:
        raise ExperimentError(
            "denominator mismatch: "
            f"expected={expected}, completed={completed}, failed={failed}"
        )
    if completed <= 0:
        raise ExperimentError("result has no successful requests for latency analysis")

    detail_fields = ("ttfts", "itls", "output_lens", "start_times", "errors")
    for field in detail_fields:
        if len(result[field]) != expected:
            raise ExperimentError(f"detailed field {field!r} has {len(result[field])} rows, expected {expected}")
    errors = list(result["errors"])
    failed_indices = [index for index, error in enumerate(errors) if error]
    if len(failed_indices) != failed:
        raise ExperimentError(
            "failed request accounting mismatch: "
            f"reported={failed}, errors={len(failed_indices)}"
        )
    success_mask = [not error for error in errors]
    if sum(success_mask) != completed:
        raise ExperimentError(
            "successful request accounting mismatch: "
            f"reported={completed}, detailed={sum(success_mask)}"
        )

    duration_s = ensure_finite("duration", result["duration"])
    offered_rate = float(sample["request_rate"])
    window_s = float(protocol["measurement_window_s"])
    if duration_s <= 0:
        raise ExperimentError(f"invalid duration: {duration_s}")

    all_ttfts_ms = [1000.0 * ensure_finite("ttft", value) for value in result["ttfts"]]
    all_tpots_ms = per_request_tpot_ms(result)
    ttfts_ms = [value for value, success in zip(all_ttfts_ms, success_mask) if success]
    tpots_ms = [value for value, success in zip(all_tpots_ms, success_mask) if success]
    start_times = [ensure_finite("start_time", value) for value in result["start_times"]]
    arrival_span_s = max(start_times) - min(start_times) if len(start_times) > 1 else 0.0
    arrival_ratio = arrival_span_s / window_s if window_s else 0.0
    tolerance = float(protocol["arrival_window_tolerance_fraction"])
    if not (1.0 - tolerance <= arrival_ratio <= 1.0 + tolerance):
        raise ExperimentError(
            f"arrival window drift: observed={arrival_span_s:.3f}s, target={window_s:.3f}s, tolerance={tolerance:.3f}"
        )

    request_throughput = ensure_finite("request_throughput", result["request_throughput"])
    delivery_ratio = request_throughput / offered_rate
    tpot_threshold = float(protocol["tpot_threshold_ms"])
    sustainable_ratio = float(protocol["sustainable_goodput_ratio"])
    sweep: dict[str, Any] = {}
    for threshold in protocol["ttft_thresholds_ms"]:
        threshold_value = float(threshold)
        good_count = sum(
            1
            for success, ttft, tpot in zip(success_mask, all_ttfts_ms, all_tpots_ms)
            if success and ttft <= threshold_value and tpot <= tpot_threshold
        )
        goodput = good_count / duration_s
        goodput_ratio = goodput / offered_rate
        sweep[f"{threshold_value:g}"] = {
            "ttft_threshold_ms": threshold_value,
            "tpot_threshold_ms": tpot_threshold,
            "good_requests": good_count,
            "goodput_req_s": goodput,
            "goodput_over_offered": goodput_ratio,
            "sustainable": goodput_ratio >= sustainable_ratio,
        }

    max_threshold_key = f"{max(float(v) for v in protocol['ttft_thresholds_ms']):g}"
    builtin_goodput = ensure_finite("request_goodput", result["request_goodput"])
    recomputed_goodput = sweep[max_threshold_key]["goodput_req_s"]
    if abs(builtin_goodput - recomputed_goodput) > float(protocol["goodput_crosscheck_abs_tolerance"]):
        raise ExperimentError(
            f"vLLM goodput cross-check failed: builtin={builtin_goodput:.6f}, recomputed={recomputed_goodput:.6f}"
        )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "sample_id": sample["sample_id"],
        "status": VALID_SAMPLE_STATUS,
        "request_failure_policy": REQUEST_FAILURE_POLICY,
        "completed": completed,
        "failed": failed,
        "failed_request_fraction": failed / expected,
        "failed_request_indices": failed_indices,
        "offered_rate_req_s": offered_rate,
        "measurement_window_s": window_s,
        "observed_arrival_span_s": arrival_span_s,
        "arrival_span_over_target": arrival_ratio,
        "benchmark_duration_s": duration_s,
        "drain_after_arrival_window_s": max(0.0, duration_s - window_s),
        "request_throughput_req_s": request_throughput,
        "request_throughput_over_offered": delivery_ratio,
        "ttft_p95_ms_recomputed": percentile(ttfts_ms, 95),
        "tpot_p95_ms_recomputed": percentile(tpots_ms, 95),
        "ttft_p99_ms_recomputed": percentile(ttfts_ms, 99),
        "tpot_p99_ms_recomputed": percentile(tpots_ms, 99),
        "reported_ttft_p99_ms": ensure_finite("p99_ttft_ms", result["p99_ttft_ms"]),
        "reported_tpot_p99_ms": ensure_finite("p99_tpot_ms", result["p99_tpot_ms"]),
        "slo_sweep": sweep,
    }


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def terminate_process_group(process: subprocess.Popen[Any], grace_s: float) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":
        process.kill()
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30)


class ServerSession:
    def __init__(
        self,
        config: Mapping[str, Any],
        allocation_name: str,
        repo_root: Path,
        attempt_dir: Path,
    ) -> None:
        self.config = config
        self.allocation_name = allocation_name
        self.repo_root = repo_root
        self.attempt_dir = attempt_dir
        self.process: subprocess.Popen[Any] | None = None
        self.log_handle: Any | None = None
        self.session_dir: Path | None = None
        self.started_monotonic: float | None = None
        self.startup_duration_s: float | None = None

    def __enter__(self) -> ServerSession:  # noqa: PYI034
        server = self.config["server"]
        host = str(server["host"])
        port = int(server["port"])
        if port_is_open(host, port):
            raise ExperimentError(f"server port already occupied: {host}:{port}")

        session_id = f"{compact_timestamp()}-{uuid.uuid4().hex[:8]}"
        self.session_dir = self.attempt_dir / "servers" / self.allocation_name / session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        command = build_server_command(self.config, self.allocation_name, self.repo_root)
        write_json_with_hash(
            self.session_dir / "contract.json",
            {
                "schema_version": SCHEMA_VERSION,
                "allocation": self.allocation_name,
                "command": command,
                "cwd": str(self.repo_root),
                "started_at": utc_timestamp(),
            },
        )
        self.log_handle = (self.session_dir / "server.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in self.config["environment"].get("env", {}).items()})
        self.started_monotonic = time.monotonic()
        self.process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        atomic_write_json(
            self.session_dir / "status.json",
            {
                "status": "starting",
                "pid": self.process.pid,
                "updated_at": utc_timestamp(),
            },
        )
        try:
            self._wait_until_ready()
        except Exception:
            terminate_process_group(
                self.process,
                float(self.config["server"].get("shutdown_grace_s", 30)),
            )
            self.log_handle.flush()
            self.log_handle.close()
            atomic_write_json(
                self.session_dir / "status.json",
                {
                    "status": "startup_failed",
                    "returncode": self.process.returncode,
                    "startup_duration_s": time.monotonic() - self.started_monotonic,
                    "updated_at": utc_timestamp(),
                },
            )
            raise
        return self

    def _wait_until_ready(self) -> None:
        assert self.process is not None
        assert self.session_dir is not None
        server = self.config["server"]
        deadline = time.monotonic() + float(server["startup_timeout_s"])
        health_url = f"http://{server['host']}:{server['port']}{server.get('health_path', '/health')}"
        last_error = ""
        while time.monotonic() < deadline:
            returncode = self.process.poll()
            if returncode is not None:
                self.log_handle.flush()
                tail = (self.session_dir / "server.log").read_text(encoding="utf-8", errors="replace")[-8000:]
                raise ExperimentError(f"server exited during startup with rc={returncode}\n{tail}")
            try:
                with urlopen(health_url, timeout=2) as response:
                    if 200 <= response.status < 300:
                        time.sleep(float(server.get("stabilization_s", 0)))
                        self.log_handle.flush()
                        self._verify_log_patterns()
                        assert self.started_monotonic is not None
                        self.startup_duration_s = time.monotonic() - self.started_monotonic
                        atomic_write_json(
                            self.session_dir / "status.json",
                            {
                                "status": "ready",
                                "pid": self.process.pid,
                                "health_url": health_url,
                                "startup_duration_s": self.startup_duration_s,
                                "updated_at": utc_timestamp(),
                            },
                        )
                        return
            except (OSError, URLError) as exc:
                last_error = str(exc)
            time.sleep(float(server.get("health_poll_interval_s", 2)))
        raise ExperimentError(
            f"server startup timed out after {server['startup_timeout_s']}s; last health error: {last_error}"
        )

    def _verify_log_patterns(self) -> None:
        assert self.session_dir is not None
        text = (self.session_dir / "server.log").read_text(encoding="utf-8", errors="replace")
        patterns = self.config["allocations"][self.allocation_name].get("required_log_substrings", [])
        missing = [pattern for pattern in patterns if str(pattern) not in text]
        if missing:
            raise ExperimentError(
                f"server log does not prove allocation {self.allocation_name!r}; missing substrings: {missing}"
            )

    def assert_healthy(self) -> None:
        assert self.process is not None
        assert self.session_dir is not None
        returncode = self.process.poll()
        if returncode is not None:
            self.log_handle.flush()
            tail = (self.session_dir / "server.log").read_text(
                encoding="utf-8",
                errors="replace",
            )[-8000:]
            raise ExperimentError(
                f"server exited during benchmark with rc={returncode}\n{tail}"
            )

        server = self.config["server"]
        health_url = (
            f"http://{server['host']}:{server['port']}"
            f"{server.get('health_path', '/health')}"
        )
        try:
            with urlopen(health_url, timeout=2) as response:
                if not 200 <= response.status < 300:
                    raise ExperimentError(
                        f"server health check failed after benchmark: "
                        f"url={health_url} status={response.status}"
                    )
        except (OSError, URLError) as exc:
            raise ExperimentError(
                f"server health check failed after benchmark: url={health_url} error={exc}"
            ) from exc

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.process is not None:
            terminate_process_group(
                self.process,
                float(self.config["server"].get("shutdown_grace_s", 30)),
            )
        if self.log_handle is not None:
            self.log_handle.flush()
            self.log_handle.close()
        if self.session_dir is not None:
            atomic_write_json(
                self.session_dir / "status.json",
                {
                    "status": "stopped",
                    "returncode": self.process.returncode if self.process else None,
                    "startup_duration_s": self.startup_duration_s,
                    "updated_at": utc_timestamp(),
                    "exception": str(exc) if exc else None,
                },
            )


def load_sample_status(sample_dir: Path) -> dict[str, Any] | None:
    status_path = sample_dir / "status.json"
    if not status_path.exists():
        return None
    with status_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run_sample(
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    repo_root: Path,
    attempt_dir: Path,
    attempt_id: str,
    git_commit: str,
    vllm_source_commit: str | None,
    post_benchmark_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    sample_dir = attempt_dir / "samples" / sample["sample_id"]
    existing = load_sample_status(sample_dir)
    if existing is not None:
        if existing.get("status") == VALID_SAMPLE_STATUS:
            return {"sample_id": sample["sample_id"], "action": "skipped_completed"}
        raise ExperimentError(
            f"sample {sample['sample_id']} already has status "
            f"{existing.get('status')!r}; retry under a new attempt ID with "
            "--parent-attempt"
        )

    sample_dir.mkdir(parents=True, exist_ok=False)
    work_dir = sample_dir / "work"
    work_dir.mkdir()
    contract_base = {
        "schema_version": SCHEMA_VERSION,
        **sample,
        "protocol": config["protocol"],
        "workload_config": config["workloads"][sample["workload"]],
        "allocation_config": config["allocations"][sample["allocation"]],
        "attempt_id": attempt_id,
        "git_commit": git_commit,
        "vllm_source_commit": vllm_source_commit,
    }
    contract_sha = sha256_bytes(canonical_json_bytes(contract_base))
    command = build_benchmark_command(
        config,
        sample,
        repo_root=repo_root,
        attempt_id=attempt_id,
        result_dir=work_dir,
        contract_sha256=contract_sha,
        git_commit=git_commit,
        vllm_source_commit=vllm_source_commit,
    )
    contract = {**contract_base, "contract_sha256": contract_sha, "command": command}
    write_json_with_hash(sample_dir / "contract.json", contract)
    atomic_write_json(
        sample_dir / "status.json",
        {"status": "running", "started_at": utc_timestamp()},
    )

    log_partial = sample_dir / "bench.log.partial"
    started = time.monotonic()
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in config["environment"].get("env", {}).items()})
    timed_out = False
    with log_partial.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=float(config["protocol"]["sample_timeout_s"]))
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_group(process, grace_s=10)
            returncode = process.returncode
    duration_s = time.monotonic() - started
    os.replace(log_partial, sample_dir / "bench.log")

    raw_result = work_dir / "result.json"
    if timed_out or returncode != 0 or not raw_result.exists():
        atomic_write_json(
            sample_dir / "status.json",
            {
                "status": "failed",
                "timed_out": timed_out,
                "returncode": returncode,
                "duration_s": duration_s,
                "finished_at": utc_timestamp(),
            },
        )
        raise ExperimentError(
            f"sample {sample['sample_id']} failed: timeout={timed_out}, "
            f"returncode={returncode}, result_exists={raw_result.exists()}"
        )

    if post_benchmark_check is not None:
        try:
            post_benchmark_check()
        except Exception as exc:
            atomic_write_json(
                sample_dir / "status.json",
                {
                    "status": "failed",
                    "timed_out": False,
                    "returncode": returncode,
                    "duration_s": duration_s,
                    "failure_stage": "server_health_check",
                    "error": str(exc),
                    "finished_at": utc_timestamp(),
                },
            )
            raise

    try:
        with raw_result.open(encoding="utf-8") as handle:
            result = json.load(handle)
        analysis = analyze_result(result, sample, config["protocol"])
    except Exception as exc:
        atomic_write_json(
            sample_dir / "status.json",
            {
                "status": "failed",
                "timed_out": False,
                "returncode": returncode,
                "duration_s": duration_s,
                "failure_stage": "result_validation",
                "error": str(exc),
                "finished_at": utc_timestamp(),
            },
        )
        raise
    final_result = sample_dir / "result.json"
    os.replace(raw_result, final_result)
    result_sha = sha256_file(final_result)
    atomic_write_bytes(final_result.with_suffix(".json.sha256"), f"{result_sha}\n".encode())
    analysis_sha = write_json_with_hash(sample_dir / "analysis.json", analysis)
    if work_dir.exists():
        try:
            work_dir.rmdir()
        except OSError:
            pass
    status = {
        "status": VALID_SAMPLE_STATUS,
        "evidence_status": "UNVERIFIED",
        "returncode": returncode,
        "runner_duration_s": duration_s,
        "result_sha256": result_sha,
        "analysis_sha256": analysis_sha,
        "finished_at": utc_timestamp(),
    }
    atomic_write_json(sample_dir / "status.json", status)
    return {"sample_id": sample["sample_id"], "action": "completed", **status}


def collect_attempt_summary(
    attempt_dir: Path,
    plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for sample in plan:
        status = load_sample_status(attempt_dir / "samples" / sample["sample_id"])
        state = status.get("status") if status else "not_started"
        counts[state] = counts.get(state, 0) + 1
        samples.append({"sample_id": sample["sample_id"], "status": state})
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_timestamp(),
        "counts": counts,
        "samples": samples,
    }


def environment_manifest(
    config: Mapping[str, Any],
    repo_root: Path,
    selected_workloads: Iterable[str],
    root_git: Mapping[str, Any],
) -> dict[str, Any]:
    environment = config["environment"]
    vllm_source = resolve_path(environment["vllm_source_dir"], repo_root)
    vllm_executable = resolve_path(environment["vllm_executable"], repo_root)
    python_executable = vllm_executable.with_name("python")
    model_path = resolve_path(environment["model_path"], repo_root)
    manifest: dict[str, Any] = {
        "captured_at": utc_timestamp(),
        "hostname": run_capture(["hostname"]),
        "root_git": root_git,
        "vllm_source_path": str(vllm_source),
        "vllm_source_commit": get_optional_git_commit(vllm_source),
        "python_version": run_capture([str(python_executable), "--version"]),
        "package_versions": run_capture(
            [
                str(python_executable),
                "-c",
                (
                    "import json, torch, vllm; "
                    "print(json.dumps({'torch': torch.__version__, "
                    "'vllm': vllm.__version__, 'vllm_file': vllm.__file__, "
                    "'cuda': torch.version.cuda}))"
                ),
            ]
        ),
        "nvidia_smi": run_capture(["nvidia-smi", "-q"], allow_failure=True),
        "model_path": str(model_path),
    }
    model_config = model_path / "config.json"
    if model_config.exists():
        manifest["model_config_sha256"] = sha256_file(model_config)
    datasets: dict[str, Any] = {}
    for name in selected_workloads:
        dataset_value = config["workloads"][name].get("dataset_path")
        if not dataset_value:
            continue
        dataset_path = resolve_path(dataset_value, repo_root)
        if not dataset_path.is_file():
            raise ExperimentError(f"dataset does not exist: {dataset_path}")
        datasets[name] = {
            "path": str(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
            "sha256": sha256_file(dataset_path),
        }
    manifest["datasets"] = datasets
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--attempt-id")
    parser.add_argument("--parent-attempt")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allocations")
    parser.add_argument("--workloads")
    parser.add_argument("--seeds")
    parser.add_argument("--rates")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    repo_root = config_path.parents[2]
    phase = resolve_phase(
        config,
        args.phase,
        allocation_filter=parse_csv(args.allocations),
        workload_filter=parse_csv(args.workloads),
        seed_filter=parse_csv(args.seeds, int),
        rate_filter=parse_csv(args.rates, float),
    )
    plan = build_sample_plan(config, phase)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ExperimentError("--max-samples must be positive")

    attempt_id = args.attempt_id or f"{args.phase}-{compact_timestamp()}-{uuid.uuid4().hex[:8]}"
    output_root = (
        args.output_root.resolve() if args.output_root else repo_root / "experiments" / config["name"] / "attempts"
    )
    attempt_dir = output_root / attempt_id
    root_git = get_git_state(repo_root, require_clean=bool(config["protocol"]["require_clean_git"]))
    vllm_source = resolve_path(config["environment"]["vllm_source_dir"], repo_root)
    vllm_source_commit = get_optional_git_commit(vllm_source)
    config_sha = sha256_file(config_path)

    attempt_contract = {
        "schema_version": SCHEMA_VERSION,
        "experiment": config["name"],
        "phase": phase,
        "attempt_id": attempt_id,
        "parent_attempt": args.parent_attempt,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "git_commit": root_git["commit"],
        "vllm_source_commit": vllm_source_commit,
        "plan": plan,
    }
    if args.dry_run:
        print(json.dumps(attempt_contract, indent=2, ensure_ascii=False))
        return 0

    if attempt_dir.exists() and not args.resume:
        raise ExperimentError(
            f"attempt already exists: {attempt_dir}; use --resume only for an unfailed frozen attempt"
        )
    attempt_dir.mkdir(parents=True, exist_ok=True)
    contract_path = attempt_dir / "attempt_contract.json"
    if contract_path.exists():
        with contract_path.open(encoding="utf-8") as handle:
            existing_contract = json.load(handle)
        if existing_contract != attempt_contract:
            raise ExperimentError("resume contract mismatch; code/config/plan changed or CLI filters differ")
    else:
        write_json_with_hash(contract_path, attempt_contract)
    environment_path = attempt_dir / "environment.json"
    if not environment_path.exists():
        manifest = environment_manifest(config, repo_root, phase["workload_rates"].keys(), root_git)
        write_json_with_hash(environment_path, manifest)

    invocation_dir = attempt_dir / "invocations"
    invocation_dir.mkdir(exist_ok=True)
    atomic_write_json(
        invocation_dir / f"{compact_timestamp()}-{uuid.uuid4().hex[:8]}.json",
        {
            "invoked_at": utc_timestamp(),
            "argv": list(sys.argv if argv is None else argv),
            "resume": args.resume,
            "max_samples": args.max_samples,
        },
    )

    pending: list[dict[str, Any]] = []
    for sample in plan:
        status = load_sample_status(attempt_dir / "samples" / sample["sample_id"])
        if status is None:
            pending.append(sample)
        elif status.get("status") != VALID_SAMPLE_STATUS:
            raise ExperimentError(
                f"attempt contains failed/incomplete sample {sample['sample_id']}; "
                "create a new attempt and set --parent-attempt"
            )
    if args.max_samples is not None:
        pending = pending[: args.max_samples]

    print(f"attempt={attempt_id} phase={args.phase} total={len(plan)} pending_this_invocation={len(pending)}")
    results: list[dict[str, Any]] = []
    try:
        for allocation_name in phase["allocations"]:
            allocation_samples = [sample for sample in pending if sample["allocation"] == allocation_name]
            if not allocation_samples:
                continue
            print(f"starting allocation={allocation_name} samples={len(allocation_samples)}")
            with ServerSession(config, allocation_name, repo_root, attempt_dir) as server_session:
                for sample in allocation_samples:
                    print(f"running sample={sample['sample_id']} prompts={sample['num_prompts']}")
                    outcome = run_sample(
                        config,
                        sample,
                        repo_root=repo_root,
                        attempt_dir=attempt_dir,
                        attempt_id=attempt_id,
                        git_commit=root_git["commit"],
                        vllm_source_commit=vllm_source_commit,
                        post_benchmark_check=server_session.assert_healthy,
                    )
                    results.append(outcome)
                    summary = collect_attempt_summary(attempt_dir, plan)
                    write_json_with_hash(attempt_dir / "summary.json", summary)
                    print(f"completed sample={sample['sample_id']} status={outcome['status']}")
    finally:
        summary = collect_attempt_summary(attempt_dir, plan)
        write_json_with_hash(attempt_dir / "summary.json", summary)

    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
