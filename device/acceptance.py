"""Bounded operational acceptance and soak evidence for the VoiceAI service.

This runner observes the resident supervisor through its public HTTP surface. It
does not load models, record audio, or infer target-hardware capability. A pass
means only that the observed service met the configured gates for this run.

Exit codes:
    0  all configured gates passed
    2  service responded but one or more gates failed
    3  infrastructure failure (no HTTP response was received)
  130  interrupted; partial evidence was still written
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


SCHEMA_VERSION = 1
EXIT_PASS = 0
EXIT_GATE_FAILED = 2
EXIT_INFRASTRUCTURE = 3
EXIT_INTERRUPTED = 130
DEVICE_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = int(os.environ.get("VOICEAI_DEVICE_PORT", "8731"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def fetch_json(url: str, timeout_s: float) -> tuple[int | None, dict, float]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body, elapsed_ms(started)
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read().decode("utf-8"))
        except Exception:
            body = {"error": str(error)}
        return error.code, body, elapsed_ms(started)
    except Exception as error:
        return None, {"error": f"{type(error).__name__}: {error}"}, elapsed_ms(started)


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)


@dataclass(frozen=True)
class AcceptanceConfig:
    base_url: str
    probe_level: str = "light"
    iterations: int = 10
    interval_s: float = 2.0
    timeout_s: float = 120.0
    max_probe_p95_ms: float = 2000.0
    min_ready_ratio: float = 1.0
    min_probe_success_ratio: float = 1.0
    max_busy_ratio: float = 0.0
    max_degraded_ratio: float = 0.0
    max_rss_growth_mb: float = 256.0
    max_restarts: int = 0

    def validate(self) -> None:
        if self.probe_level not in {"light", "full"}:
            raise ValueError("probe_level must be light or full")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.interval_s < 0 or self.timeout_s <= 0:
            raise ValueError("interval_s must be >= 0 and timeout_s must be > 0")
        for name in (
            "min_ready_ratio",
            "min_probe_success_ratio",
            "max_busy_ratio",
            "max_degraded_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_probe_p95_ms <= 0 or self.max_rss_growth_mb < 0:
            raise ValueError("latency must be > 0 and RSS growth must be >= 0")
        if self.max_restarts < 0:
            raise ValueError("max_restarts must be >= 0")


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _health_sample(code: int | None, body: dict, request_ms: float) -> dict:
    keepalive = body.get("keepalive") if isinstance(body.get("keepalive"), dict) else {}
    return {
        "http_status": code,
        "request_ms": request_ms,
        "ready": bool(code == 200 and body.get("ready") is True),
        "state": body.get("state"),
        "run_id": body.get("run_id"),
        "uptime_s": _finite_number(body.get("uptime_s")),
        "rss_mb": _finite_number(body.get("rss_mb")),
        "rss_peak_mb": _finite_number(body.get("rss_peak_mb")),
        "degraded": body.get("degraded") if isinstance(body.get("degraded"), list) else [],
        "keepalive_thread_alive": keepalive.get("thread_alive"),
        "error": body.get("error"),
    }


def _probe_sample(level: str, code: int | None, body: dict, request_ms: float) -> dict:
    return {
        "level": level,
        "http_status": code,
        "request_ms": request_ms,
        "ok": bool(code == 200 and body.get("ok") is True),
        "busy": bool(code == 429 or body.get("busy") is True),
        "total_ms": _finite_number(body.get("total_ms")),
        "asr_ms": _finite_number(body.get("asr_ms")),
        "mt_ms": _finite_number(body.get("mt_ms")),
        "tts_ms": _finite_number(body.get("tts_ms")),
        "degraded": body.get("degraded") if isinstance(body.get("degraded"), list) else [],
        "detail": body.get("detail"),
        "error": body.get("error"),
    }


def sample_once(config: AcceptanceConfig, fetcher=fetch_json) -> dict:
    health_code, health_body, health_request_ms = fetcher(
        f"{config.base_url}/healthz", config.timeout_s
    )
    probe_code, probe_body, probe_request_ms = fetcher(
        f"{config.base_url}/probe?level={config.probe_level}", config.timeout_s
    )
    return {
        "sampled_utc": utc_now(),
        "health": _health_sample(health_code, health_body, health_request_ms),
        "probe": _probe_sample(
            config.probe_level, probe_code, probe_body, probe_request_ms
        ),
    }


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return round(ordered[rank - 1], 1)


def _ratio(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def summarize(samples: list[dict], config: AcceptanceConfig) -> tuple[dict, list[dict]]:
    total = len(samples)
    ready_count = sum(sample["health"]["ready"] for sample in samples)
    probe_ok_count = sum(sample["probe"]["ok"] for sample in samples)
    busy_count = sum(sample["probe"]["busy"] for sample in samples)
    degraded_count = sum(
        bool(sample["health"]["degraded"] or sample["probe"]["degraded"])
        for sample in samples
    )
    keepalive_dead_count = sum(
        sample["health"]["keepalive_thread_alive"] is False for sample in samples
    )
    reachable_count = sum(
        sample[part]["http_status"] is not None
        for sample in samples
        for part in ("health", "probe")
    )
    latencies = [
        sample["probe"]["total_ms"]
        for sample in samples
        if sample["probe"]["ok"] and sample["probe"]["total_ms"] is not None
    ]
    run_ids = [sample["health"]["run_id"] for sample in samples if sample["health"]["run_id"]]
    restart_count = sum(left != right for left, right in zip(run_ids, run_ids[1:]))
    rss = [sample["health"]["rss_mb"] for sample in samples if sample["health"]["rss_mb"] is not None]
    rss_peak = [
        sample["health"]["rss_peak_mb"]
        for sample in samples
        if sample["health"]["rss_peak_mb"] is not None
    ]
    rss_growth = round(max(0.0, max(rss) - rss[0]), 1) if rss else None
    rss_peak_growth = round(max(0.0, max(rss_peak) - rss_peak[0]), 1) if rss_peak else None
    observed_growth = max(
        value for value in (rss_growth, rss_peak_growth) if value is not None
    ) if rss or rss_peak else None

    summary = {
        "sample_count": total,
        "reachable_http_responses": reachable_count,
        "ready_count": ready_count,
        "ready_ratio": _ratio(ready_count, total),
        "probe_ok_count": probe_ok_count,
        "probe_success_ratio": _ratio(probe_ok_count, total),
        "busy_count": busy_count,
        "busy_ratio": _ratio(busy_count, total),
        "degraded_count": degraded_count,
        "degraded_ratio": _ratio(degraded_count, total),
        "keepalive_dead_count": keepalive_dead_count,
        "supervisor_run_ids": list(dict.fromkeys(run_ids)),
        "restart_count": restart_count,
        "probe_total_ms": {
            "count": len(latencies),
            "p50": nearest_rank(latencies, 50),
            "p95": nearest_rank(latencies, 95),
            "p99": nearest_rank(latencies, 99),
            "max": round(max(latencies), 1) if latencies else None,
        },
        "rss_mb": {
            "initial": rss[0] if rss else None,
            "max_observed": max(rss) if rss else None,
            "growth_from_initial": rss_growth,
            "peak_initial": rss_peak[0] if rss_peak else None,
            "peak_max_observed": max(rss_peak) if rss_peak else None,
            "peak_growth_from_initial": rss_peak_growth,
            "max_growth_signal": observed_growth,
        },
    }

    reasons = []

    def fail(code: str, observed, threshold, relation: str) -> None:
        reasons.append({
            "code": code,
            "observed": observed,
            "threshold": threshold,
            "required_relation": relation,
        })

    if reachable_count == 0:
        fail("supervisor_unreachable", 0, 1, ">=")
    if summary["ready_ratio"] < config.min_ready_ratio:
        fail("ready_ratio_below_min", summary["ready_ratio"], config.min_ready_ratio, ">=")
    if summary["probe_success_ratio"] < config.min_probe_success_ratio:
        fail(
            "probe_success_ratio_below_min",
            summary["probe_success_ratio"],
            config.min_probe_success_ratio,
            ">=",
        )
    if summary["busy_ratio"] > config.max_busy_ratio:
        fail("busy_ratio_above_max", summary["busy_ratio"], config.max_busy_ratio, "<=")
    if summary["degraded_ratio"] > config.max_degraded_ratio:
        fail(
            "degraded_ratio_above_max",
            summary["degraded_ratio"],
            config.max_degraded_ratio,
            "<=",
        )
    if keepalive_dead_count:
        fail("keepalive_not_running", keepalive_dead_count, 0, "==")
    if summary["probe_total_ms"]["p95"] is None:
        fail("insufficient_latency_samples", 0, 1, ">=")
    elif summary["probe_total_ms"]["p95"] > config.max_probe_p95_ms:
        fail(
            "latency_p95_above_max",
            summary["probe_total_ms"]["p95"],
            config.max_probe_p95_ms,
            "<=",
        )
    if observed_growth is not None and observed_growth > config.max_rss_growth_mb:
        fail("rss_growth_above_max", observed_growth, config.max_rss_growth_mb, "<=")
    if restart_count > config.max_restarts:
        fail("supervisor_restarts_above_max", restart_count, config.max_restarts, "<=")
    return summary, reasons


def make_document(config: AcceptanceConfig, samples: list[dict], interrupted=False) -> dict:
    summary, reasons = summarize(samples, config)
    if interrupted:
        reasons.append({
            "code": "run_interrupted",
            "observed": len(samples),
            "threshold": config.iterations,
            "required_relation": "==",
        })
    if interrupted:
        status, exit_code = "interrupted", EXIT_INTERRUPTED
    elif summary["reachable_http_responses"] == 0:
        status, exit_code = "infrastructure_failure", EXIT_INFRASTRUCTURE
    elif reasons:
        status, exit_code = "failed", EXIT_GATE_FAILED
    else:
        status, exit_code = "passed", EXIT_PASS
    return {
        "schema": "voiceai.operational_acceptance",
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "generated_utc": utc_now(),
        "scope": "observed supervisor session; not target-hardware certification",
        "config": asdict(config),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "summary": summary,
        "verdict": {"status": status, "exit_code": exit_code, "reasons": reasons},
        "limitations": [
            "A pass applies only to the sampled process, host, configuration, and interval.",
            "This runner does not validate Snapdragon, QNN, Hexagon, energy, thermals, or audio hardware.",
            "The runner observes service behavior; it does not assess ASR, translation, or TTS quality.",
        ],
        "samples": samples,
    }


def write_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def execute(
    config: AcceptanceConfig,
    output: Path,
    fetcher: Callable = fetch_json,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[int, dict]:
    config.validate()
    samples = []
    interrupted = False
    try:
        for sequence in range(1, config.iterations + 1):
            sample = sample_once(config, fetcher=fetcher)
            sample["sequence"] = sequence
            samples.append(sample)
            if sequence < config.iterations and config.interval_s:
                sleeper(config.interval_s)
    except KeyboardInterrupt:
        interrupted = True
    document = make_document(config, samples, interrupted=interrupted)
    write_atomic(output, document)
    return document["verdict"]["exit_code"], document


def default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEVICE_DIR / "evidence" / f"acceptance-{stamp}.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="run bounded VoiceAI readiness/inference checks and write JSON evidence"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--probe-level", choices=["light", "full"], default="light")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--interval-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-probe-p95-ms", type=float, default=2000.0)
    parser.add_argument("--min-ready-ratio", type=float, default=1.0)
    parser.add_argument("--min-probe-success-ratio", type=float, default=1.0)
    parser.add_argument("--max-busy-ratio", type=float, default=0.0)
    parser.add_argument("--max-degraded-ratio", type=float, default=0.0)
    parser.add_argument("--max-rss-growth-mb", type=float, default=256.0)
    parser.add_argument("--max-restarts", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = AcceptanceConfig(
        base_url=f"http://{args.host}:{args.port}",
        probe_level=args.probe_level,
        iterations=args.iterations,
        interval_s=args.interval_s,
        timeout_s=args.timeout_s,
        max_probe_p95_ms=args.max_probe_p95_ms,
        min_ready_ratio=args.min_ready_ratio,
        min_probe_success_ratio=args.min_probe_success_ratio,
        max_busy_ratio=args.max_busy_ratio,
        max_degraded_ratio=args.max_degraded_ratio,
        max_rss_growth_mb=args.max_rss_growth_mb,
        max_restarts=args.max_restarts,
    )
    output = args.output or default_output()
    try:
        exit_code, document = execute(config, output)
    except ValueError as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return EXIT_GATE_FAILED
    summary = document["summary"]
    verdict = document["verdict"]
    print(
        f"{verdict['status'].upper()} samples={summary['sample_count']} "
        f"ready={summary['ready_ratio']:.3f} probe_ok={summary['probe_success_ratio']:.3f} "
        f"p95={summary['probe_total_ms']['p95']}ms restarts={summary['restart_count']}"
    )
    for reason in verdict["reasons"]:
        print(
            f"  {reason['code']}: observed={reason['observed']} "
            f"required {reason['required_relation']} {reason['threshold']}"
        )
    print(f"evidence={output.resolve()}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
