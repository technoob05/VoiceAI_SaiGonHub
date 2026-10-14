from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DEVICE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEVICE_DIR))

import acceptance


class SequenceFetcher:
    def __init__(self, health_rows, probe_rows):
        self.health_rows = iter(health_rows)
        self.probe_rows = iter(probe_rows)

    def __call__(self, url, _timeout):
        return next(self.health_rows if url.endswith("/healthz") else self.probe_rows)


def config(**updates):
    values = {
        "base_url": "http://127.0.0.1:8731",
        "iterations": 3,
        "interval_s": 0,
        "max_probe_p95_ms": 2000,
        "max_rss_growth_mb": 256,
    }
    values.update(updates)
    return acceptance.AcceptanceConfig(**values)


def healthy_fetcher(latencies=(400, 500, 600), rss=(1000, 1010, 1020), run_ids=None):
    run_ids = run_ids or ("run-a",) * len(latencies)
    health = [
        (200, {
            "ready": True,
            "state": "ready",
            "run_id": run_id,
            "uptime_s": index + 10,
            "rss_mb": memory,
            "rss_peak_mb": memory + 50,
            "degraded": [],
            "keepalive": {"thread_alive": True},
        }, 2.0)
        for index, (memory, run_id) in enumerate(zip(rss, run_ids))
    ]
    probe = [
        (200, {
            "ok": True,
            "total_ms": latency,
            "asr_ms": 0,
            "mt_ms": 200,
            "tts_ms": latency - 200,
            "degraded": [],
        }, latency + 5)
        for latency in latencies
    ]
    return SequenceFetcher(health, probe)


def test_passing_run_writes_stable_json_evidence(tmp_path):
    output = tmp_path / "acceptance.json"
    code, document = acceptance.execute(config(), output, healthy_fetcher(), lambda _: None)

    assert code == acceptance.EXIT_PASS
    assert document["verdict"]["status"] == "passed"
    assert document["summary"]["probe_total_ms"] == {
        "count": 3, "p50": 500.0, "p95": 600.0, "p99": 600.0, "max": 600.0
    }
    assert document["summary"]["rss_mb"]["max_growth_signal"] == 20.0
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema"] == "voiceai.operational_acceptance"
    assert persisted["schema_version"] == 1
    assert len(persisted["samples"]) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_gate_failure_reports_latency_restart_and_memory_reasons(tmp_path):
    fetcher = healthy_fetcher(
        latencies=(400, 500, 2400),
        rss=(1000, 1010, 1300),
        run_ids=("run-a", "run-a", "run-b"),
    )
    code, document = acceptance.execute(config(), tmp_path / "failed.json", fetcher, lambda _: None)
    reasons = {reason["code"] for reason in document["verdict"]["reasons"]}

    assert code == acceptance.EXIT_GATE_FAILED
    assert reasons == {
        "latency_p95_above_max",
        "rss_growth_above_max",
        "supervisor_restarts_above_max",
    }


def test_busy_probe_is_availability_failure_not_infrastructure_failure(tmp_path):
    health = [(200, {"ready": True, "state": "ready", "run_id": "a"}, 1.0)]
    probe = [(429, {"ok": False, "busy": True, "error": "busy"}, 1.0)]
    code, document = acceptance.execute(
        config(iterations=1), tmp_path / "busy.json", SequenceFetcher(health, probe), lambda _: None
    )
    reasons = {reason["code"] for reason in document["verdict"]["reasons"]}

    assert code == acceptance.EXIT_GATE_FAILED
    assert "busy_ratio_above_max" in reasons
    assert "probe_success_ratio_below_min" in reasons
    assert document["summary"]["reachable_http_responses"] == 2


def test_degraded_voice_or_dead_keepalive_cannot_silently_pass(tmp_path):
    health = [(200, {
        "ready": True,
        "state": "ready",
        "run_id": "a",
        "degraded": ["tts"],
        "keepalive": {"thread_alive": False},
    }, 1.0)]
    probe = [(200, {
        "ok": True,
        "total_ms": 400,
        "degraded": ["tts"],
    }, 1.0)]
    code, document = acceptance.execute(
        config(iterations=1), tmp_path / "degraded.json", SequenceFetcher(health, probe), lambda _: None
    )
    reasons = {reason["code"] for reason in document["verdict"]["reasons"]}

    assert code == acceptance.EXIT_GATE_FAILED
    assert reasons == {"degraded_ratio_above_max", "keepalive_not_running"}


def test_no_http_response_has_distinct_infrastructure_exit(tmp_path):
    unavailable = [(None, {"error": "ConnectionRefusedError"}, 1.0)]
    code, document = acceptance.execute(
        config(iterations=1),
        tmp_path / "unreachable.json",
        SequenceFetcher(unavailable, unavailable),
        lambda _: None,
    )

    assert code == acceptance.EXIT_INFRASTRUCTURE
    assert document["verdict"]["status"] == "infrastructure_failure"
    assert "supervisor_unreachable" in {
        reason["code"] for reason in document["verdict"]["reasons"]
    }


def test_interrupt_preserves_partial_evidence(tmp_path):
    calls = 0

    def interrupting_fetcher(url, _timeout):
        nonlocal calls
        calls += 1
        if calls > 2:
            raise KeyboardInterrupt
        if url.endswith("/healthz"):
            return 200, {"ready": True, "state": "ready", "run_id": "a"}, 1.0
        return 200, {"ok": True, "total_ms": 500}, 1.0

    output = tmp_path / "interrupted.json"
    code, document = acceptance.execute(
        config(iterations=3), output, interrupting_fetcher, lambda _: None
    )

    assert code == acceptance.EXIT_INTERRUPTED
    assert document["verdict"]["status"] == "interrupted"
    assert document["summary"]["sample_count"] == 1
    assert output.exists()


@pytest.mark.parametrize("updates", [
    {"iterations": 0},
    {"min_ready_ratio": 1.1},
    {"max_busy_ratio": -0.1},
    {"max_degraded_ratio": 1.1},
    {"max_rss_growth_mb": -1},
    {"probe_level": "status"},
])
def test_invalid_gates_fail_before_sampling(tmp_path, updates):
    with pytest.raises(ValueError):
        acceptance.execute(config(**updates), tmp_path / "never.json")


def test_nearest_rank_is_deterministic_for_small_operational_samples():
    values = [1, 2, 3, 4, 100]
    assert acceptance.nearest_rank(values, 50) == 3
    assert acceptance.nearest_rank(values, 95) == 100
    assert acceptance.nearest_rank([], 95) is None
