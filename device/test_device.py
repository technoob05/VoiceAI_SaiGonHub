"""test_device.py — integration tests for the device layer.

Scope: everything that can be proven WITHOUT loading 1.4 GB of models. That is
deliberately most of the operational surface, because almost none of the device
layer's risk lives in the models themselves — it lives in what the supervisor
does when a model misbehaves.

Covered here:
  * config validation            (device_config)
  * structured logging           (devlog)
  * the readiness state machine  (Supervisor.probe / health / set_state)
  * degradation policy           (DEGRADATION, demote, tts/corrector/frontend)
  * the concurrency guard        (pipeline_guard: serialization, overlap, Busy)
  * the HTTP surface             (real server, fake pipeline)
  * the manifest verifier's FAILURE branch  (provenance.verify)
  * the watchdog's decision logic (watchdog.decide / backoff_seconds)

Not covered (and why): anything that needs real weights. `python device/
supervisor.py --warm-only` is the test for that, it takes 50-100 s and 1.4 GB,
and it is a rehearsal step in DEMO_RUNBOOK.md rather than a unit test.

    python -m pytest device/test_device.py -q
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

DEVICE_DIR = Path(__file__).resolve().parent
PROJECT = DEVICE_DIR.parent
sys.path.insert(0, str(DEVICE_DIR))

import device_config          # noqa: E402
import devlog                 # noqa: E402
import pipeline_guard         # noqa: E402
import provenance             # noqa: E402
import supervisor as sup_mod  # noqa: E402
import watchdog as wd         # noqa: E402


# ===========================================================================
# fakes
# ===========================================================================
class FakeCfg:
    def __init__(self):
        self.corrector_enabled = True
        self.asr_beam = 1
        self.asr_lang = "vi"


class FakePipe:
    """A FastCascadeS2ST-shaped object with no models.

    It records concurrency: `max_concurrent[stage]` is the highest number of
    threads that were inside that stage at the same time. That is how the
    serialization tests assert a property rather than a timing coincidence.
    """

    def __init__(self, work_s: float = 0.02):
        self.cfg = FakeCfg()
        self.work_s = work_s
        self._asr = object()          # stand-in for a WhisperModel
        self._asr_prewarm_ms = 1.0
        self.fail = {}                # stage -> exception to raise
        self.calls = {}
        self._active = {}
        self.max_concurrent = {}
        self._lock = threading.Lock()

    # -- bookkeeping ------------------------------------------------------
    def _enter(self, stage):
        with self._lock:
            self.calls[stage] = self.calls.get(stage, 0) + 1
            self._active[stage] = self._active.get(stage, 0) + 1
            self.max_concurrent[stage] = max(self.max_concurrent.get(stage, 0),
                                             self._active[stage])

    def _exit(self, stage):
        with self._lock:
            self._active[stage] -= 1

    def _work(self, stage):
        self._enter(stage)
        try:
            time.sleep(self.work_s)
            exc = self.fail.get(stage)
            if exc is not None:
                raise exc
        finally:
            self._exit(stage)

    # -- pipeline surface -------------------------------------------------
    def _ensure_asr(self): self._work("asr")
    def _ensure_mt(self): self._work("mt")
    def _ensure_piper(self): self._work("tts")

    def _ensure_corrector(self):
        self._work("corrector")
        return self

    def prewarm(self): return True
    def correct(self, text): return text

    def transcribe(self, audio, sr):
        self._work("asr")
        return "máy nén khí ở khu vực b đang quá nhiệt"

    def transcribe_raw(self, audio, sr):
        self._work("asr")
        return "raw"

    def finalize_transcript(self, text):
        self._work("corrector")
        return text, {}

    def translate(self, text, target, source="vi"):
        self._work("mt")
        return f"[{source}->{target}] {text}"

    def synthesize(self, text, target):
        import array
        self._work("tts")
        return array.array("f", [0.0] * 16000), 16000

    def run_streaming(self, audio, sr, target):
        self._enter("cascade")
        try:
            time.sleep(self.work_s)
            yield {"stage": "asr_done", "vi_text": "x"}
        finally:
            self._exit("cascade")


import contextlib


@contextlib.contextmanager
def stage_held_by_another_thread(guard, *stages):
    """Hold `stages` from a DIFFERENT thread.

    This matters: the guard uses RLock, so a lock taken on the calling thread is
    re-entrant and would prove nothing. Real contention is always cross-thread —
    a mic turn versus an HTTP request.
    """
    entered = threading.Event()
    release = threading.Event()
    failure = {}

    def holder():
        try:
            held = guard._acquire(stages, timeout_s=10)
        except Exception as e:            # pragma: no cover
            failure["e"] = e
            entered.set()
            return
        entered.set()
        release.wait(30)
        guard._release(held)

    t = threading.Thread(target=holder, name="lock-holder", daemon=True)
    t.start()
    assert entered.wait(10), "holder thread never acquired"
    assert not failure, failure
    try:
        yield
    finally:
        release.set()
        t.join(timeout=10)


def _busy_or_ok(fn, *a, **kw):
    """Run `fn` and report whether the guard refused it."""
    try:
        fn(*a, **kw)
        return "ok"
    except pipeline_guard.Busy:
        return "busy"


def quiet_logger(name="test"):
    return devlog.Logger(name, path=None, console=False, strict=True)


def make_supervisor(fake: FakePipe | None = None, ready: bool = True,
                    frontend_ok: bool = True) -> sup_mod.Supervisor:
    fake = fake or FakePipe()
    s = sup_mod.Supervisor(target="en", log=quiet_logger())
    s.pipe = pipeline_guard.GuardedPipeline(fake, timeout_s=5.0)
    for name in ("asr", "mt", "tts"):
        s.stage[name].update(ready=True, load_ms=1.0, prewarm_ms=1.0)
    s.stage["corrector"].update(ready=True, load_ms=1.0)
    s.stage["frontend"].update(ready=frontend_ok, load_ms=1.0,
                               detail={"vad_ok": frontend_ok, "denoiser_ok": frontend_ok})
    if ready:
        s.set_state("ready", "test")
        s.ready_utc = "1970-01-01T00:00:00Z"
        s.warm_total_ms = 1.0
    return s


# ===========================================================================
# 1. configuration validation
# ===========================================================================
class TestConfigValidation:
    def test_clean_environment_passes(self):
        errors, _ = device_config.validate({})
        assert errors == []

    @pytest.mark.parametrize("value", ["true", "yes", "on", "", "1.0", "TRUE"])
    def test_intbool_rejects_everything_int_would_choke_on(self, value):
        # prototype/pipeline_fast.py does bool(int(os.environ[...])). Any of
        # these raises ValueError from inside an import, ~20 s into a start.
        errors, _ = device_config.validate({"VOICEAI_CACHE": value})
        assert len(errors) == 1 and "VOICEAI_CACHE" in errors[0]

    @pytest.mark.parametrize("value", ["0", "1"])
    def test_intbool_accepts_0_and_1(self, value):
        errors, _ = device_config.validate({"VOICEAI_CACHE": value})
        assert errors == []

    def test_denoiser_is_only_warmed_after_explicit_opt_in(self):
        assert sup_mod.denoise_requested({}) is False
        assert sup_mod.denoise_requested({"VOICEAI_DENOISE": "0"}) is False
        assert sup_mod.denoise_requested({"VOICEAI_DENOISE": "1"}) is True
        assert sup_mod.denoise_requested({"VOICEAI_DENOISE_VAD": "1"}) is True
        assert sup_mod.denoise_requested({"VOICEAI_DENOISE_GTCRN": "1"}) is True

    def test_enum_rejects_unknown_backend(self):
        errors, _ = device_config.validate({"VOICEAI_TTS_BACKEND": "espeak"})
        assert errors and "piper|mms" in errors[0]

    def test_enum_rejects_wrong_case_because_the_code_is_inconsistent(self):
        # pipeline_fast lowercases VOICEAI_TTS_BACKEND; local_tts_piper does NOT
        # lowercase VOICEAI_TTS_PROVIDER. "CPU" therefore silently misbehaves.
        errors, _ = device_config.validate({"VOICEAI_TTS_PROVIDER": "CPU"})
        assert errors and "wrong case" in errors[0]

    def test_port_range(self):
        assert device_config.validate({"VOICEAI_DEVICE_PORT": "70000"})[0]
        assert device_config.validate({"VOICEAI_DEVICE_PORT": "8731"})[0] == []

    def test_float01_range(self):
        assert device_config.validate({"VOICEAI_CACHE_COSINE": "1.5"})[0]
        assert device_config.validate({"VOICEAI_CACHE_COSINE": "0.95"})[0] == []
        assert device_config.validate({"VOICEAI_CACHE_COSINE": "nan"})[0]

    def test_int_range(self):
        assert device_config.validate({"VOICEAI_CT2_INTRA_THREADS": "-1"})[0]
        assert device_config.validate({"VOICEAI_CT2_INTRA_THREADS": "4"})[0] == []

    def test_missing_directory_is_an_error(self, tmp_path):
        bad = str(tmp_path / "nope")
        errors, _ = device_config.validate({"VOICEAI_ASR_VI_DIR": bad})
        assert errors and "does not exist" in errors[0]
        errors, _ = device_config.validate({"VOICEAI_ASR_VI_DIR": str(tmp_path)})
        assert errors == []

    def test_unknown_voiceai_variable_warns_and_suggests(self):
        _, warnings = device_config.validate({"VOICEAI_TTS_BACKENDS": "piper"})
        assert warnings and "VOICEAI_TTS_BACKENDS" in warnings[0]

    def test_cross_variable_warnings(self):
        _, w = device_config.validate({"VOICEAI_TTS_BACKEND": "mms"})
        assert any("Mandarin" in x for x in w)
        _, w = device_config.validate({"VOICEAI_CORRECTOR_LOCAL_ONLY": "0"})
        assert any("network" in x for x in w)
        _, w = device_config.validate({"VOICEAI_KEEPALIVE_S": "2"})
        assert any("contend" in x for x in w)

    def test_require_valid_raises_with_the_variable_named(self):
        with pytest.raises(device_config.ConfigError) as ei:
            device_config.require_valid({"VOICEAI_TM": "maybe"})
        assert "VOICEAI_TM" in str(ei.value)

    def test_require_valid_returns_warnings_and_does_not_raise(self):
        assert device_config.require_valid({"VOICEAI_TTS_BACKEND": "mms"})

    def test_check_models_flags_a_missing_tree(self, tmp_path):
        errors, _ = device_config.validate({}, check_models=True, project=tmp_path)
        assert len(errors) == len(device_config.REQUIRED_MODEL_DIRS)

    def test_every_spec_entry_is_reachable_by_check_var(self):
        for name in device_config.SPEC:
            device_config.check_var(name, "0")   # must not raise on any kind


# ===========================================================================
# 2. structured logging
# ===========================================================================
class TestDevLog:
    def test_unknown_event_is_rejected(self):
        log = quiet_logger()
        with pytest.raises(ValueError):
            log.info("some.event.i.invented")

    def test_jsonl_is_one_parseable_object_per_line(self, tmp_path):
        p = tmp_path / "x.jsonl"
        log = devlog.Logger("t", path=p, console=False)
        log.info(devlog.E_STAGE_OK, "loaded", stage="mt", load_ms=12.5)
        log.warn(devlog.E_STAGE_DEMOTE, "gone", stage="tts")
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
        assert [r["event"] for r in rows] == [devlog.E_STAGE_OK, devlog.E_STAGE_DEMOTE]
        assert rows[0]["stage"] == "mt" and rows[0]["load_ms"] == 12.5
        assert rows[0]["level"] == "info" and rows[1]["level"] == "warn"
        assert rows[0]["run_id"] == rows[1]["run_id"]

    def test_greppable_by_event_and_stage(self, tmp_path):
        p = tmp_path / "x.jsonl"
        log = devlog.Logger("t", path=p, console=False)
        for i in range(5):
            log.info(devlog.E_PROBE_OK, mt_ms=i)
        log.error(devlog.E_PROBE_FAIL, "boom", stage="mt")
        text = p.read_text(encoding="utf-8")
        assert text.count('"event": "probe.ok"') == 5
        assert text.count('"event": "probe.fail"') == 1

    def test_exception_records_traceback(self, tmp_path):
        p = tmp_path / "x.jsonl"
        log = devlog.Logger("t", path=p, console=False)
        try:
            raise RuntimeError("model gone")
        except RuntimeError as e:
            log.exception(devlog.E_STAGE_FAIL, e, stage="asr")
        rec = json.loads(p.read_text(encoding="utf-8").splitlines()[-1])
        assert rec["error_type"] == "RuntimeError"
        assert "RuntimeError: model gone" in rec["traceback"]

    def test_unserialisable_field_does_not_kill_the_log(self, tmp_path):
        p = tmp_path / "x.jsonl"
        log = devlog.Logger("t", path=p, console=False)
        log.info(devlog.E_READY, obj=object())
        assert json.loads(p.read_text(encoding="utf-8").splitlines()[0])["obj"]

    def test_rotation_bounds_the_file(self, tmp_path):
        p = tmp_path / "x.jsonl"
        log = devlog.Logger("t", path=p, console=False, max_bytes=2000, backups=2)
        for i in range(200):
            log.info(devlog.E_KEEPALIVE_TICK, "tick" * 10, i=i)
        assert p.stat().st_size < 4000
        assert (tmp_path / "x.jsonl.1").exists()

    def test_ring_buffer_survives_a_dead_disk(self, tmp_path):
        log = devlog.Logger("t", path=None, console=False, ring=5)
        for i in range(20):
            log.info(devlog.E_PROBE_OK, i=i)
        recent = log.recent(10)
        assert len(recent) == 5 and recent[-1]["i"] == 19

    def test_logging_is_thread_safe(self, tmp_path):
        p = tmp_path / "x.jsonl"
        log = devlog.Logger("t", path=p, console=False)

        def spam():
            for i in range(50):
                log.info(devlog.E_PROBE_OK, i=i)

        threads = [threading.Thread(target=spam) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        lines = p.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 300
        for line in lines:            # no interleaved/torn writes
            json.loads(line)


# ===========================================================================
# 3. readiness state machine
# ===========================================================================
class TestHealthStateMachine:
    def test_ready_requires_both_state_and_required_stages(self):
        s = make_supervisor()
        assert s.health()["ready"] is True
        s.stage["mt"]["ready"] = False
        assert s.health()["ready"] is False

    def test_a_partially_warmed_device_is_not_ready(self):
        s = make_supervisor(ready=False)
        s.stage["tts"]["ready"] = False
        assert s.state == "starting"
        assert s.health()["ready"] is False

    def test_probe_success_keeps_ready_and_records_timings(self):
        s = make_supervisor()
        r = s.probe("light")
        assert r["ok"] and r["mt_ms"] is not None and r["tts_ms"] is not None
        assert s.health()["ready"] is True
        assert s.counters["probes"] == 1
        assert s.last["probe_utc"]

    def test_full_probe_needs_the_pinned_wav(self, monkeypatch):
        s = make_supervisor()
        monkeypatch.setattr(sup_mod, "PROBE_WAV", Path("does/not/exist.wav"))
        r = s.probe("full")
        assert not r["ok"] and "probe wav missing" in r["detail"]

    def test_healthz_stops_claiming_ready_after_repeated_real_failures(self):
        """The failure this whole layer exists to prevent: a live process with a
        dead model whose health endpoint still says 'ready'."""
        fake = FakePipe()
        s = make_supervisor(fake)
        fake.fail["mt"] = RuntimeError("CT2 translator died")
        for i in range(sup_mod.FAULT_AFTER_FAILED_PROBES - 1):
            s.probe("light")
            assert s.health()["ready"] is True, "must tolerate transients"
        s.probe("light")
        h = s.health()
        assert h["ready"] is False and h["state"] == "faulted"
        assert "CT2 translator died" in h["fault_reason"]

    def test_recovery_restores_ready_without_a_restart(self):
        fake = FakePipe()
        s = make_supervisor(fake)
        fake.fail["mt"] = RuntimeError("transient")
        for _ in range(sup_mod.FAULT_AFTER_FAILED_PROBES):
            s.probe("light")
        assert s.state == "faulted"
        fake.fail.pop("mt")
        assert s.probe("light")["ok"]
        assert s.state == "ready" and s.health()["ready"] is True

    def test_busy_never_faults_the_device(self):
        """Contention is not a fault. If busy could fault the device, a judge
        curling the endpoint during a long utterance would trigger a restart."""
        s = make_supervisor()
        with stage_held_by_another_thread(s.pipe, "mt"):
            for _ in range(10):
                r = s.probe("light", timeout_s=0.05)
                assert r["busy"] and not r["ok"]
        assert s.state == "ready"
        assert s.counters["consecutive_probe_failures"] == 0
        assert s.counters["busy_rejections"] == 10

    def test_probe_never_raises_even_with_a_broken_pipeline(self):
        s = make_supervisor()
        s.pipe = None
        r = s.probe("light")
        assert r["ok"] is False and r["total_ms"] is not None

    def test_metrics_text_is_wellformed(self):
        s = make_supervisor()
        s.probe("light")
        text = s.metrics_text()
        for line in text.strip().splitlines():
            name, _, value = line.rpartition(" ")
            assert name.startswith("voiceai_")
            float(value)          # every metric value must be numeric


# ===========================================================================
# 4. degradation
# ===========================================================================
class TestDegradation:
    def test_policy_table_covers_every_stage_exactly(self):
        assert set(sup_mod.DEGRADATION) == set(sup_mod.Supervisor.STAGES)
        for name, p in sup_mod.DEGRADATION.items():
            assert p["startup"] in ("fatal", "degrade")
            assert p["runtime"] in ("fault", "demote")
            if p["runtime"] == "demote":
                assert p["demote_after"] >= 1, name

    def test_required_flag_follows_the_policy_table(self):
        s = make_supervisor()
        for name, st in s.stage.items():
            assert st["required"] == (sup_mod.DEGRADATION[name]["startup"] == "fatal")

    def test_tts_demotes_after_three_failures_and_the_device_keeps_translating(self):
        fake = FakePipe()
        s = make_supervisor(fake)
        fake.fail["tts"] = RuntimeError("sherpa-onnx session gone")
        for i in range(2):
            r = s.probe("light")
            assert not r["ok"], "before demotion a TTS failure is a probe failure"
            assert not s.stage["tts"]["demoted"]
        s.probe("light")
        assert s.stage["tts"]["demoted"] is True
        # And now the device is healthy again, without audio.
        r = s.probe("light")
        assert r["ok"] is True and r["tts_ms"] is None
        assert "tts" in r["degraded"]
        assert s.health()["ready"] is True
        assert s.health()["degraded_consequences"]["tts"].startswith("text-only")

    def test_a_demoted_stage_is_not_probed_again(self):
        fake = FakePipe()
        s = make_supervisor(fake)
        s.demote("tts", "test")
        before = fake.calls.get("tts", 0)
        s.probe("light")
        assert fake.calls.get("tts", 0) == before

    def test_corrector_demotes_on_the_first_failure(self):
        fake = FakePipe()
        s = make_supervisor(fake)
        assert s.stage_failed("corrector", "boom") == "demote"
        assert s.stage["corrector"]["demoted"]
        # ... and the pipeline is told to stop using it, not just marked.
        assert s.pipe.cfg.corrector_enabled is False

    def test_asr_and_mt_fault_rather_than_demote(self):
        s = make_supervisor()
        assert s.stage_failed("asr", "boom") == "fault"
        assert s.stage_failed("mt", "boom") == "fault"
        assert not s.stage["asr"]["demoted"] and not s.stage["mt"]["demoted"]

    def test_frontend_failure_does_not_stop_http_translation(self):
        s = make_supervisor(frontend_ok=False)
        ok, why = s.mic_available()
        assert ok is False and "VAD" in why
        assert s.health()["ready"] is True          # HTTP path is unaffected
        assert s.probe("light")["ok"] is True

    def test_mic_refuses_to_start_without_a_vad(self):
        s = make_supervisor(frontend_ok=False)
        with pytest.raises(RuntimeError, match="microphone unavailable"):
            s.start_mic()

    def test_denoiser_alone_may_fail_and_the_mic_still_runs(self):
        s = make_supervisor()
        s.stage["frontend"]["detail"] = {"vad_ok": True, "denoiser_ok": False}
        ok, _ = s.mic_available()
        assert ok is True

    def test_no_frontend_flag_disables_the_mic_explicitly(self):
        s = make_supervisor()
        s.warm_frontend = False
        ok, why = s.mic_available()
        assert ok is False and "--no-frontend" in why

    def test_stage_recovery_clears_the_failure_counter(self):
        fake = FakePipe()
        s = make_supervisor(fake)
        fake.fail["mt"] = RuntimeError("blip")
        s.probe("light")
        assert s.stage["mt"]["runtime_failures"] == 1
        fake.fail.pop("mt")
        s.probe("light")
        assert s.stage["mt"]["runtime_failures"] == 0

    def test_every_degraded_stage_states_its_consequence(self):
        for name in sup_mod.Supervisor.STAGES:
            assert sup_mod.Supervisor.demotion_consequence(name) != "unknown"


# ===========================================================================
# 5. the concurrency guard
# ===========================================================================
class TestConcurrencyGuard:
    def test_the_unguarded_pipeline_really_does_run_concurrently(self):
        """Control experiment. Without this the serialization test below could
        pass because the fake is slow, not because the guard works."""
        fake = FakePipe(work_s=0.05)
        threads = [threading.Thread(target=fake.translate, args=("x", "en"))
                   for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert fake.max_concurrent["mt"] > 1

    def test_the_same_stage_is_serialized(self):
        fake = FakePipe(work_s=0.05)
        g = pipeline_guard.GuardedPipeline(fake, timeout_s=30)
        threads = [threading.Thread(target=g.translate, args=("x", "en"))
                   for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert fake.max_concurrent["mt"] == 1
        assert fake.calls["mt"] == 4

    def test_asr_is_serialized_against_a_concurrent_http_request(self):
        """The demo scenario: a judge curls /translate while the mic is live."""
        fake = FakePipe(work_s=0.05)
        g = pipeline_guard.GuardedPipeline(fake, timeout_s=30)
        threads = ([threading.Thread(target=g.transcribe, args=(None, 16000))
                    for _ in range(3)]
                   + [threading.Thread(target=g.translate, args=("x", "en"))
                      for _ in range(3)])
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert fake.max_concurrent["asr"] == 1
        assert fake.max_concurrent["mt"] == 1

    def test_different_stages_still_overlap(self):
        """The cost control: a single global lock would serialize MT against TTS
        and destroy the measured 8.9x first-audio win from streaming TTS."""
        fake = FakePipe(work_s=0.15)
        g = pipeline_guard.GuardedPipeline(fake, timeout_s=30)
        done = threading.Event()

        def mt():
            for _ in range(3):
                g.translate("x", "en")
            done.set()

        t = threading.Thread(target=mt)
        t.start()
        while not done.is_set():
            g.synthesize("y", "en")
        t.join()
        assert max(fake._active.values(), default=0) == 0
        # both stages ran; neither ever doubled up
        assert fake.max_concurrent["mt"] == 1 and fake.max_concurrent["tts"] == 1
        assert fake.calls["mt"] >= 3 and fake.calls["tts"] >= 1

    def test_a_blocked_caller_gives_up_instead_of_hanging(self):
        fake = FakePipe()
        g = pipeline_guard.GuardedPipeline(fake, timeout_s=0.05)
        with stage_held_by_another_thread(g, "mt"):
            t0 = time.monotonic()
            with pytest.raises(pipeline_guard.Busy) as ei:
                g.translate("x", "en")
            assert time.monotonic() - t0 < 2.0
            assert ei.value.stage == "mt"
        assert g.stats()["mt"]["timeouts"] == 1

    def test_a_timeout_releases_the_locks_it_had_already_taken(self):
        """transcribe() needs asr+corrector. If corrector times out, asr must not
        stay locked or the device wedges permanently."""
        fake = FakePipe()
        g = pipeline_guard.GuardedPipeline(fake, timeout_s=0.05)
        with stage_held_by_another_thread(g, "corrector"):
            with pytest.raises(pipeline_guard.Busy):
                g.transcribe(None, 16000)
        # asr must be free again, checked from a thread that never held it
        freed = []
        t = threading.Thread(target=lambda: freed.append(
            g._locks["asr"].acquire(timeout=1.0)))
        t.start()
        t.join()
        assert freed == [True]

    def test_deadline_override_is_per_thread(self):
        fake = FakePipe()
        g = pipeline_guard.GuardedPipeline(fake, timeout_s=30)
        other = {}

        def blocked():
            other["result"] = _busy_or_ok(g.translate, "x", "en")

        t = threading.Thread(target=blocked)
        with stage_held_by_another_thread(g, "mt"):
            with g.deadline(0.05):
                # this thread asked for a 0.05 s budget and must give up
                assert _busy_or_ok(g.translate, "x", "en") == "busy"
            t.start()                     # this one keeps the default 30 s budget
            time.sleep(0.4)
            assert "result" not in other, "the other thread must still be waiting"
        t.join(timeout=10)
        assert other["result"] == "ok"

    def test_whole_cascade_entry_points_hold_everything(self):
        fake = FakePipe(work_s=0.2)
        g = pipeline_guard.GuardedPipeline(fake, timeout_s=0.05)
        gen = g.run_streaming(None, 16000, "en")
        next(gen)                                  # enters the generator, takes locks
        out = []
        try:
            t = threading.Thread(target=lambda: out.append(
                _busy_or_ok(g.translate, "x", "en")))
            t.start()
            t.join(timeout=10)
            assert out == ["busy"]
        finally:
            gen.close()
        t = threading.Thread(target=lambda: out.append(
            _busy_or_ok(g.translate, "x", "en")))
        t.start()
        t.join(timeout=10)
        assert out[-1] == "ok"                     # released on close

    def test_the_asr_model_view_holds_the_lock_across_the_generator(self):
        """faster-whisper's transcribe() returns a LAZY generator; the decode
        happens during iteration. A lock released at return would guard nothing."""
        class LazyModel:
            def transcribe(self, *a, **kw):
                def gen():
                    yield "seg1"
                    yield "seg2"
                return gen(), {"language": "vi"}

        fake = FakePipe()
        fake._asr = LazyModel()
        g = pipeline_guard.GuardedPipeline(fake, timeout_s=0.05)
        segs, info = g._asr.transcribe(None)
        assert info["language"] == "vi"
        next(segs)
        out = []
        t = threading.Thread(target=lambda: out.append(
            _busy_or_ok(g.transcribe, None, 16000)))
        t.start()
        t.join(timeout=10)
        assert out == ["busy"], "asr must still be held by the live generator"
        list(segs)                                  # drain -> releases
        t = threading.Thread(target=lambda: out.append(
            _busy_or_ok(g.transcribe, None, 16000)))
        t.start()
        t.join(timeout=10)
        assert out[-1] == "ok"

    def test_attribute_passthrough_and_raw_escape_hatch(self):
        fake = FakePipe()
        g = pipeline_guard.GuardedPipeline(fake)
        assert g.cfg is fake.cfg
        assert g.raw is fake
        g.cfg.corrector_enabled = False
        assert fake.cfg.corrector_enabled is False

    def test_stats_report_contention_so_healthz_can_prove_it(self):
        fake = FakePipe(work_s=0.08)
        g = pipeline_guard.GuardedPipeline(fake, timeout_s=30)
        threads = [threading.Thread(target=g.translate, args=("x", "en"))
                   for _ in range(3)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        st = g.stats()["mt"]
        assert st["calls"] == 3 and st["waits"] >= 1 and st["max_wait_ms"] > 0
        assert g.contended() is True


# ===========================================================================
# 6. the HTTP surface (real server, fake pipeline)
# ===========================================================================
@pytest.fixture
def served():
    fake = FakePipe()
    s = make_supervisor(fake)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), sup_mod.make_handler(s))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield s, fake, f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _post(url, payload, timeout=10, raw=None):
    body = raw if raw is not None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}


class TestHttpSurface:
    def test_healthz_200_when_ready_503_when_not(self, served):
        s, _, base = served
        code, body = _get(base + "/healthz")
        assert code == 200 and json.loads(body)["ready"] is True
        s.set_state("faulted", "test")
        code, _ = _get(base + "/healthz")
        assert code == 503

    def test_probe_levels_and_bad_level(self, served):
        _, _, base = served
        assert _get(base + "/probe?level=light")[0] == 200
        assert _get(base + "/probe?level=banana")[0] == 400

    def test_probe_returns_429_when_busy_not_503(self, served):
        """The watchdog restarts on 503 and tolerates 429. Getting this wrong
        means a restart every time someone probes during a mic turn."""
        s, _, base = served
        s.probe_timeout_s = 0.05
        with stage_held_by_another_thread(s.pipe, "mt"):
            code, body = _get(base + "/probe?level=light")
        assert code == 429 and json.loads(body)["busy"] is True

    def test_metrics_and_events(self, served):
        _, _, base = served
        code, text = _get(base + "/metrics")
        assert code == 200 and "voiceai_ready 1" in text
        code, body = _get(base + "/events?n=5")
        assert code == 200 and "events" in json.loads(body)

    def test_unknown_path_lists_the_endpoints(self, served):
        _, _, base = served
        code, body = _get(base + "/nope")
        assert code == 404 and "/healthz" in body

    def test_translate_happy_path(self, served):
        _, _, base = served
        code, body = _post(base + "/translate", {"text": "xin chào", "target": "en"})
        assert code == 200
        assert body["translation"] == "[vi->en] xin chào"
        assert body["audio_s"] == 1.0 and body["tts_error"] is None

    @pytest.mark.parametrize("payload,expect", [
        ({"text": ""}, 400),
        ({"text": "x", "target": "klingon"}, 400),
        ({"text": "x" * (sup_mod.MAX_TEXT_CHARS + 1)}, 413),
    ])
    def test_translate_rejects_bad_input_with_4xx_not_500(self, served, payload, expect):
        _, _, base = served
        assert _post(base + "/translate", payload)[0] == expect

    def test_translate_rejects_a_non_object_body(self, served):
        _, _, base = served
        assert _post(base + "/translate", None, raw=b'["a"]')[0] == 400

    def test_translate_rejects_malformed_json_with_a_hint(self, served):
        _, _, base = served
        code, body = _post(base + "/translate", None, raw=b"\xff\xfe not json")
        assert code == 400 and "hint" in body

    def test_translate_rejects_an_oversized_body_without_reading_it(self, served):
        _, _, base = served
        big = b'{"text": "' + b"x" * (sup_mod.MAX_BODY_BYTES + 100) + b'"}'
        assert _post(base + "/translate", None, raw=big)[0] == 413

    def test_translate_survives_a_dead_tts_and_still_returns_text(self, served):
        s, fake, base = served
        fake.fail["tts"] = RuntimeError("voice evicted")
        code, body = _post(base + "/translate", {"text": "x", "target": "en"})
        assert code == 200
        assert body["translation"] and body["audio_s"] is None
        assert "voice evicted" in body["tts_error"]

    def test_translate_503s_when_mt_is_not_ready(self, served):
        s, _, base = served
        s.stage["mt"]["ready"] = False
        assert _post(base + "/translate", {"text": "x"})[0] == 503

    def test_translate_returns_429_not_500_when_busy(self, served):
        s, _, base = served
        object.__setattr__(s.pipe, "_timeout_s", 0.05)
        with stage_held_by_another_thread(s.pipe, "mt"):
            code, body = _post(base + "/translate", {"text": "x"})
        assert code == 429 and body["busy"] is True

    def test_a_handler_exception_returns_500_and_the_server_survives(self, served):
        s, _, base = served
        broken = object()
        original = s.health
        s.health = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            code, _ = _get(base + "/healthz")
            assert code == 500
        finally:
            s.health = original
        assert _get(base + "/probe?level=light")[0] == 200   # still serving
        assert broken is not None


# ===========================================================================
# 7. manifest verification — the FAILURE branch
# ===========================================================================
class TestProvenanceFailureBranch:
    @staticmethod
    def _manifest(tmp_path, assets):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"manifest_version": 1, "assets": assets}),
                     encoding="utf-8")
        return p

    def test_no_manifest_is_exit_6(self, tmp_path, monkeypatch):
        monkeypatch.setattr(provenance, "MANIFEST", tmp_path / "absent.json")
        assert provenance.verify() == 6

    def test_a_good_manifest_passes(self, tmp_path, monkeypatch):
        f = tmp_path / "model.bin"
        f.write_bytes(b"weights")
        monkeypatch.setattr(provenance, "PROJECT", tmp_path)
        m = self._manifest(tmp_path, [{
            "stage": "mt", "path": "model.bin", "bytes": 7,
            "sha256": provenance.sha256_file(f), "required": True}])
        monkeypatch.setattr(provenance, "MANIFEST", m)
        assert provenance.verify() == 0

    def test_a_truncated_required_file_fails(self, tmp_path, monkeypatch, capsys):
        f = tmp_path / "model.bin"
        f.write_bytes(b"trunc")
        monkeypatch.setattr(provenance, "PROJECT", tmp_path)
        m = self._manifest(tmp_path, [{
            "stage": "mt", "path": "model.bin", "bytes": 999,
            "sha256": "0" * 64, "required": True}])
        monkeypatch.setattr(provenance, "MANIFEST", m)
        assert provenance.verify() == 5
        assert "SIZE" in capsys.readouterr().err

    def test_a_corrupt_required_file_fails_on_hash_not_size(self, tmp_path,
                                                            monkeypatch, capsys):
        f = tmp_path / "model.bin"
        f.write_bytes(b"corrupt")
        monkeypatch.setattr(provenance, "PROJECT", tmp_path)
        m = self._manifest(tmp_path, [{
            "stage": "mt", "path": "model.bin", "bytes": 7,
            "sha256": "0" * 64, "required": True}])
        monkeypatch.setattr(provenance, "MANIFEST", m)
        assert provenance.verify() == 5
        assert "SHA256" in capsys.readouterr().err
        # ... and --quick misses it, which is exactly why --quick is the weaker mode
        assert provenance.verify(quick=True) == 0

    def test_a_missing_required_file_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(provenance, "PROJECT", tmp_path)
        m = self._manifest(tmp_path, [{
            "stage": "asr", "path": "gone.bin", "bytes": 1,
            "sha256": "0" * 64, "required": True}])
        monkeypatch.setattr(provenance, "MANIFEST", m)
        assert provenance.verify() == 5

    def test_a_missing_OPTIONAL_file_only_warns(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(provenance, "PROJECT", tmp_path)
        m = self._manifest(tmp_path, [{
            "stage": "tts", "path": "ko_voice.onnx", "bytes": 1,
            "sha256": "0" * 64, "required": False}])
        monkeypatch.setattr(provenance, "MANIFEST", m)
        assert provenance.verify() == 0
        assert "WARN (optional asset)" in capsys.readouterr().out

    def test_the_real_manifest_lists_the_stages_the_device_loads(self):
        doc = json.loads((DEVICE_DIR / "models.manifest.json").read_text(encoding="utf-8"))
        stages = {a["stage"] for a in doc["assets"]}
        assert {"asr", "mt", "mt_tokenizer", "tts", "frontend"} <= stages


# ===========================================================================
# 8. watchdog decision logic
# ===========================================================================
class TestWatchdogDecision:
    def test_a_healthy_probe_is_ok(self):
        assert wd.decide(200, {"ok": True}, 0, 0, 0, 5)[0] == wd.A_OK

    def test_a_healthy_probe_wins_even_after_the_restart_budget(self):
        assert wd.decide(200, {"ok": True}, 9, 0, 99, 5)[0] == wd.A_OK

    def test_one_failure_waits_two_restarts(self):
        assert wd.decide(503, {"ok": False, "detail": "MT died"}, 0, 0, 0, 5)[0] == wd.A_WAIT
        assert wd.decide(503, {"ok": False, "detail": "MT died"}, 1, 0, 0, 5)[0] == wd.A_RESTART

    def test_unreachable_counts_as_a_real_failure(self):
        assert wd.decide(None, {"error": "ConnectionRefused"}, 1, 0, 0, 5)[0] == wd.A_RESTART

    def test_busy_is_tolerated_because_it_means_the_device_is_working(self):
        for n in range(wd.BUSY_TOLERANCE):
            a, _ = wd.decide(429, {"ok": False, "busy": True}, 0, n, 0, 5)
            assert a == wd.A_TOLERATE
        a, why = wd.decide(429, {"ok": False, "busy": True}, 0, wd.BUSY_TOLERANCE, 0, 5)
        assert a == wd.A_RESTART and "stuck" in why

    def test_a_legacy_busy_detail_string_is_still_recognised(self):
        a, _ = wd.decide(503, {"ok": False, "detail": "busy: inference lock timeout"},
                         0, 0, 0, 5)
        assert a == wd.A_TOLERATE

    def test_a_None_detail_does_not_crash_the_decision(self):
        assert wd.decide(503, {"ok": False, "detail": None}, 1, 0, 0, 5)[0] == wd.A_RESTART

    def test_it_gives_up_instead_of_looping_restarts_forever(self):
        assert wd.decide(503, {"ok": False}, 1, 0, 5, 5)[0] == wd.A_GIVEUP
        assert wd.decide(503, {"ok": False}, 1, 0, 4, 5)[0] == wd.A_RESTART

    def test_max_restarts_minus_one_never_gives_up(self):
        assert wd.decide(503, {"ok": False}, 9, 0, 999, -1)[0] == wd.A_RESTART

    def test_backoff_grows_and_is_capped(self):
        vals = [wd.backoff_seconds(i) for i in range(8)]
        assert vals == sorted(vals)
        assert vals[0] == 5.0 and max(vals) <= 120.0

    def test_fault_threshold_is_configurable(self):
        assert wd.decide(503, {"ok": False}, 0, 0, 0, 5, fault_after=1)[0] == wd.A_RESTART


class TestWatchdogEvidence:
    def test_the_bundle_answers_the_questions_you_cannot_re_ask(self, tmp_path,
                                                                monkeypatch):
        monkeypatch.setattr(wd, "EVIDENCE", tmp_path)
        health = {"state": "faulted", "ready": False, "pid": 1234,
                  "stages": {"mt": {"ready": False, "error": "CT2 died"}},
                  "last_probe": {"failed_stage": "mt"}}
        out = wd.collect_evidence(1, "mt probe failed", 503, health, None)
        text = out.read_text(encoding="utf-8")
        for section in ("reason", "last response body", "memory", "disk",
                        "port owner", "VOICEAI_* environment",
                        "supervisor.stdout.log", "supervisor.stderr.log",
                        "kernel OOM evidence"):
            assert section in text, section
        assert "CT2 died" in text
        doc = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert doc["reason"] == "mt probe failed" and doc["health"]["pid"] == 1234
        assert "voiceai_env" in doc and "disk" in doc

    def test_evidence_survives_a_supervisor_that_is_not_answering(self, tmp_path,
                                                                  monkeypatch):
        monkeypatch.setattr(wd, "EVIDENCE", tmp_path)
        out = wd.collect_evidence(1, "unreachable", None, {"error": "refused"}, None)
        assert out.is_file() and out.stat().st_size > 200


# ===========================================================================
# 9. supervisor argument / startup guards that need no models
# ===========================================================================
class TestStartupGuards:
    def test_check_config_exits_zero_without_loading_anything(self, capsys):
        assert sup_mod.main(["--check-config", "--warm-only"]) == 0
        assert "VOICEAI_TTS_BACKEND" in capsys.readouterr().out

    def test_an_invalid_environment_exits_2_before_any_model_loads(self, monkeypatch):
        monkeypatch.setenv("VOICEAI_CACHE", "true")
        assert sup_mod.main(["--warm-only"]) == 2

    def test_bad_target_is_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            sup_mod.main(["--target", "klingon"])

    def test_a_second_supervisor_refuses_the_port(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        try:
            with pytest.raises(SystemExit) as ei:
                sup_mod.bind_or_die("127.0.0.1", port)
            assert ei.value.code == 3
        finally:
            s.close()
