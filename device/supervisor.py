"""supervisor.py — the resident, always-warm VoiceAI service for the demo device.

The project today has four entry points (desktop_app.py, live_translate.py,
web_demo_edge.py, mic_demo.py). Each of them loads the models again, in its own
process, when a human double-clicks it. On a portable box in a noisy room that
is the single largest stability risk: the first thing a judge triggers pays the
cold start (measured 13.5 s of model load on a T4 kernel, far more on CPU), and
nothing notices if a model dies mid-demo.

This process fixes exactly that:

  * ONE process owns every model. It loads each exactly once and NEVER unloads.
  * Startup order is mandatory and it refuses to start on a resource conflict.
  * Configuration is validated BEFORE any model is touched (device_config.py).
  * Every stage is prewarmed with a REAL inference before "ready" is printed.
  * Every call into the pipeline is serialized per stage (pipeline_guard.py),
    because faster-whisper and the shared NLLB tokenizer are not thread-safe and
    the mic loop, the keepalive thread and HTTP all share one pipeline.
  * A localhost HTTP surface reports per-stage readiness, last-inference
    latency, uptime and RSS — and can be asked to PROBE the pipeline for real,
    because a live process with a dead model is the failure that matters.
  * Readiness is EARNED and can be LOST: repeated real-probe failures move the
    service out of "ready", so /healthz cannot claim health while the pipeline
    is broken.
  * Stages degrade instead of dying where the demo can survive it, and every
    degradation is an explicit, logged, testable policy (DEGRADATION below).

Order rationale (this is our analogue of the AIBOX 733 MB HTP allocation in
references/Summer_School_AIEye/aibox_eyes/run.sh, but the constraint here is
host RAM and CPU threads, not an NPU arena):
  1. ASR   — biggest resident CT2 model in the hot path; if RAM is short we want
             to fail here, before anything else has claimed memory.
  2. corrector — optional; if it cannot load, the pipeline degrades to raw ASR
             (prototype/live_translate.py does the same) and we keep going.
  3. MT    — 622 MB CT2 + tokenizer; the stage with the longest first-call.
  4. TTS   — sherpa-onnx Piper; 2-slot LRU inside the pipeline, so we warm the
             session target and note that switching target languages evicts.
  5. frontend (Silero VAD + GTCRN) — cheapest, and only needed once audio flows.

Endpoints (all on 127.0.0.1, no network exposure):
  GET  /healthz            state, per-stage readiness, uptime, RSS, last probe
  GET  /probe?level=light  REAL MT + TTS inference          (~1 s, default)
  GET  /probe?level=full   REAL ASR + MT + TTS on a pinned wav
  GET  /metrics            same numbers, plain text
  GET  /events?n=200       the last N structured log records (crash triage)
  POST /translate          {"text": "...", "target": "en"} -> real MT (+TTS)

Usage
    python device/supervisor.py                       # verify manifest, load, serve
    python device/supervisor.py --quick-verify        # size-only manifest check
    python device/supervisor.py --skip-verify         # dev only
    python device/supervisor.py --check-config        # validate env and exit

Exit codes: 0 clean | 2 bad args / invalid config | 3 port/instance conflict |
4 stage failed to warm | 5 manifest verification failed
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEVICE_DIR = Path(__file__).resolve().parent
PROJECT = DEVICE_DIR.parent
PROTOTYPE = PROJECT / "prototype"
LOG_DIR = DEVICE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROTOTYPE))
sys.path.insert(0, str(DEVICE_DIR))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import devlog                                  # noqa: E402
import device_config                           # noqa: E402
from lang_router import normalize_pair         # noqa: E402
from pipeline_guard import Busy, GuardedPipeline   # noqa: E402

LOG = devlog.get_logger("supervisor", LOG_DIR / "supervisor.jsonl")

# Pinned probe audio: a real 4.06 s VIVOS test utterance already in the repo.
PROBE_WAV = PROJECT / "data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav"
PROBE_TEXT_VI = "Máy nén khí ở khu vực B đang quá nhiệt."
PROBE_TTS_TEXT = "Ready."
TARGETS = ("en", "zh", "ko", "vi")
DENOISE_ENV_VARS = ("VOICEAI_DENOISE", "VOICEAI_DENOISE_VAD",
                     "VOICEAI_DENOISE_GTCRN")

MAX_BODY_BYTES = 64 * 1024        # /translate payload cap; a demo sentence is <1 KB
MAX_TEXT_CHARS = 2000             # NLLB truncates far below this anyway
HTTP_HANDLER_TIMEOUT_S = 30.0     # a half-open socket must not pin a thread forever


def denoise_requested(env=None) -> bool:
    """Return True only for an explicit 1-valued denoiser opt-in."""
    env = os.environ if env is None else env
    return any(str(env.get(name, "0")).strip() == "1" for name in DENOISE_ENV_VARS)


def content_logging_enabled(env=None) -> bool:
    """Persist transcript text only after an explicit operator opt-in."""
    env = os.environ if env is None else env
    return str(env.get("VOICEAI_LOG_CONTENT", "0")).strip() == "1"


def mic_configuration_error(*, mic: bool, pair, source, multilingual_asr: bool,
                            target: str) -> str | None:
    """Validate pre-ASR routing before the supervisor loads heavy models."""
    if not mic:
        return None
    if source is None:
        return "--mic requires --source so direction is fixed before ASR"
    raw_pair = (tuple(x.strip() for x in pair.replace("/", ",").split(","))
                if isinstance(pair, str) and pair else (pair or ("vi", target)))
    try:
        configured = normalize_pair(raw_pair)
    except (TypeError, ValueError) as exc:
        return f"invalid --pair: {exc}"
    if "vi" not in configured:
        return "submission mic pairs must include Vietnamese"
    if source not in configured:
        return f"--source {source} is not a member of pair {configured[0]},{configured[1]}"
    if source != "vi" and not multilingual_asr:
        return "a non-Vietnamese --source requires --multilingual-asr"
    return None

# --------------------------------------------------------------------------
# Degradation policy — what happens when a stage fails, per stage, per phase.
#
#   startup: "fatal"   refuse to become ready (the device is not a device)
#            "degrade" start anyway, stage marked not-ready
#   runtime: "fault"   the device stops claiming ready; the watchdog restarts it
#            "demote"  disable the stage permanently, keep serving without it
#
# This table is the specification. `test_device.py` asserts against it rather
# than against prose, so the behaviour cannot drift away from this comment.
# --------------------------------------------------------------------------
DEGRADATION: dict[str, dict] = {
    # No ASR = no speech translation. A restart is the only useful response.
    "asr": {"startup": "fatal", "runtime": "fault", "demote_after": None},
    # Optional by design; prototype/live_translate.py already falls back to the
    # raw transcript. One runtime failure is enough to stop paying for it.
    "corrector": {"startup": "degrade", "runtime": "demote", "demote_after": 1},
    # No MT = nothing to say. Restart.
    "mt": {"startup": "fatal", "runtime": "fault", "demote_after": None},
    # A device that shows correct translated TEXT with no audio still wins a
    # demo; one that stops mid-sentence does not. Demote after 3 consecutive
    # failures so a single transient sherpa-onnx error does not silence it.
    "tts": {"startup": "fatal", "runtime": "demote", "demote_after": 3},
    # VAD + denoiser are only needed once audio flows. Without them the mic
    # cannot run, but HTTP translation is untouched — so this is not fatal.
    "frontend": {"startup": "degrade", "runtime": "demote", "demote_after": 1},
}

# Consecutive failed REAL probes before the device stops calling itself ready.
# One failure can be contention; three in a row is a broken pipeline.
FAULT_AFTER_FAILED_PROBES = 3


def rss_mb(peak: bool = False) -> float:
    """Resident set size in MiB (peak=True -> high-water mark).

    Read the peak, not just the current value, on Windows: the OS trims a
    process's working set aggressively while it is idle. A supervisor holding
    1.6 GB of models reported rss=12.3 MB after two minutes of no traffic
    (measured). The pages are still owned, they are just no longer resident —
    which is precisely the residency risk the keepalive probe exists to fight,
    and the reason a naive "rss dropped, the model must be gone" alarm would be
    wrong here.
    """
    try:
        import psutil  # type: ignore
        mi = psutil.Process().memory_info()
        return (getattr(mi, "peak_wset", mi.rss) if peak else mi.rss) / 2 ** 20
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return (c.PeakWorkingSetSize if peak else c.WorkingSetSize) / 2 ** 20
        except Exception:
            return -1.0
    try:
        if peak:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) / 1024
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except Exception:
        return -1.0


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Supervisor:
    STAGES = ["asr", "corrector", "mt", "tts", "frontend"]

    def __init__(self, target: str = "en", warm_frontend: bool = True,
                 strict_frontend: bool = False, log=None):
        self.log = log or LOG
        self.target = target
        self.warm_frontend = warm_frontend
        self.strict_frontend = strict_frontend
        self.log_content = content_logging_enabled()
        self.started_monotonic = time.monotonic()
        self.started_utc = _utc()
        self.state = "starting"
        self.stage: dict[str, dict] = {
            s: {"ready": False,
                "required": DEGRADATION[s]["startup"] == "fatal",
                "load_ms": None, "prewarm_ms": None, "error": None,
                "runtime_failures": 0, "demoted": False, "demoted_reason": None,
                "detail": {}}
            for s in self.STAGES
        }
        self.pipe: GuardedPipeline | None = None
        self._denoiser = None
        self._vad = None
        self.last = {"probe_utc": None, "level": None, "ok": None,
                     "asr_ms": None, "mt_ms": None, "tts_ms": None,
                     "total_ms": None, "detail": None, "degraded": []}
        self._counters_lock = threading.Lock()
        self.counters = {"probes": 0, "probe_failures": 0, "translate_calls": 0,
                         "translate_failures": 0, "busy_rejections": 0,
                         "consecutive_probe_failures": 0, "faults": 0}
        self.mic = {"enabled": False, "running": False, "turns": 0,
                    "last_utc": None, "last_direction": None, "error": None}
        self._mic_thread = None
        self._live = None
        self._keepalive_thread = None
        self.keepalive_last_utc = None
        self.ready_utc = None
        self.warm_total_ms = None
        self.config_warnings: list[str] = []
        self.fault_reason: str | None = None
        # How long a probe may wait for a busy stage before reporting busy.
        # Set from --lock-timeout; kept as an attribute so it is one knob, not
        # a default buried in a signature.
        self.probe_timeout_s = 90.0

    def _mic_turn_log_fields(self, event: dict) -> dict:
        """Return an audit record without persisting speech content by default."""
        src = str(event.get("src") or "")
        tgt = str(event.get("tgt") or "")
        fields = {
            "turn": self.mic["turns"],
            "source": event.get("source"),
            "target": event.get("target"),
            "src_chars": len(src),
            "tgt_chars": len(tgt),
            "total_ms": event.get("total_ms"),
        }
        if self.log_content:
            fields.update(src_preview=src[:120], tgt_preview=tgt[:120])
        return fields

    # ---------------- state machine ----------------
    def set_state(self, new: str, reason: str = "") -> None:
        """States: starting -> warming -> ready <-> faulted -> stopping.

        `ready` is not sticky. Losing it is the whole point: the AIBOX failure
        this project is modelled on was a health endpoint that kept saying
        "active" after the model had been OOM-killed.
        """
        if new == self.state:
            return
        old, self.state = self.state, new
        if new == "faulted":
            self.fault_reason = reason or self.fault_reason
            with self._counters_lock:
                self.counters["faults"] += 1
        elif new == "ready":
            self.fault_reason = None
        self.log.emit(devlog.E_STATE, "warn" if new == "faulted" else "info",
                      reason, old_state=old, new_state=new)

    def _bump(self, key: str, n: int = 1) -> None:
        with self._counters_lock:
            self.counters[key] = self.counters.get(key, 0) + n

    # ---------------- degradation ----------------
    def stage_failed(self, name: str, error: str) -> str:
        """Record a RUNTIME failure of `name` and apply DEGRADATION. Returns the
        action taken: 'demote' | 'fault' | 'count'."""
        st = self.stage[name]
        st["runtime_failures"] += 1
        st["error"] = error
        policy = DEGRADATION[name]
        if policy["runtime"] == "demote":
            after = policy["demote_after"] or 1
            if st["runtime_failures"] >= after and not st["demoted"]:
                self.demote(name, error)
                return "demote"
            self.log.warn(devlog.E_STAGE_FAIL, error, stage=name,
                          runtime_failures=st["runtime_failures"],
                          demote_after=after)
            return "count"
        self.log.error(devlog.E_STAGE_FAIL, error, stage=name,
                       runtime_failures=st["runtime_failures"], policy="fault")
        return "fault"

    def demote(self, name: str, reason: str) -> None:
        """Turn a stage off for the rest of this process's life, and keep going."""
        st = self.stage[name]
        st["demoted"] = True
        st["ready"] = False
        st["demoted_reason"] = reason
        if name == "corrector" and self.pipe is not None:
            try:
                self.pipe.cfg.corrector_enabled = False
            except Exception:
                pass
        if name == "frontend":
            self._denoiser = None
        self.log.warn(devlog.E_STAGE_DEMOTE, reason, stage=name,
                      consequence=self.demotion_consequence(name))

    @staticmethod
    def demotion_consequence(name: str) -> str:
        return {
            "corrector": "raw ASR transcripts, no contextual correction",
            "tts": "text-only translation; /translate returns no audio",
            "frontend": "microphone disabled; HTTP translation unaffected",
            "asr": "no speech input; the device is not usable for its purpose",
            "mt": "no translation; the device is not usable for its purpose",
        }.get(name, "unknown")

    def stage_recovered(self, name: str) -> None:
        st = self.stage[name]
        if st["runtime_failures"] and not st["demoted"]:
            self.log.info(devlog.E_STAGE_RESTORE, stage=name,
                          after_failures=st["runtime_failures"])
            st["runtime_failures"] = 0
            st["error"] = None

    def tts_available(self) -> bool:
        return self.stage["tts"]["ready"] and not self.stage["tts"]["demoted"]

    def mic_available(self) -> tuple[bool, str]:
        """The mic needs a VAD. The denoiser is a quality upgrade, not a
        prerequisite — live_translate accepts denoiser=None."""
        fe = self.stage["frontend"]
        if not self.warm_frontend:
            return False, "frontend warm was skipped (--no-frontend)"
        if fe["demoted"]:
            return False, f"frontend demoted: {fe['demoted_reason']}"
        if not fe["detail"].get("vad_ok"):
            return False, f"Silero VAD unavailable: {fe.get('error')}"
        return True, ""

    # ---------------- startup ----------------
    def load_and_prewarm(self) -> bool:
        """Mandatory order. Each stage is warmed with a REAL inference using the
        project's own prewarm code (FastCascadeS2ST._asr_prewarm,
        TranscriptCorrector.prewarm, the translate/synthesize first calls that
        live_translate.LiveTranslator.__init__ performs)."""
        os.environ.setdefault("VOICEAI_TTS_BACKEND", "piper")
        os.environ.setdefault("VOICEAI_STREAMING_TTS", "1")
        os.environ.setdefault("VOICEAI_DENOISE_GTCRN", "0")
        os.environ.setdefault("VOICEAI_SILERO_CONFIRM", "1")
        os.environ.setdefault("VOICEAI_CORRECTOR", "1")

        self.set_state("warming")
        t_all = time.perf_counter()

        # ---- 1. import + construct ----------------------------------------
        self.log.info(devlog.E_STAGE_BEGIN, "import pipeline_fast", stage="pipeline")
        t = time.perf_counter()
        try:
            from pipeline_fast import FastCascadeS2ST, FastPipelineConfig
            cfg = FastPipelineConfig()
            cfg.corrector_enabled = True
            cfg.corrector_local_only = True
            raw = FastCascadeS2ST(cfg)
        except Exception as e:
            self.log.exception(devlog.E_STAGE_FAIL, e, stage="pipeline")
            self.stage["asr"]["error"] = repr(e)
            return False
        # Everything downstream — probes, HTTP, the mic loop — goes through the
        # guard. Nothing may touch `raw` after this point.
        self.pipe = GuardedPipeline(raw, logger=self.log)
        self.log.info(devlog.E_STAGE_OK, stage="pipeline",
                      ms=round((time.perf_counter() - t) * 1000, 1))

        # ---- 2. ASR -------------------------------------------------------
        self.log.info(devlog.E_STAGE_BEGIN, "PhoWhisper-small INT8, CT2", stage="asr")
        t = time.perf_counter()
        try:
            self.pipe._ensure_asr()          # project code; calls _asr_prewarm()
            ms = (time.perf_counter() - t) * 1000
            self.stage["asr"].update(
                ready=True, load_ms=round(ms, 1),
                prewarm_ms=round(self.pipe.raw._asr_prewarm_ms or 0.0, 1))
            self.log.info(devlog.E_STAGE_OK, stage="asr", load_ms=round(ms, 1),
                          project_prewarm_ms=round(self.pipe.raw._asr_prewarm_ms or 0.0, 1))
        except Exception as e:
            self.stage["asr"]["error"] = repr(e)
            self.log.exception(devlog.E_STAGE_FAIL, e, stage="asr", policy="fatal")
            return False

        # ---- 3. corrector (optional) --------------------------------------
        self.log.info(devlog.E_STAGE_BEGIN, "Vietnamese corrector (optional)",
                      stage="corrector")
        t = time.perf_counter()
        try:
            corrector = self.pipe._ensure_corrector()
            if corrector.prewarm():
                corrector.correct("xin chào")
                ms = (time.perf_counter() - t) * 1000
                self.stage["corrector"].update(ready=True, load_ms=round(ms, 1),
                                               prewarm_ms=round(ms, 1))
                self.log.info(devlog.E_STAGE_OK, stage="corrector", load_ms=round(ms, 1))
            else:
                err = str(getattr(corrector, "_load_error", "unavailable"))
                self.stage["corrector"]["error"] = err
                self._startup_degrade("corrector", err)
        except Exception as e:
            self.stage["corrector"]["error"] = repr(e)
            self._startup_degrade("corrector", repr(e))

        # ---- 4. MT --------------------------------------------------------
        self.log.info(devlog.E_STAGE_BEGIN, "NLLB-200-distilled-600M INT8, CT2", stage="mt")
        t = time.perf_counter()
        try:
            self.pipe._ensure_mt()
            load_ms = (time.perf_counter() - t) * 1000
            t2 = time.perf_counter()
            out = self.pipe.translate(PROBE_TEXT_VI, self.target, source="vi")
            warm_ms = (time.perf_counter() - t2) * 1000
            if not out.strip():
                raise RuntimeError("MT prewarm returned empty string")
            self.stage["mt"].update(ready=True, load_ms=round(load_ms, 1),
                                    prewarm_ms=round(warm_ms, 1))
            self.log.info(devlog.E_STAGE_OK, stage="mt", load_ms=round(load_ms, 1),
                          first_translation_ms=round(warm_ms, 1), output=out[:120])
        except Exception as e:
            self.stage["mt"]["error"] = repr(e)
            self.log.exception(devlog.E_STAGE_FAIL, e, stage="mt", policy="fatal")
            return False

        # ---- 5. TTS -------------------------------------------------------
        self.log.info(devlog.E_STAGE_BEGIN, f"Piper/sherpa-onnx target={self.target}",
                      stage="tts")
        t = time.perf_counter()
        try:
            audio, sr = self.pipe.synthesize(PROBE_TTS_TEXT, self.target)
            ms = (time.perf_counter() - t) * 1000
            dur = len(audio) / float(sr or 1)
            if dur < 0.05:
                raise RuntimeError(f"TTS prewarm produced {dur:.3f}s of audio")
            self.stage["tts"].update(ready=True, load_ms=round(ms, 1),
                                     prewarm_ms=round(ms, 1))
            self.log.info(devlog.E_STAGE_OK, stage="tts", load_ms=round(ms, 1),
                          audio_s=round(dur, 2), sample_rate=sr)
        except Exception as e:
            self.stage["tts"]["error"] = repr(e)
            self.log.exception(devlog.E_STAGE_FAIL, e, stage="tts", policy="fatal")
            return False

        # ---- 6. frontend --------------------------------------------------
        self.log.info(devlog.E_STAGE_BEGIN, "Silero VAD + GTCRN denoiser",
                      stage="frontend")
        if not self.warm_frontend:
            self.stage["frontend"].update(ready=False, load_ms=0, prewarm_ms=0,
                                          error="skipped (--no-frontend)")
            self.log.info(devlog.E_STAGE_SKIP, "--no-frontend", stage="frontend",
                          consequence="microphone unavailable; HTTP unaffected")
        else:
            self._warm_frontend()
            if self.strict_frontend and not self.stage["frontend"]["ready"]:
                self.log.error(devlog.E_STAGE_FAIL, "strict frontend requested",
                               stage="frontend", policy="fatal")
                return False

        self.warm_total_ms = (time.perf_counter() - t_all) * 1000
        return True

    def _warm_frontend(self) -> None:
        """Warm VAD and denoiser INDEPENDENTLY. They fail for different reasons
        and only one of them is a prerequisite for the mic."""
        st = self.stage["frontend"]
        t = time.perf_counter()
        errs = []
        try:
            import numpy as np
        except Exception as e:            # pragma: no cover - numpy is a hard dep
            st["error"] = repr(e)
            self._startup_degrade("frontend", repr(e))
            return
        silence = np.zeros(16000, dtype=np.float32)
        st["detail"]["denoiser_enabled"] = denoise_requested()
        if st["detail"]["denoiser_enabled"]:
            try:
                from edge_frontend import get_denoiser
                self._denoiser = get_denoiser()
                self._denoiser.denoise(silence, 16000)
                st["detail"]["denoiser_ok"] = True
            except Exception as e:
                self._denoiser = None
                st["detail"]["denoiser_ok"] = False
                errs.append(f"denoiser: {type(e).__name__}: {e}")
        else:
            self._denoiser = None
            st["detail"]["denoiser_ok"] = False
        try:
            from edge_frontend import get_vad
            self._vad = get_vad()
            self._vad.segments(silence, 16000)
            st["detail"]["vad_ok"] = True
        except Exception as e:
            self._vad = None
            st["detail"]["vad_ok"] = False
            errs.append(f"vad: {type(e).__name__}: {e}")
        ms = round((time.perf_counter() - t) * 1000, 1)
        st["load_ms"] = ms
        st["prewarm_ms"] = ms
        st["error"] = "; ".join(errs) or None
        if st["detail"].get("vad_ok") and st["detail"].get("denoiser_ok"):
            st["ready"] = True
            self.log.info(devlog.E_STAGE_OK, stage="frontend", load_ms=ms)
        elif st["detail"].get("vad_ok"):
            st["ready"] = True                # mic works, just without denoising
            self.log.warn(devlog.E_STAGE_DEGRADED, st["error"], stage="frontend",
                          consequence="microphone runs without GTCRN denoising")
        else:
            self._startup_degrade("frontend", st["error"] or "vad unavailable")

    def _startup_degrade(self, name: str, error: str) -> None:
        self.stage[name]["ready"] = False
        self.stage[name]["error"] = error
        self.log.warn(devlog.E_STAGE_DEGRADED, error, stage=name,
                      consequence=self.demotion_consequence(name))

    # ---------------- probes ----------------
    def probe(self, level: str = "light", timeout_s: float | None = None) -> dict:
        """REAL inference. This is the whole point: a process can be alive with a
        dead or evicted model, and only an actual forward pass proves otherwise.

        Never raises. Attributes every failure to a stage and applies the
        degradation policy for that stage.
        """
        timeout_s = self.probe_timeout_s if timeout_s is None else timeout_s
        res = {"ok": False, "level": level, "asr_ms": None, "mt_ms": None,
               "tts_ms": None, "total_ms": None, "detail": None, "busy": False,
               "failed_stage": None, "handled": False,
               "degraded": self.degraded_stages()}
        if self.pipe is None:
            res["detail"] = "pipeline not loaded"
            res["total_ms"] = 0.0
            return res

        t_all = time.perf_counter()
        self.log.emit(devlog.E_PROBE_BEGIN, "debug", level=level)
        stage = None
        try:
            ctx = self.pipe.deadline(timeout_s)
            ctx.__enter__()
        except Exception:                     # a fake pipeline in tests may lack it
            ctx = None
        try:
            if level == "full":
                stage = "asr"
                import soundfile as sf
                if not PROBE_WAV.is_file():
                    raise RuntimeError(f"probe wav missing: {PROBE_WAV}")
                audio, sr = sf.read(str(PROBE_WAV), dtype="float32")
                t = time.perf_counter()
                text = self.pipe.transcribe(audio, sr)
                res["asr_ms"] = round((time.perf_counter() - t) * 1000, 1)
                if not text.strip():
                    raise RuntimeError("ASR returned empty transcript on pinned wav")
                src_text = text
                self.stage_recovered("asr")
            else:
                src_text = PROBE_TEXT_VI

            stage = "mt"
            t = time.perf_counter()
            out = self.pipe.translate(src_text, self.target, source="vi")
            res["mt_ms"] = round((time.perf_counter() - t) * 1000, 1)
            if not out.strip():
                raise RuntimeError("MT returned empty string")
            self.stage_recovered("mt")

            # A demoted TTS is not probed: the whole point of demotion is that
            # the device keeps working without it, and re-probing a stage we
            # have decided not to use would fault a device that is fine.
            if self.tts_available():
                stage = "tts"
                t = time.perf_counter()
                wav, sr2 = self.pipe.synthesize(PROBE_TTS_TEXT, self.target)
                res["tts_ms"] = round((time.perf_counter() - t) * 1000, 1)
                dur = len(wav) / float(sr2 or 1)
                if dur < 0.05:
                    raise RuntimeError(f"TTS produced {dur:.3f}s of audio")
                self.stage_recovered("tts")

            res["ok"] = True
            res["detail"] = out[:120]
        except Busy as e:
            res["busy"] = True
            res["detail"] = f"busy: {e}"
            res["failed_stage"] = e.stage
            self._bump("busy_rejections")
            self.log.warn(devlog.E_PROBE_BUSY, str(e), level=level, stage=e.stage)
        except Exception as e:
            res["detail"] = f"{type(e).__name__}: {e}"
            res["failed_stage"] = stage
            if stage:
                # If the policy demoted the stage, the failure has already been
                # ANSWERED — the device is now running without it on purpose.
                # Counting it towards a fault as well would restart a device
                # that just finished degrading gracefully.
                res["handled"] = self.stage_failed(stage, res["detail"]) == "demote"
            self.log.exception(devlog.E_PROBE_FAIL, e, level=level, stage=stage)
        finally:
            res["total_ms"] = round((time.perf_counter() - t_all) * 1000, 1)
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass

        res["degraded"] = self.degraded_stages()
        self._record_probe(res)
        return res

    def _record_probe(self, res: dict) -> None:
        """Update counters and the readiness state machine from one probe.

        A `busy` result is contention, not a fault: it means another caller had
        the model, which is the guard working. It must never push the device
        towards 'faulted' or the watchdog will restart a healthy service in the
        middle of a long utterance.
        """
        with self._counters_lock:
            self.counters["probes"] += 1
            if res["ok"]:
                self.counters["consecutive_probe_failures"] = 0
            elif res["busy"] or res["handled"]:
                self.counters["consecutive_probe_failures"] = 0
            else:
                self.counters["probe_failures"] += 1
                self.counters["consecutive_probe_failures"] += 1
            consecutive = self.counters["consecutive_probe_failures"]

        self.last = dict(res)
        self.last["probe_utc"] = _utc()

        if res["ok"]:
            if self.state == "faulted":
                self.set_state("ready", "real probe passed again")
            else:
                self.log.emit(devlog.E_PROBE_OK, "debug", level=res["level"],
                              mt_ms=res["mt_ms"], tts_ms=res["tts_ms"],
                              asr_ms=res["asr_ms"])
            return
        if res["busy"] or res["handled"]:
            return
        if self.state == "ready" and consecutive >= FAULT_AFTER_FAILED_PROBES:
            self.set_state("faulted",
                           f"{consecutive} consecutive real probes failed "
                           f"({res['failed_stage']}: {res['detail']})")

    def degraded_stages(self) -> list[str]:
        return sorted(k for k, v in self.stage.items()
                      if v["demoted"] or (not v["ready"] and not v["required"]))

    # ---------------- microphone ----------------
    def start_mic(self, pair=None, manual_source=None, multilingual_asr=False,
                  play=True) -> None:
        """Run the live mic loop INSIDE this process, on the already-warm models.

        `prototype/live_translate.py mic` loads its own FastCascadeS2ST, so
        running it beside the supervisor puts a second ~1.4 GB copy of every model
        in RAM - which does not fit on an 8 GB box and would not fit on the target
        device either. LiveDeps exists precisely so the pipeline can be handed in.

        The GUARDED pipeline is handed in, not the raw one: the mic loop and its
        own worker threads then serialize against HTTP requests instead of
        racing them on the same model.
        """
        ok, why = self.mic_available()
        if not ok:
            raise RuntimeError(f"microphone unavailable: {why}")
        error = mic_configuration_error(mic=True, pair=pair, source=manual_source,
                                        multilingual_asr=multilingual_asr,
                                        target=self.target)
        if error:
            raise ValueError(error)
        import live_translate as lt

        cfg = lt.LiveConfig(target=self.target, pair=pair,
                            manual_source=manual_source,
                            multilingual_asr=multilingual_asr, play=play,
                            prewarm=False)   # this process is already warm

        # LiveDeps must be filled COMPLETELY. `LiveTranslator._attach_injected`
        # takes every heavy component from deps and builds none of them itself:
        # a deps object carrying only `pipe` leaves `stts = None` and
        # `_new_vad = lambda: None`, so the loop dies on `vad.accept_waveform`
        # at the first audio frame (AttributeError on None) and again on
        # `self.stts.stream(...)` when streaming TTS fires. The mic path was
        # never run against real hardware, so that never surfaced.
        deps_kw = {"pipe": self.pipe, "denoiser": self._denoiser}

        if getattr(self.pipe.cfg, "streaming_tts", False):
            from streaming_tts import StreamingTTS
            # Synthesis goes through the GUARD, so a mic fragment and an HTTP
            # request cannot be inside sherpa-onnx at the same time.
            deps_kw["stts"] = StreamingTTS(
                lambda tx, tg: self.pipe.synthesize(tx, tg))

        import sherpa_onnx
        from edge_frontend import SILERO_URL, _ensure
        vc = sherpa_onnx.VadModelConfig()
        vc.silero_vad.model = _ensure(SILERO_URL)
        vc.silero_vad.min_silence_duration = cfg.min_silence
        vc.silero_vad.threshold = cfg.vad_threshold
        vc.sample_rate = lt.SR
        deps_kw["vad_window"] = vc.silero_vad.window_size
        deps_kw["vad_factory"] = lambda: sherpa_onnx.VoiceActivityDetector(
            vc, buffer_size_in_seconds=100)

        self._live = lt.LiveTranslator(cfg, deps=lt.LiveDeps(**deps_kw))
        self.mic["enabled"] = True

        def emit(line: str) -> None:
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                return
            kind = event.get("type")
            if kind == "direction":
                self.mic["last_direction"] = f"{event.get('source')}->{event.get('target')}"
            elif kind == "final":
                self.mic["turns"] += 1
                self.mic["last_utc"] = _utc()
                self.log.emit(devlog.E_MIC_TURN, "debug",
                              **self._mic_turn_log_fields(event))
            print(line, flush=True)

        def run() -> None:
            self.mic["running"] = True
            try:
                self._live.run_mic(emit=emit)
            except Exception as exc:                     # noqa: BLE001 - reported, not raised
                self.mic["error"] = f"{type(exc).__name__}: {exc}"
                self.log.exception(devlog.E_MIC_FAIL, exc,
                                   consequence="HTTP translation still available")
            finally:
                self.mic["running"] = False
                self.log.info(devlog.E_MIC_STOP, turns=self.mic["turns"])

        self._mic_thread = threading.Thread(target=run, name="mic", daemon=True)
        self._mic_thread.start()
        self.log.info(devlog.E_MIC_START,
                      pair=list(pair) if pair else ["vi", self.target],
                      multilingual_asr=multilingual_asr,
                      denoiser=bool(self._denoiser is not None
                                    and (cfg.denoise or cfg.denoise_vad)))

    # ---------------- reporting ----------------
    def blocks_readiness(self, name: str) -> bool:
        """A required stage blocks readiness UNLESS it was demoted on purpose.

        Demotion is a decision the device already made and reported (it is in
        `degraded` and in the log). Letting it also hold `ready` at False would
        mean the graceful-degradation path ends in a watchdog restart loop —
        the opposite of what it is for.
        """
        st = self.stage[name]
        return st["required"] and not st["demoted"] and not st["ready"]

    def health(self) -> dict:
        required_ok = not any(self.blocks_readiness(k) for k in self.stage)
        mic_ok, mic_why = self.mic_available()
        with self._counters_lock:
            counters = dict(self.counters)
        return {
            "state": self.state,
            "ready": self.state == "ready" and required_ok,
            "fault_reason": self.fault_reason,
            "degraded": self.degraded_stages(),
            "degraded_consequences": {k: self.demotion_consequence(k)
                                      for k in self.degraded_stages()},
            "target": self.target,
            "pid": os.getpid(),
            "run_id": self.log.run_id,
            "uptime_s": round(time.monotonic() - self.started_monotonic, 1),
            "started_utc": self.started_utc,
            "ready_utc": self.ready_utc,
            "warm_total_ms": None if self.warm_total_ms is None else round(self.warm_total_ms, 1),
            "rss_mb": round(rss_mb(), 1),
            "rss_peak_mb": round(rss_mb(peak=True), 1),
            "stages": self.stage,
            "last_probe": self.last,
            "counters": counters,
            "concurrency": (self.pipe.stats() if self.pipe is not None else {}),
            "mic": dict(self.mic, available=mic_ok,
                        unavailable_reason=None if mic_ok else mic_why),
            "keepalive": {
                "thread_alive": bool(self._keepalive_thread
                                     and self._keepalive_thread.is_alive()),
                "last_utc": self.keepalive_last_utc,
            },
            "config_warnings": self.config_warnings,
            "privacy": {
                "content_logging": self.log_content,
                "structured_log_policy": ("content-preview-opt-in" if self.log_content
                                          else "metadata-only"),
            },
            "models_resident": True,
            "note": ("Models are never unloaded by this process. The only eviction "
                     "in the system is the pipeline's 2-slot Piper voice LRU: a "
                     "third target language reloads a voice (~2.2 s, "
                     "Results/kaggle_t4/README.md TTS load)."),
        }

    def metrics_text(self) -> str:
        h = self.health()
        lines = [
            f"voiceai_ready {int(bool(h['ready']))}",
            f"voiceai_faulted {int(h['state'] == 'faulted')}",
            f"voiceai_uptime_seconds {h['uptime_s']}",
            f"voiceai_rss_mb {h['rss_mb']}",
            f"voiceai_rss_peak_mb {h['rss_peak_mb']}",
            f"voiceai_warm_total_ms {h['warm_total_ms']}",
            f"voiceai_keepalive_alive {int(bool(h['keepalive']['thread_alive']))}",
            f"voiceai_mic_running {int(bool(h['mic']['running']))}",
            f"voiceai_mic_turns_total {h['mic']['turns']}",
        ]
        for k, v in h["counters"].items():
            lines.append(f"voiceai_{k}_total {v}")
        for s, v in h["stages"].items():
            lines.append(f"voiceai_stage_ready{{stage=\"{s}\"}} {int(bool(v['ready']))}")
            lines.append(f"voiceai_stage_demoted{{stage=\"{s}\"}} {int(bool(v['demoted']))}")
            lines.append(f"voiceai_stage_runtime_failures{{stage=\"{s}\"}} "
                         f"{v['runtime_failures']}")
            if v["load_ms"] is not None:
                lines.append(f"voiceai_stage_load_ms{{stage=\"{s}\"}} {v['load_ms']}")
        for s, v in h["concurrency"].items():
            lines.append(f"voiceai_guard_calls{{stage=\"{s}\"}} {v['calls']}")
            lines.append(f"voiceai_guard_waits{{stage=\"{s}\"}} {v['waits']}")
            lines.append(f"voiceai_guard_wait_ms_total{{stage=\"{s}\"}} {v['wait_ms_total']}")
            lines.append(f"voiceai_guard_max_wait_ms{{stage=\"{s}\"}} {v['max_wait_ms']}")
            lines.append(f"voiceai_guard_timeouts{{stage=\"{s}\"}} {v['timeouts']}")
        lp = h["last_probe"]
        for k in ("asr_ms", "mt_ms", "tts_ms", "total_ms"):
            if lp.get(k) is not None:
                lines.append(f"voiceai_last_probe_{k} {lp[k]}")
        return "\n".join(lines) + "\n"

    # ---------------- keepalive ----------------
    def keepalive_loop(self, period_s: float, stop: threading.Event | None = None,
                       max_iterations: int | None = None) -> None:
        """CPython does not unload a loaded model, so this is not the AIBOX
        keepalive (which fought a runtime that dropped the model after 300 s
        idle). What it does buy on a portable box is real: it keeps the model
        pages resident against the OS paging them out during a long Q&A gap,
        and it keeps CT2's thread pool hot. Cheap: one MT+TTS call.

        This loop must never die. If it does, nothing is exercising the models
        between demos and nothing says so — so every iteration is wrapped, and
        `/healthz.keepalive.thread_alive` reports whether it is still running.
        """
        stop = stop or threading.Event()
        i = 0
        while not stop.is_set():
            if stop.wait(period_s):
                break
            i += 1
            try:
                if self.state not in ("ready", "faulted"):
                    continue
                # A live mic turn holds the models; a keepalive is pointless then
                # and would only queue behind it.
                if self.mic["running"] and self.pipe is not None:
                    holders = self.pipe.holders()
                    if any(h for h in holders.values()):
                        self.log.emit(devlog.E_KEEPALIVE_TICK, "debug",
                                      skipped="pipeline busy")
                        continue
                r = self.probe("light", timeout_s=30.0)
                self.keepalive_last_utc = _utc()
                if not r["ok"] and not r["busy"]:
                    self.log.warn(devlog.E_KEEPALIVE_FAIL, str(r["detail"]),
                                  failed_stage=r["failed_stage"])
                else:
                    self.log.emit(devlog.E_KEEPALIVE_TICK, "debug",
                                  ok=r["ok"], mt_ms=r["mt_ms"], tts_ms=r["tts_ms"])
            except Exception as exc:      # noqa: BLE001 - the loop outlives any error
                self.log.exception(devlog.E_KEEPALIVE_FAIL, exc)
            if max_iterations is not None and i >= max_iterations:
                return
        self.log.warn(devlog.E_THREAD_DIED, "keepalive loop exited", thread="keepalive")


def make_handler(sup: Supervisor):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # Without this a client that opens a socket and never sends a request
        # line pins a worker thread forever on rfile.readline().
        timeout = HTTP_HANDLER_TIMEOUT_S

        def log_message(self, fmt, *a):  # silence per-request stderr spam
            pass

        def log_error(self, fmt, *a):
            sup.log.warn(devlog.E_HTTP_ERROR, (fmt % a) if a else str(fmt),
                         client=self.client_address[0] if self.client_address else None)

        def _send(self, code: int, body: bytes, ctype="application/json"):
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # The operator hit Ctrl-C in curl. Not an incident.
                self.close_connection = True

        def _json(self, code: int, obj):
            try:
                body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
            except Exception as e:      # never fail to answer because of encoding
                body = json.dumps({"error": f"unserialisable response: {e!r}"}).encode()
                code = 500
            self._send(code, body)

        # -- routing --------------------------------------------------------
        def do_GET(self):
            t0 = time.perf_counter()
            try:
                self._route_get()
            except Exception as e:      # a handler thread must never die silently
                sup.log.exception(devlog.E_HTTP_ERROR, e, path=self.path, method="GET")
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            finally:
                sup.log.emit(devlog.E_HTTP_REQ, "debug", method="GET", path=self.path,
                             ms=round((time.perf_counter() - t0) * 1000, 1))

        def do_POST(self):
            t0 = time.perf_counter()
            try:
                self._route_post()
            except Exception as e:
                sup.log.exception(devlog.E_HTTP_ERROR, e, path=self.path, method="POST")
                self._json(500, {"error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc(limit=3)})
            finally:
                sup.log.emit(devlog.E_HTTP_REQ, "debug", method="POST", path=self.path,
                             ms=round((time.perf_counter() - t0) * 1000, 1))

        def _route_get(self):
            path, _, query = self.path.partition("?")
            q = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
            if path in ("/healthz", "/health", "/"):
                h = sup.health()
                self._json(200 if h["ready"] else 503, h)
            elif path == "/probe":
                level = q.get("level", "light")
                if level not in ("light", "full"):
                    self._json(400, {"error": "level must be light|full"})
                    return
                r = sup.probe(level)
                # 503 for a real failure; 429 for contention, so the watchdog can
                # tell "the model is broken" from "someone else is using it".
                code = 200 if r["ok"] else (429 if r["busy"] else 503)
                self._json(code, r)
            elif path == "/metrics":
                self._send(200, sup.metrics_text().encode("utf-8"),
                           "text/plain; charset=utf-8")
            elif path == "/events":
                try:
                    n = max(1, min(2000, int(q.get("n", "200"))))
                except ValueError:
                    n = 200
                self._json(200, {"run_id": sup.log.run_id,
                                 "events": sup.log.recent(n)})
            else:
                self._json(404, {"error": "not found",
                                 "endpoints": ["/healthz", "/probe?level=light|full",
                                               "/metrics", "/events?n=200",
                                               "POST /translate"]})

        def _route_post(self):
            if self.path != "/translate":
                self._json(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                self._json(400, {"error": "bad Content-Length"})
                return
            if n < 0 or n > MAX_BODY_BYTES:
                sup.log.warn(devlog.E_HTTP_REJECT, "oversized body",
                             content_length=n, limit=MAX_BODY_BYTES)
                self._json(413, {"error": f"body over {MAX_BODY_BYTES} bytes"})
                return
            raw = self.rfile.read(n) if n else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except (ValueError, UnicodeDecodeError) as e:
                # Git Bash curl re-encodes Vietnamese and lands here; say so.
                self._json(400, {"error": f"body is not valid UTF-8 JSON: {e}",
                                 "hint": "send UTF-8; see DEMO_RUNBOOK.md §2"})
                return
            if not isinstance(payload, dict):
                self._json(400, {"error": "body must be a JSON object"})
                return

            text = str(payload.get("text", "")).strip()
            target = str(payload.get("target", sup.target)).strip().lower()
            source = str(payload.get("source", "vi")).strip().lower()
            speak = bool(payload.get("tts", True))
            if not text:
                self._json(400, {"error": "empty text"})
                return
            if len(text) > MAX_TEXT_CHARS:
                self._json(413, {"error": f"text over {MAX_TEXT_CHARS} characters"})
                return
            if target not in TARGETS or source not in TARGETS:
                self._json(400, {"error": f"target/source must be one of {list(TARGETS)}"})
                return
            if not sup.stage["mt"]["ready"] or sup.pipe is None:
                self._json(503, {"error": "MT stage not ready",
                                 "state": sup.state,
                                 "detail": sup.stage["mt"]["error"]})
                return

            try:
                t = time.perf_counter()
                out = sup.pipe.translate(text, target, source=source)
                mt_ms = (time.perf_counter() - t) * 1000
                sup.stage_recovered("mt")
            except Busy as e:
                sup._bump("busy_rejections")
                sup.log.warn(devlog.E_HTTP_REJECT, str(e), stage=e.stage)
                self._json(429, {"error": "busy", "busy": True, "stage": e.stage,
                                 "detail": str(e)})
                return
            except Exception as e:
                sup._bump("translate_failures")
                sup.stage_failed("mt", f"{type(e).__name__}: {e}")
                sup.log.exception(devlog.E_TRANSLATE_FAIL, e, stage="mt")
                self._json(503, {"error": f"translation failed: {type(e).__name__}: {e}"})
                return

            # TTS is degradable: a failure here returns the TEXT plus an explicit
            # tts_error rather than a 500. Text without audio still demos.
            tts_ms = dur = None
            tts_error = None
            if speak and sup.tts_available():
                try:
                    t = time.perf_counter()
                    wav, sr = sup.pipe.synthesize(out, target)
                    tts_ms = (time.perf_counter() - t) * 1000
                    dur = len(wav) / float(sr or 1)
                    sup.stage_recovered("tts")
                except Busy as e:
                    tts_error = f"busy: {e}"
                    sup._bump("busy_rejections")
                except Exception as e:
                    tts_error = f"{type(e).__name__}: {e}"
                    sup.stage_failed("tts", tts_error)
                    sup.log.exception(devlog.E_TRANSLATE_FAIL, e, stage="tts",
                                      consequence="text returned without audio")
            elif speak:
                tts_error = (f"tts unavailable: "
                             f"{sup.stage['tts']['demoted_reason'] or sup.stage['tts']['error']}")

            sup._bump("translate_calls")
            sup.log.emit(devlog.E_TRANSLATE_OK, "debug", target=target, source=source,
                         chars=len(text), mt_ms=round(mt_ms, 1),
                         tts_ms=None if tts_ms is None else round(tts_ms, 1),
                         degraded=bool(tts_error))
            self._json(200, {"src": text, "target": target, "source": source,
                             "translation": out,
                             "mt_ms": round(mt_ms, 1),
                             "tts_ms": None if tts_ms is None else round(tts_ms, 1),
                             "audio_s": None if dur is None else round(dur, 2),
                             "tts_error": tts_error,
                             "degraded": sup.degraded_stages()})
    return H


def bind_or_die(host: str, port: int) -> socket.socket:
    """Claim the port BEFORE loading 1.6 GB of models. This is our version of the
    AIBOX camera-ownership check: a second supervisor must refuse to start rather
    than half-load and fight the first one for CPU."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if not sys.platform.startswith("win"):
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
    except OSError as e:
        s.close()
        LOG.fatal(devlog.E_PORT_CONFLICT,
                  f"{host}:{port} is already in use ({e.strerror or e})",
                  host=host, port=port,
                  remedy=f"python device/healthcheck.py --port {port}")
        sys.exit(3)
    s.listen(16)
    LOG.info(devlog.E_PORT_CLAIMED, host=host, port=port)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VoiceAI resident device supervisor")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("VOICEAI_DEVICE_PORT", "8731") or "8731"))
    ap.add_argument("--target", default=os.environ.get("VOICEAI_TARGET", "en"),
                    choices=list(TARGETS))
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the model manifest check (dev only)")
    ap.add_argument("--quick-verify", action="store_true",
                    help="size-only manifest check (seconds instead of a full hash)")
    ap.add_argument("--no-frontend", action="store_true",
                    help="skip VAD/denoiser warm (text-only demo, mic disabled)")
    ap.add_argument("--strict-frontend", action="store_true",
                    help="treat a frontend warm failure as fatal (default: degrade, "
                         "mic off, HTTP translation keeps working)")
    ap.add_argument("--keepalive", type=float,
                    default=float(os.environ.get("VOICEAI_KEEPALIVE_S", "120") or "120"),
                    help="seconds between idle keepalive probes; 0 disables")
    ap.add_argument("--lock-timeout", type=float, default=90.0,
                    help="seconds a call may wait for a busy model stage before "
                         "the request is refused with 429 instead of hanging")
    ap.add_argument("--mic", action="store_true",
                    help="own the microphone in this process, on the resident models "
                         "(running prototype/live_translate.py mic separately loads a "
                         "SECOND full copy of every model)")
    ap.add_argument("--pair", help="bidirectional pair for mic mode, e.g. vi,ko")
    ap.add_argument("--source", "--manual-source", dest="manual_source",
                    choices=list(TARGETS),
                    help="initial mic source selected before ASR; --manual-source is "
                         "kept as a compatibility alias")
    ap.add_argument("--multilingual-asr", action="store_true",
                    help="mic mode: route en/zh/ko turns to the multilingual ASR")
    ap.add_argument("--no-play", action="store_true",
                    help="mic mode: do not play synthesized audio")
    ap.add_argument("--warm-only", action="store_true",
                    help="load, prewarm, probe once, print READY, exit (timing runs)")
    ap.add_argument("--check-config", action="store_true",
                    help="validate the environment and model tree, then exit")
    args = ap.parse_args(argv)

    mic_error = mic_configuration_error(
        mic=args.mic, pair=args.pair, source=args.manual_source,
        multilingual_asr=args.multilingual_asr, target=args.target,
    )
    if mic_error:
        ap.error(mic_error)

    LOG.info(devlog.E_START, "VoiceAI device supervisor",
             pid=os.getpid(), target=args.target, port=args.port,
             python=sys.version.split()[0], platform=sys.platform,
             cpu_count=os.cpu_count(), argv=sys.argv[1:])

    # -- step 0a: configuration validation (before ANY model touches RAM) --
    # pipeline_fast reads its config in dataclass field defaults, i.e. at import
    # time. A bad value there raises inside an import, ~20 s in. Catch it here.
    try:
        warnings = device_config.require_valid(check_models=args.skip_verify, logger=LOG)
    except device_config.ConfigError as e:
        LOG.fatal(devlog.E_CONFIG_INVALID, str(e))
        return 2
    LOG.info(devlog.E_CONFIG_OK, warnings=len(warnings))
    if args.check_config:
        print(device_config.explain())
        return 0

    # -- step 0b: resource conflict check --
    sock = None
    if not args.warm_only:
        sock = bind_or_die(args.host, args.port)

    # -- step 1: provenance --
    if args.skip_verify:
        LOG.warn(devlog.E_MANIFEST_SKIP, "--skip-verify: model integrity NOT checked")
    else:
        LOG.info(devlog.E_MANIFEST_BEGIN, quick=bool(args.quick_verify))
        t = time.perf_counter()
        try:
            import provenance
            rc = provenance.verify(quick=args.quick_verify)
        except Exception as e:
            LOG.exception(devlog.E_MANIFEST_FAIL, e)
            rc = 5
        if rc != 0:
            if sock:
                sock.close()
            LOG.fatal(devlog.E_NOT_READY, "model verification failed", rc=rc)
            return 5
        LOG.info(devlog.E_MANIFEST_OK, ms=round((time.perf_counter() - t) * 1000, 1))

    # -- step 2..6: load + prewarm --
    sup = Supervisor(target=args.target, warm_frontend=not args.no_frontend,
                     strict_frontend=args.strict_frontend, log=LOG)
    sup.config_warnings = warnings
    try:
        ok = sup.load_and_prewarm()
    except Exception as e:                    # a warm must never crash the process
        LOG.exception(devlog.E_STAGE_FAIL, e, stage="unknown")
        ok = False
    sup.probe_timeout_s = float(args.lock_timeout)
    if sup.pipe is not None:
        object.__setattr__(sup.pipe, "_timeout_s", float(args.lock_timeout))
    if not ok:
        if sock:
            sock.close()
        failed = [k for k, v in sup.stage.items() if v["required"] and not v["ready"]]
        LOG.fatal(devlog.E_NOT_READY, "stage(s) failed to warm", failed=failed,
                  errors={k: sup.stage[k]["error"] for k in failed})
        return 4

    # -- step 7: one real end-to-end probe before declaring ready --
    LOG.info(devlog.E_PROBE_BEGIN, "acceptance probe (full: ASR + MT + TTS on "
                                   "pinned VIVOS wav)", level="full", phase="acceptance")
    r = sup.probe("full")
    if not r["ok"]:
        if sock:
            sock.close()
        LOG.fatal(devlog.E_NOT_READY, "acceptance probe failed", detail=r["detail"],
                  failed_stage=r["failed_stage"])
        return 4

    sup.set_state("ready", "acceptance probe passed")
    sup.ready_utc = _utc()
    LOG.info(devlog.E_READY, "DEVICE READY",
             warm_total_s=round(sup.warm_total_ms / 1000.0, 1),
             rss_mb=round(rss_mb(), 1), rss_peak_mb=round(rss_mb(peak=True), 1),
             probe_total_ms=r["total_ms"], asr_ms=r["asr_ms"], mt_ms=r["mt_ms"],
             tts_ms=r["tts_ms"], degraded=sup.degraded_stages(),
             endpoint=f"http://{args.host}:{args.port}/healthz")

    if args.warm_only:
        print(json.dumps(sup.health(), ensure_ascii=False, indent=2))
        return 0

    if args.mic:
        pair = tuple(x.strip() for x in args.pair.split(",")) if args.pair else None
        try:
            sup.start_mic(pair=pair, manual_source=args.manual_source,
                          multilingual_asr=args.multilingual_asr,
                          play=not args.no_play)
        except Exception as exc:                          # noqa: BLE001
            sup.mic["error"] = f"{type(exc).__name__}: {exc}"
            LOG.exception(devlog.E_MIC_FAIL, exc,
                          consequence="HTTP translation still available")

    stop = threading.Event()
    if args.keepalive > 0:
        sup._keepalive_thread = threading.Thread(
            target=sup.keepalive_loop, args=(args.keepalive, stop),
            name="keepalive", daemon=True)
        sup._keepalive_thread.start()
        LOG.info(devlog.E_KEEPALIVE_TICK, "keepalive armed", period_s=args.keepalive)

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(sup),
                                bind_and_activate=False)
    httpd.socket = sock

    def _stop(signum, _frame):
        # systemd sends SIGTERM, not SIGINT. Without this the unit is killed
        # after the stop timeout and the log ends mid-sentence.
        LOG.warn(devlog.E_SIGNAL, "shutdown signal", signal=int(signum))
        stop.set()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for signame in ("SIGTERM", "SIGINT", "SIGBREAK"):
        s = getattr(signal, signame, None)
        if s is not None:
            try:
                signal.signal(s, _stop)
            except (ValueError, OSError):
                pass

    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOG.warn(devlog.E_SIGNAL, "KeyboardInterrupt")
    finally:
        stop.set()
        sup.set_state("stopping", "shutdown")
        httpd.server_close()
        LOG.info(devlog.E_SHUTDOWN, uptime_s=round(time.monotonic() - sup.started_monotonic, 1),
                 counters=sup.counters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
