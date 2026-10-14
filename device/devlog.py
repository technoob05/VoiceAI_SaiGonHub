"""devlog.py — dependency-free structured logging for the device layer.

Why this exists
---------------
The device is debugged from a log file, after the fact, in a noisy hall, by
someone who cannot reproduce the failure. `print(f"[{ts}] {msg}")` cannot serve
that: you cannot grep it by event, you cannot diff two runs, and the fields you
need (which stage, how long, which attempt) are welded into prose.

This module writes TWO streams from one call:

  * a human line to the console            -> the operator watching the window
  * one JSON object per line to a .jsonl   -> the thing you grep after a failure

No new pip package. `json` + `threading` + the stdlib is all of it.

Event vocabulary
----------------
Every event name is a dotted `<subject>.<verb>` string declared in EVENTS below
and nowhere else. A typo cannot invent a new event: `emit()` rejects names that
are not in the vocabulary (in strict mode, which the tests use), and the
constants are what the code references. To find every failure of a stage:

    grep '"event": "stage.fail"' device/logs/supervisor.jsonl

To reconstruct one restart cycle, filter on `run_id`, which is stable for the
lifetime of a process and is stamped on every record.

Rotation
--------
A device that runs for a day emits a keepalive probe every 120 s. Left alone the
log grows without bound on a box with a small disk, so the writer rotates at
`max_bytes` (default 8 MiB) keeping `backups` (default 3) generations. Rotation
is size-checked on write, not scheduled, so it needs no extra thread.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

# --------------------------------------------------------------------------
# Stable event vocabulary. Adding an event means adding it HERE first.
# --------------------------------------------------------------------------
# process lifecycle
E_START = "process.start"
E_CONFIG_OK = "config.ok"
E_CONFIG_INVALID = "config.invalid"
E_CONFIG_WARN = "config.warn"
E_PORT_CLAIMED = "port.claimed"
E_PORT_CONFLICT = "port.conflict"
E_MANIFEST_BEGIN = "manifest.begin"
E_MANIFEST_OK = "manifest.ok"
E_MANIFEST_FAIL = "manifest.fail"
E_MANIFEST_SKIP = "manifest.skip"
E_SHUTDOWN = "process.shutdown"
E_SIGNAL = "process.signal"

# model stages
E_STAGE_BEGIN = "stage.begin"
E_STAGE_OK = "stage.ok"
E_STAGE_FAIL = "stage.fail"
E_STAGE_SKIP = "stage.skip"
E_STAGE_DEGRADED = "stage.degraded"      # optional stage unavailable at startup
E_STAGE_DEMOTE = "stage.demote"          # working stage disabled at RUNTIME
E_STAGE_RESTORE = "stage.restore"

# readiness / state machine
E_STATE = "state.change"
E_READY = "device.ready"
E_NOT_READY = "device.not_ready"

# probes
E_PROBE_BEGIN = "probe.begin"
E_PROBE_OK = "probe.ok"
E_PROBE_FAIL = "probe.fail"
E_PROBE_BUSY = "probe.busy"

# http surface
E_HTTP_REQ = "http.request"
E_HTTP_REJECT = "http.reject"
E_HTTP_ERROR = "http.error"
E_TRANSLATE_OK = "translate.ok"
E_TRANSLATE_FAIL = "translate.fail"

# background threads
E_KEEPALIVE_TICK = "keepalive.tick"
E_KEEPALIVE_FAIL = "keepalive.fail"
E_THREAD_DIED = "thread.died"
E_MIC_START = "mic.start"
E_MIC_FAIL = "mic.fail"
E_MIC_TURN = "mic.turn"
E_MIC_STOP = "mic.stop"

# concurrency guard
E_GUARD_WAIT = "guard.wait"              # a call queued behind another
E_GUARD_TIMEOUT = "guard.timeout"

# watchdog
E_WD_START = "watchdog.start"
E_WD_PROBE = "watchdog.probe"
E_WD_FAULT = "watchdog.fault"
E_WD_EVIDENCE = "watchdog.evidence"
E_WD_RESTART = "watchdog.restart"
E_WD_RECOVERED = "watchdog.recovered"
E_WD_GIVEUP = "watchdog.giveup"
E_WD_BACKOFF = "watchdog.backoff"

EVENTS = frozenset(v for k, v in list(globals().items())
                   if k.startswith("E_") and isinstance(v, str))

LEVELS = ("debug", "info", "warn", "error", "fatal")


def _utc() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


class Logger:
    """One logger per process. Thread-safe; every write holds a short lock."""

    def __init__(self, component: str, path: Path | None = None,
                 console: bool = True, strict: bool = True,
                 max_bytes: int = 8 << 20, backups: int = 3,
                 run_id: str | None = None, ring: int = 400):
        self.component = component
        self.path = Path(path) if path else None
        self.console = console
        self.strict = strict
        self.max_bytes = int(max_bytes)
        self.backups = int(backups)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        # In-memory ring of the most recent records. The watchdog cannot read a
        # crashed child's memory, but the supervisor CAN dump this into its own
        # evidence and /healthz, which is how you see the last N events when the
        # log file write is what failed.
        self._ring: list[dict] = []
        self._ring_max = int(ring)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- vocabulary ---------------------------------------------------------
    def _check(self, event: str) -> None:
        if event not in EVENTS:
            msg = (f"devlog: unknown event {event!r}. Add it to devlog.EVENTS "
                   f"before using it — the vocabulary is the point.")
            if self.strict:
                raise ValueError(msg)
            print(msg, file=sys.stderr, flush=True)

    # -- writing ------------------------------------------------------------
    def emit(self, event: str, level: str = "info", msg: str = "", /, **fields) -> dict:
        # positional-only: a FIELD may legitimately be called "level", "msg" or
        # "event" (probe level, http method...) and must not collide with the
        # signature.
        self._check(event)
        if level not in LEVELS:
            level = "info"
        rec = {
            "ts": _utc(),
            "uptime_s": round(time.monotonic() - self._t0, 3),
            "level": level,
            "component": self.component,
            "event": event,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
        }
        if msg:
            rec["msg"] = msg
        for k, v in fields.items():
            rec[k] = _safe(v)
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._ring.append(rec)
            if len(self._ring) > self._ring_max:
                del self._ring[:len(self._ring) - self._ring_max]
            if self.console:
                print(self._human(rec), flush=True)
            if self.path:
                try:
                    self._rotate_if_needed(len(line) + 1)
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except Exception as exc:                  # never let logging kill us
                    print(f"[devlog] write failed: {exc!r}", file=sys.stderr, flush=True)
        return rec

    def _rotate_if_needed(self, incoming: int) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            if self.path.stat().st_size + incoming < self.max_bytes:
                return
        except OSError:
            return
        for i in range(self.backups - 1, 0, -1):
            src = self.path.with_suffix(self.path.suffix + f".{i}")
            dst = self.path.with_suffix(self.path.suffix + f".{i + 1}")
            if src.exists():
                try:
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)
                except OSError:
                    pass
        try:
            first = self.path.with_suffix(self.path.suffix + ".1")
            if first.exists():
                first.unlink()
            self.path.rename(first)
        except OSError:
            pass

    @staticmethod
    def _human(rec: dict) -> str:
        skip = {"ts", "level", "component", "event", "run_id", "pid", "thread",
                "uptime_s", "msg"}
        extra = " ".join(f"{k}={_short(v)}" for k, v in rec.items() if k not in skip)
        lvl = rec["level"].upper()
        head = f"{rec['ts']} {lvl:<5} {rec['event']:<20}"
        body = rec.get("msg", "")
        return f"{head} {body}{('  ' + extra) if extra else ''}".rstrip()

    # -- convenience --------------------------------------------------------
    def info(self, event, msg="", /, **f):  return self.emit(event, "info", msg, **f)
    def warn(self, event, msg="", /, **f):  return self.emit(event, "warn", msg, **f)
    def error(self, event, msg="", /, **f): return self.emit(event, "error", msg, **f)
    def fatal(self, event, msg="", /, **f): return self.emit(event, "fatal", msg, **f)

    def exception(self, event: str, exc: BaseException, msg: str = "", /, **fields):
        import traceback
        return self.emit(event, "error", msg or f"{type(exc).__name__}: {exc}",
                         error_type=type(exc).__name__, error=str(exc),
                         traceback=traceback.format_exc(limit=8), **fields)

    def recent(self, n: int = 100) -> list[dict]:
        with self._lock:
            return list(self._ring[-n:])


def _safe(v):
    """Make a field JSON-serialisable without ever raising."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in v.items()}
    try:
        json.dumps(v)
        return v
    except Exception:
        return repr(v)[:400]


def _short(v, n: int = 90) -> str:
    s = v if isinstance(v, str) else json.dumps(_safe(v), ensure_ascii=False)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n - 1] + "…"


_DEFAULT: Logger | None = None


def get_logger(component: str = "device", path: Path | None = None, **kw) -> Logger:
    """Process-wide logger. First call wins; later calls return the same object
    so a module imported twice does not open a second file handle."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Logger(component, path, **kw)
    return _DEFAULT


def reset_logger() -> None:
    """Tests only."""
    global _DEFAULT
    _DEFAULT = None
