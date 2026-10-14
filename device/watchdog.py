"""watchdog.py — probe the supervisor on an interval; on failure COLLECT
EVIDENCE first, then restart and re-warm.

Modelled directly on
references/Summer_School_AIEye/aibox_eyes/systemd/geniex-watchdog.sh. The lesson
there was not "restart things": it was that the crash was invisible
(`docker run --rm` wiped the logs, `systemctl is-active` reported "active"), so
the watchdog had to *write down what it saw* before it destroyed the evidence by
restarting. Same discipline here: an evidence bundle lands in device/evidence/
before the process is killed.

Differences from the AIBOX version, stated honestly:
  * It is Python, not sh, because the dev box is Windows and the target is
    Linux/ARM — one implementation, both platforms.
  * There is no systemd on the dev box, so this process owns the supervisor as a
    child rather than calling `systemctl restart`. On the Linux target you can
    instead run the supervisor under systemd and this watchdog with
    --restart-cmd "systemctl restart voiceai-supervisor" (see device/systemd/).
  * dmesg OOM evidence is Linux-only; on Windows we collect tasklist + global
    memory instead and say so in the bundle.

The restart DECISION is a pure function (`decide`) rather than a tangle of
conditions inside the loop, so it can be tested without a supervisor, a model,
or a demo. See device/test_device.py.

Usage
    python device/watchdog.py                      # spawn + supervise + probe
    python device/watchdog.py --attach             # supervise a supervisor already running
    python device/watchdog.py --once               # single probe, no restart (CI)
    python device/watchdog.py --interval 45 --deep-every 5

Exit codes: 0 stopped cleanly | 2 probe failed in --once mode |
            7 gave up after --max-restarts
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEVICE_DIR = Path(__file__).resolve().parent
PROJECT = DEVICE_DIR.parent
EVIDENCE = DEVICE_DIR / "evidence"
LOG_DIR = DEVICE_DIR / "logs"
EVIDENCE.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
WATCHDOG_LOG = LOG_DIR / "watchdog.log"
SUPERVISOR_LOG = LOG_DIR / "supervisor.log"
SUPERVISOR_JSONL = LOG_DIR / "supervisor.jsonl"
SUPERVISOR_STDOUT = LOG_DIR / "supervisor.stdout.log"
SUPERVISOR_STDERR = LOG_DIR / "supervisor.stderr.log"

sys.path.insert(0, str(DEVICE_DIR))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import devlog  # noqa: E402

LOG = devlog.get_logger("watchdog", LOG_DIR / "watchdog.jsonl")


# ---------------------------------------------------------------------------
# The decision. Pure: no I/O, no clock, no subprocess. Everything the loop can
# get wrong lives here where a test can reach it.
# ---------------------------------------------------------------------------
# actions
A_OK = "ok"                 # healthy, keep going
A_TOLERATE = "tolerate"     # not healthy, but not a fault (contention)
A_WAIT = "wait"             # failed, but below the fault threshold
A_RESTART = "restart"       # collect evidence and restart
A_GIVEUP = "giveup"         # restarting is not helping; a human must look

FAULT_AFTER = 2             # consecutive real failures before restarting
BUSY_TOLERANCE = 5          # consecutive busy results tolerated before restarting


def decide(http_code, body: dict | None, consecutive_failures: int,
           consecutive_busy: int, restarts_done: int, max_restarts: int,
           fault_after: int = FAULT_AFTER,
           busy_tolerance: int = BUSY_TOLERANCE) -> tuple[str, str]:
    """Return (action, reason) for one probe result.

    Rules, and why each one exists:

    * ok=True                       -> A_OK.
    * busy / HTTP 429               -> A_TOLERATE up to `busy_tolerance`.
      A busy probe means ANOTHER caller holds the model — a long utterance, or
      the judge's own request. Restarting then would kill a working device in
      the middle of the sentence it is translating. But busy forever means
      something is stuck holding a lock, so tolerance is finite.
    * any other failure             -> A_WAIT until `fault_after` in a row, then
      A_RESTART. One failure can be a transient; two in a row is a pipeline.
    * unreachable (http_code None)  -> counts as a real failure. The process is
      gone or wedged.
    * restarts_done >= max_restarts -> A_GIVEUP. A restart loop on a broken model
      file burns minutes per cycle and produces one evidence bundle per cycle;
      after N of them the answer is not another restart.
    """
    body = body or {}
    # A healthy probe always wins, even after the restart budget is spent.
    if http_code == 200 and body.get("ok"):
        return A_OK, "probe ok"

    busy = bool(body.get("busy")) or http_code == 429 or \
        str(body.get("detail") or "").startswith("busy")
    if busy:
        if consecutive_busy < busy_tolerance:
            return A_TOLERATE, (f"pipeline busy ({consecutive_busy + 1}/{busy_tolerance}) "
                                f"— another caller holds the model, not a fault")
        reason = (f"busy for {consecutive_busy + 1} consecutive probes — "
                  f"a lock is stuck, not a long utterance")
        return (_giveup_or(A_RESTART, restarts_done, max_restarts), reason)

    detail = body.get("error") or body.get("detail") or f"http={http_code}"
    if consecutive_failures + 1 < fault_after:
        return A_WAIT, f"probe failed ({detail}); below the {fault_after}-failure threshold"
    return (_giveup_or(A_RESTART, restarts_done, max_restarts),
            f"{consecutive_failures + 1} consecutive probe failures: {detail}")


def _giveup_or(action: str, restarts_done: int, max_restarts: int) -> str:
    if max_restarts >= 0 and restarts_done >= max_restarts:
        return A_GIVEUP
    return action


def backoff_seconds(restarts_done: int, base: float = 5.0, cap: float = 120.0) -> float:
    """Exponential, capped. Restarting instantly in a loop starves the box of the
    CPU the NEXT start needs, and on this hardware a start is 50-100 s of model
    loading — so a tight loop is strictly worse than waiting."""
    return min(cap, base * (2 ** max(0, restarts_done)))


# ---------------------------------------------------------------------------


def http_get(url: str, timeout: float):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


def http_text(url: str, timeout: float) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"<unavailable: {type(e).__name__}: {e}>"


def run(cmd: list[str] | str, timeout: float = 15.0) -> str:
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, timeout=timeout, errors="replace")
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def global_memory() -> str:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return (f"load {m.dwMemoryLoad}%  phys total {m.ullTotalPhys/2**20:.0f} MB "
                    f"avail {m.ullAvailPhys/2**20:.0f} MB  "
                    f"pagefile avail {m.ullAvailPageFile/2**20:.0f} MB")
        except Exception as e:
            return f"<unavailable: {e}>"
    return run(["free", "-m"])


def disk_free() -> str:
    try:
        u = shutil.disk_usage(str(PROJECT))
        return (f"{PROJECT}: total {u.total/2**30:.1f} GiB  used {u.used/2**30:.1f} GiB  "
                f"free {u.free/2**30:.1f} GiB")
    except Exception as e:
        return f"<unavailable: {e}>"


def tail(path: Path, n: int = 40) -> str:
    if not path.is_file():
        return f"<no {path}>"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"<unreadable: {e}>"


def env_snapshot() -> str:
    rows = [f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("VOICEAI_")]
    return "\n".join(rows) if rows else "<no VOICEAI_* variables set in the watchdog's env>"


def collect_evidence(port: int, reason: str, health_code, health_body,
                     child: subprocess.Popen | None) -> Path:
    """Write everything this platform can give us BEFORE we restart. Restarting
    first is how you end up with a demo that broke twice and no idea why.

    The bundle has to answer, without reproducing the failure:
      WHAT failed      -> the probe body, /healthz, the last structured events
      WHERE            -> failed_stage, per-stage readiness and demotions
      WHY (resource)   -> RSS/peak RSS, global memory, disk, process table
      WHY (code)       -> the child's stdout/stderr, which is where a traceback
                          from an unhandled exception actually lands (the
                          supervisor's own log only has records it chose to write
                          — a hard crash is invisible there)
      WITH WHAT CONFIG -> VOICEAI_* environment and the supervisor's argv
    A machine-readable .json is written next to the .txt so a later run can diff
    two failures instead of eyeballing them.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = EVIDENCE / f"evidence-{ts}.txt"
    out_json = EVIDENCE / f"evidence-{ts}.json"
    win = sys.platform.startswith("win")
    base = f"http://127.0.0.1:{port}"

    metrics = http_text(f"{base}/metrics", 10)
    events = http_get(f"{base}/events?n=250", 10)[1]
    proc_table = run('tasklist /FI "IMAGENAME eq python.exe" /V' if win
                     else "ps -eo pid,ppid,rss,pcpu,etime,cmd --sort=-rss | head -25")
    port_owner = run(f'netstat -ano | findstr :{port}' if win
                     else f"ss -lptn 'sport = :{port}'")
    oom = ("<Windows: no dmesg. The equivalent signal is an abrupt process exit with "
           "no traceback plus low ullAvailPhys above; check Event Viewer "
           "'Application Error' if a hard crash is suspected.>" if win
           else run("dmesg | grep -i -E 'killed process|oom' | tail -5"))

    doc = {
        "ts": ts,
        "reason": reason,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "endpoint": base,
        "http_status": health_code,
        "health": health_body,
        "child": ({"pid": child.pid, "returncode": child.poll()} if child else None),
        "memory": global_memory(),
        "disk": disk_free(),
        "port_owner": port_owner.strip().splitlines(),
        "voiceai_env": dict(sorted((k, v) for k, v in os.environ.items()
                                   if k.startswith("VOICEAI_"))),
        "supervisor_events": events.get("events") if isinstance(events, dict) else events,
    }
    try:
        out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    except Exception as e:
        LOG.warn(devlog.E_WD_EVIDENCE, f"json bundle failed: {e!r}")

    parts = [
        f"=== VoiceAI watchdog evidence bundle {ts} ===",
        f"reason              : {reason}",
        f"platform            : {platform.platform()}  python {platform.python_version()}",
        f"probe endpoint      : {base}",
        f"http status         : {health_code}",
        f"machine-readable    : {out_json.name}",
        "",
        "--- last response body ---",
        json.dumps(health_body, ensure_ascii=False, indent=2)[:6000],
        "",
        "--- supervisor child process ---",
        (f"pid={child.pid} returncode={child.poll()}" if child else "<not owned by this watchdog>"),
        "",
        "--- /metrics ---",
        metrics[:6000],
        "",
        "--- process table (python) ---",
        proc_table,
        "",
        "--- memory ---",
        global_memory(),
        "",
        "--- disk ---",
        disk_free(),
        "",
        "--- port owner ---",
        port_owner,
        "",
        "--- kernel OOM evidence ---",
        oom,
        "",
        "--- VOICEAI_* environment (watchdog's view) ---",
        env_snapshot(),
        "",
        "--- supervisor structured events, last 60 (from /events) ---",
        _format_events(doc["supervisor_events"], 60),
        "",
        "--- supervisor.jsonl tail (60) ---",
        tail(SUPERVISOR_JSONL, 60),
        "",
        "--- supervisor.stdout.log tail (80) --- "
        "(a hard crash traceback lands HERE, not in the structured log)",
        tail(SUPERVISOR_STDOUT, 80),
        "",
        "--- supervisor.stderr.log tail (80) ---",
        tail(SUPERVISOR_STDERR, 80),
        "",
        "--- supervisor.log tail (40, legacy text log) ---",
        tail(SUPERVISOR_LOG, 40),
        "",
        "--- watchdog.jsonl tail (30) ---",
        tail(LOG_DIR / "watchdog.jsonl", 30),
    ]
    out.write_text("\n".join(parts), encoding="utf-8")
    LOG.warn(devlog.E_WD_EVIDENCE, "evidence written", path=str(out),
             json_path=str(out_json), reason=reason)
    return out


def _format_events(events, n: int) -> str:
    if not isinstance(events, list):
        return f"<unavailable: {events!r}>"[:400]
    rows = []
    for e in events[-n:]:
        if not isinstance(e, dict):
            continue
        extra = " ".join(f"{k}={v}" for k, v in e.items()
                         if k not in ("ts", "level", "event", "component", "run_id",
                                      "pid", "thread", "uptime_s", "msg"))
        rows.append(f"{e.get('ts')} {str(e.get('level')).upper():<5} "
                    f"{e.get('event'):<20} {e.get('msg', '')} {extra}".rstrip())
    return "\n".join(rows) or "<none>"


def spawn_supervisor(args) -> subprocess.Popen:
    cmd = [sys.executable, "-u", str(DEVICE_DIR / "supervisor.py"),
           "--port", str(args.port), "--target", args.target]
    if args.quick_verify:
        cmd.append("--quick-verify")
    if args.skip_verify:
        cmd.append("--skip-verify")
    LOG.info(devlog.E_WD_RESTART, "launching supervisor",
             cmd=[shlex.quote(c) for c in cmd])
    logf = open(LOG_DIR / "supervisor.stdout.log", "a", encoding="utf-8")
    errf = open(LOG_DIR / "supervisor.stderr.log", "a", encoding="utf-8")
    kw = {}
    if not sys.platform.startswith("win"):
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, stdout=logf, stderr=errf, cwd=str(PROJECT), **kw)


def kill_tree(child: subprocess.Popen | None, pid_hint: int | None) -> None:
    if child and child.poll() is None:
        try:
            if sys.platform.startswith("win"):
                run(["taskkill", "/PID", str(child.pid), "/T", "/F"])
            else:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        except Exception as e:
            LOG.warn(devlog.E_WD_RESTART, f"kill child failed: {e!r}", pid=child.pid)
        try:
            child.wait(timeout=20)
        except Exception:
            pass
    elif pid_hint:
        try:
            if sys.platform.startswith("win"):
                run(["taskkill", "/PID", str(pid_hint), "/T", "/F"])
            else:
                os.kill(pid_hint, signal.SIGKILL)
            LOG.warn(devlog.E_WD_RESTART, "killed external supervisor", pid=pid_hint)
        except Exception as e:
            LOG.warn(devlog.E_WD_RESTART, f"kill external pid failed: {e!r}", pid=pid_hint)


def wait_ready(port: int, timeout: float, child: subprocess.Popen | None = None
               ) -> tuple[bool, float, dict]:
    """Wait until /healthz reports ready AND a real light probe passes.
    Ready is not 'the port answers' — it is 'a forward pass completed'.

    Returns early if the child process exits: waiting 900 s for a process that
    died 3 s in is 15 minutes of a demo spent on nothing.
    """
    t0 = time.monotonic()
    body: dict = {}
    while time.monotonic() - t0 < timeout:
        if child is not None and child.poll() is not None:
            return False, time.monotonic() - t0, {
                "error": f"supervisor exited with code {child.returncode} during warm",
                "exit_code": child.returncode}
        code, body = http_get(f"http://127.0.0.1:{port}/healthz", 10)
        if code == 200 and body.get("ready"):
            pcode, pbody = http_get(f"http://127.0.0.1:{port}/probe?level=light", 120)
            if pcode == 200 and pbody.get("ok"):
                return True, time.monotonic() - t0, pbody
        time.sleep(2.0)
    return False, time.monotonic() - t0, body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VoiceAI device watchdog")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("VOICEAI_DEVICE_PORT", "8731") or "8731"))
    ap.add_argument("--target", default=os.environ.get("VOICEAI_TARGET", "en"))
    ap.add_argument("--interval", type=float, default=45.0,
                    help="seconds between probes (AIBOX timer used 45 s)")
    ap.add_argument("--deep-every", type=int, default=10,
                    help="run the FULL probe (incl. ASR) every Nth cycle; 0 disables")
    ap.add_argument("--attach", action="store_true",
                    help="do not spawn; supervise a supervisor started elsewhere")
    ap.add_argument("--restart-cmd", default="",
                    help="shell command used to restart instead of respawning "
                         "(e.g. 'systemctl restart voiceai-supervisor')")
    ap.add_argument("--once", action="store_true", help="one probe, no restart")
    ap.add_argument("--ready-timeout", type=float, default=900.0,
                    help="how long a cold start may take before we call it failed")
    ap.add_argument("--fault-after", type=int, default=FAULT_AFTER,
                    help="consecutive real probe failures before restarting")
    ap.add_argument("--busy-tolerance", type=int, default=BUSY_TOLERANCE,
                    help="consecutive busy probes tolerated before treating it as stuck")
    ap.add_argument("--max-restarts", type=int, default=5,
                    help="give up after this many restarts (-1 = never give up)")
    ap.add_argument("--quick-verify", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    LOG.info(devlog.E_WD_START, "watchdog start", port=args.port,
             interval_s=args.interval, deep_every=args.deep_every,
             attach=args.attach, fault_after=args.fault_after,
             busy_tolerance=args.busy_tolerance, max_restarts=args.max_restarts)

    child: subprocess.Popen | None = None

    if args.once:
        code, body = http_get(f"http://127.0.0.1:{args.port}/probe?level=light", 120)
        ok = code == 200 and body.get("ok")
        LOG.emit(devlog.E_WD_PROBE, "info" if ok else "error", "single probe",
                 http=code, ok=bool(ok), detail=body.get("detail") or body.get("error"))
        if not ok:
            collect_evidence(args.port, "single --once probe failed", code, body, None)
            return 2
        return 0

    if not args.attach:
        code, _ = http_get(f"http://127.0.0.1:{args.port}/healthz", 5)
        if code is not None:
            LOG.warn(devlog.E_WD_START,
                     "a supervisor already answers on this port; switching to --attach",
                     port=args.port)
            args.attach = True
        else:
            child = spawn_supervisor(args)
            ok, dt, p = wait_ready(args.port, args.ready_timeout, child)
            LOG.emit(devlog.E_WD_PROBE, "info" if ok else "error", "initial warm",
                     ready=ok, after_s=round(dt, 1), mt_ms=p.get("mt_ms"),
                     tts_ms=p.get("tts_ms"), detail=p.get("error"))
            if not ok:
                collect_evidence(args.port, "initial warm timed out or child exited",
                                 None, p, child)
                kill_tree(child, None)
                return 2

    cycle = 0
    consecutive_failures = 0
    consecutive_busy = 0
    restarts_done = 0
    try:
        while True:
            time.sleep(args.interval)
            cycle += 1
            deep = bool(args.deep_every) and cycle % args.deep_every == 0
            level = "full" if deep else "light"
            code, body = http_get(f"http://127.0.0.1:{args.port}/probe?level={level}",
                                  180 if deep else 120)
            action, reason = decide(code, body, consecutive_failures, consecutive_busy,
                                    restarts_done, args.max_restarts,
                                    args.fault_after, args.busy_tolerance)

            if action == A_OK:
                consecutive_failures = consecutive_busy = 0
                LOG.emit(devlog.E_WD_PROBE, "debug", f"probe {level} ok",
                         level=level, asr_ms=body.get("asr_ms"),
                         mt_ms=body.get("mt_ms"), tts_ms=body.get("tts_ms"),
                         degraded=body.get("degraded"))
                continue
            if action == A_TOLERATE:
                consecutive_busy += 1
                LOG.warn(devlog.E_WD_PROBE, reason, level=level, http=code,
                         consecutive_busy=consecutive_busy)
                continue
            if action == A_WAIT:
                consecutive_failures += 1
                LOG.warn(devlog.E_WD_PROBE, reason, level=level, http=code,
                         consecutive_failures=consecutive_failures)
                continue

            consecutive_failures += 1
            hcode, hbody = http_get(f"http://127.0.0.1:{args.port}/healthz", 10)
            LOG.error(devlog.E_WD_FAULT, reason, level=level, http=code,
                      action=action, consecutive_failures=consecutive_failures,
                      consecutive_busy=consecutive_busy, restarts_done=restarts_done)
            collect_evidence(args.port, reason, hcode, hbody or body, child)

            if action == A_GIVEUP:
                LOG.fatal(devlog.E_WD_GIVEUP,
                          "not restarting again — a human must look at the evidence "
                          "bundles in device/evidence/",
                          restarts_done=restarts_done, max_restarts=args.max_restarts)
                return 7

            wait_s = backoff_seconds(restarts_done)
            if wait_s > 0 and restarts_done > 0:
                LOG.warn(devlog.E_WD_BACKOFF, "waiting before restart",
                         seconds=wait_s, restarts_done=restarts_done)
                time.sleep(wait_s)

            t0 = time.monotonic()
            restarts_done += 1
            if args.restart_cmd:
                LOG.warn(devlog.E_WD_RESTART, "restarting via command",
                         cmd=args.restart_cmd)
                LOG.info(devlog.E_WD_RESTART, run(args.restart_cmd, timeout=60)[:2000])
            else:
                pid_hint = (hbody or {}).get("pid")
                kill_tree(child, pid_hint if isinstance(pid_hint, int) else None)
                child = spawn_supervisor(args)

            ok, dt, p = wait_ready(args.port, args.ready_timeout, child)
            total = time.monotonic() - t0
            if ok:
                LOG.info(devlog.E_WD_RECOVERED, "recovered",
                         total_s=round(total, 1), mt_ms=p.get("mt_ms"),
                         tts_ms=p.get("tts_ms"), attempt=restarts_done)
                consecutive_failures = consecutive_busy = 0
            else:
                LOG.error(devlog.E_WD_FAULT, "restart did not reach ready",
                          total_s=round(total, 1), attempt=restarts_done,
                          detail=p.get("error"))
                collect_evidence(args.port, "restart did not reach ready", None, p, child)
    except KeyboardInterrupt:
        LOG.warn(devlog.E_SHUTDOWN, "watchdog stopped by operator")
        if child and child.poll() is None:
            kill_tree(child, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
