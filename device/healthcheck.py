"""healthcheck.py — operator-facing probe client for the resident supervisor.

Three levels, cheapest first:

  --level status   GET /healthz          no inference; is the service ready at all
  --level light    GET /probe?level=light REAL MT + TTS forward pass
  --level full     GET /probe?level=full  REAL ASR + MT + TTS on the pinned wav

`status` alone is NOT a health check — it is exactly the lie that
references/Summer_School_AIEye/aibox_eyes/systemd/geniex-watchdog.sh was written
to defeat (`systemctl is-active` said "active" while the model was OOM-killed).
Use it for a fast glance; use `light`/`full` before you tell a judge it works.

Exit codes: 0 ready | 1 unreachable | 2 not ready / probe failed |
            3 busy (another caller holds the model — retry, do not restart)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

DEFAULT_PORT = int(os.environ.get("VOICEAI_DEVICE_PORT", "8731"))


def fetch(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="probe the VoiceAI device supervisor")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--level", default="status", choices=["status", "light", "full"])
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--json", action="store_true", help="raw JSON only")
    ap.add_argument("--wait", type=float, default=0.0,
                    help="keep retrying for N seconds until ready (boot/recovery)")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    url = f"{base}/healthz" if args.level == "status" else f"{base}/probe?level={args.level}"

    deadline = time.monotonic() + args.wait
    t0 = time.monotonic()
    while True:
        code, body = fetch(url, args.timeout)
        ok = code == 200
        if ok or time.monotonic() >= deadline:
            break
        time.sleep(2.0)
    waited = time.monotonic() - t0

    if args.json:
        print(json.dumps(body, ensure_ascii=False, indent=2))
    elif code is None:
        print(f"UNREACHABLE  {base}  ({body.get('error')})")
        print("  The supervisor is not running (or not on this port). Start it:")
        print("    sh device/run_device.sh          # Linux/ARM target")
        print("    powershell -File device\\run_device.ps1   # Windows dev box")
        return 1
    elif args.level == "status":
        st = body.get("stages", {})
        print(f"state={body.get('state')} ready={body.get('ready')} "
              f"uptime={body.get('uptime_s')}s rss={body.get('rss_mb')}MB "
              f"(peak {body.get('rss_peak_mb')}MB) "
              f"warm_total={body.get('warm_total_ms')}ms")
        if body.get("fault_reason"):
            print(f"  FAULT: {body['fault_reason']}")
        for name, v in st.items():
            if v.get("demoted"):
                flag = "DEMO"
            elif v["ready"]:
                flag = "OK  "
            else:
                flag = "MISS" if v["required"] else "SKIP"
            extra = "" if v["ready"] else f"  ({v.get('demoted_reason') or v.get('error')})"
            print(f"  {flag} {name:<10} load={v.get('load_ms')}ms{extra}")
        for name, why in (body.get("degraded_consequences") or {}).items():
            print(f"  degraded: {name} -> {why}")
        mic = body.get("mic") or {}
        if mic.get("enabled") or mic.get("error"):
            print(f"  mic: running={mic.get('running')} turns={mic.get('turns')} "
                  f"error={mic.get('error')}")
        ka = body.get("keepalive") or {}
        if ka and not ka.get("thread_alive"):
            print("  WARNING: keepalive thread is NOT running — nothing is exercising "
                  "the models between demos.")
        for s, g in (body.get("concurrency") or {}).items():
            if g.get("waits") or g.get("timeouts"):
                print(f"  contention: {s} waits={g['waits']} "
                      f"max_wait={g['max_wait_ms']}ms timeouts={g['timeouts']}")
        for w in body.get("config_warnings") or []:
            print(f"  config WARN: {w}")
        lp = body.get("last_probe") or {}
        print(f"  last_probe: {lp.get('level')} ok={lp.get('ok')} "
              f"asr={lp.get('asr_ms')} mt={lp.get('mt_ms')} tts={lp.get('tts_ms')} "
              f"total={lp.get('total_ms')} at {lp.get('probe_utc')}")
    else:
        print(f"probe({args.level}) ok={body.get('ok')} "
              f"asr={body.get('asr_ms')} mt={body.get('mt_ms')} "
              f"tts={body.get('tts_ms')} total={body.get('total_ms')}ms")
        if body.get("busy"):
            print("  BUSY: another caller holds the model (mic turn or a concurrent "
                  "request). This is contention, not a fault — retry.")
        if body.get("degraded"):
            print(f"  degraded stages: {body['degraded']}")
        if body.get("detail"):
            print(f"  detail: {body['detail']}")

    if args.wait:
        print(f"  (waited {waited:.1f}s)")

    if code is None:
        return 1
    if args.level == "status":
        return 0 if body.get("ready") else 2
    if body.get("ok"):
        return 0
    return 3 if (body.get("busy") or code == 429) else 2


if __name__ == "__main__":
    sys.exit(main())
