#!/bin/sh
# run_device.sh — ONE launcher for the portable Linux/ARM demo box.
#
# Same job as references/Summer_School_AIEye/aibox_eyes/run.sh: encode the
# mandatory startup order in one place so nobody has to remember it at 9am on
# demo day, refuse to start on a resource conflict, and prewarm with a REAL
# inference before printing a readiness line.
#
#   1. refuse if something already owns the port (a second supervisor would
#      fight the first one for CPU threads and RAM)
#   2. optional: pin CPU performance (device/max_performance.sh, needs root)
#   3. verify the model manifest (SHA-256) — fail loudly, do not load
#   4. start the supervisor: loads every model once, never unloads, prewarms
#      ASR -> corrector -> MT -> TTS -> frontend with real inferences
#   5. block until a REAL end-to-end probe passes, then print PASS/FAIL
#   6. optionally start the watchdog (--watchdog)
#
# Usage: sh device/run_device.sh [--port 8731] [--target en|zh|ko] \
#            [--quick-verify] [--maxperf] [--watchdog]
#
# !!! UNTESTED ON LINUX/ARM !!!  Written and reviewed on a Windows dev box; the
# Python it drives (supervisor.py / provenance.py / watchdog.py) has been run
# there, this shell wrapper has not. Nothing in it is Qualcomm-specific.
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
PROJECT=$(cd "$HERE/.." && pwd)
PORT=8731
TARGET=en
VERIFY_FLAG=""
MAXPERF=0
WATCHDOG=0
PY=${PYTHON:-python3}

while [ $# -gt 0 ]; do
  case "$1" in
    --port)   PORT="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --quick-verify) VERIFY_FLAG="--quick-verify"; shift ;;
    --skip-verify)  VERIFY_FLAG="--skip-verify"; shift ;;
    --maxperf)  MAXPERF=1; shift ;;
    --watchdog) WATCHDOG=1; shift ;;
    *) echo "unknown argument: $1"; exit 2 ;;
  esac
done

cd "$PROJECT"
mkdir -p device/logs device/evidence

echo "== 0. resource conflict check (port $PORT) =="
# The supervisor also claims the port itself before loading models; this is the
# friendlier, earlier version of the same refusal.
if command -v ss >/dev/null 2>&1; then
  BUSY=$(ss -lptn "sport = :$PORT" 2>/dev/null | tail -n +2)
elif command -v netstat >/dev/null 2>&1; then
  BUSY=$(netstat -lptn 2>/dev/null | grep ":$PORT ")
else
  BUSY=""
fi
if [ -n "$BUSY" ]; then
  echo "   REFUSING TO START: something already listens on $PORT:"
  echo "   $BUSY"
  echo "   Check it first:  $PY device/healthcheck.py --port $PORT"
  exit 3
fi
echo "   OK"

if [ "$MAXPERF" = "1" ]; then
  echo "== 0b. CPU performance pinning =="
  if [ "$(id -u)" != "0" ]; then
    echo "   SKIPPED: needs root (run: sudo sh device/max_performance.sh)"
  else
    sh "$HERE/max_performance.sh" || echo "   WARNING: max_performance.sh failed; continuing"
  fi
fi

echo "== 1-6. supervisor: verify manifest, load models once, prewarm each stage =="
echo "   (cold start is minutes, not seconds — measured 250 s on the x86 dev box;"
echo "    ARM is untested. Start this BEFORE the judges arrive.)"
LOG="$PROJECT/device/logs/supervisor.stdout.log"
nohup "$PY" -u device/supervisor.py --port "$PORT" --target "$TARGET" $VERIFY_FLAG \
      >>"$LOG" 2>&1 &
SUP_PID=$!
echo "   supervisor pid=$SUP_PID  log=$LOG"

echo "== 7. waiting for a REAL end-to-end probe to pass =="
# Not "is the process alive" and not "does the port answer" — healthcheck.py
# --level full runs ASR + MT + TTS forward passes. See the AIEye watchdog for
# why the weaker checks are worthless.
if "$PY" device/healthcheck.py --port "$PORT" --level status --wait 900 >/dev/null 2>&1; then
  "$PY" device/healthcheck.py --port "$PORT" --level full --timeout 300
  RC=$?
else
  RC=1
fi

if [ "$RC" = "0" ]; then
  echo "----------------------------------------------------------------------"
  echo "DEVICE READY — PASS.  http://127.0.0.1:$PORT/healthz   target=$TARGET"
  echo "Pre-flight numbers and triage: device/DEMO_RUNBOOK.md"
  echo "----------------------------------------------------------------------"
else
  echo "----------------------------------------------------------------------"
  echo "DEVICE NOT READY — FAIL (rc=$RC). Do not start the demo."
  echo "Look at: $LOG   then device/DEMO_RUNBOOK.md section 'Triage'"
  echo "----------------------------------------------------------------------"
  exit 4
fi

if [ "$WATCHDOG" = "1" ]; then
  echo "== 8. watchdog (attach mode, probe every 45 s) =="
  nohup "$PY" -u device/watchdog.py --port "$PORT" --target "$TARGET" --attach \
        --interval 45 --deep-every 10 >>"$PROJECT/device/logs/watchdog.stdout.log" 2>&1 &
  echo "   watchdog pid=$!  log=device/logs/watchdog.log"
fi
