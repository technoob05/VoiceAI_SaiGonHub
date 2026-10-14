#!/bin/sh
# max_performance.sh — pin the Linux/ARM demo box to its highest sustained clock.
#
# Adapted from references/Summer_School_AIEye/aibox_eyes/max_performance.sh.
# What carried over: CPU governor = performance with min = max, all cores online,
# deep C-states disabled, our process reniced. What did NOT carry over, and is
# deliberately absent:
#
#   * Adreno GPU (/sys/class/kgsl/kgsl-3d0) — this pipeline runs CTranslate2 INT8
#     and onnxruntime on the CPU. There is no GPU work to pin.
#   * DDR/bus devfreq pinning — kept, it is generic, but the payoff here is much
#     smaller than on the AIBOX: our bottleneck is CT2 GEMM on CPU, not NPU
#     weight streaming.
#   * QNN/HTP performance mode — we have no Hexagon path in code yet
#     (prototype/DEPLOYMENT_ROADMAP.md documents it as future work). Nothing to set.
#
# !!! UNTESTED !!! No Linux/ARM device was available; this was written on
# Windows. Every write is guarded and failures are non-fatal, but the *effect*
# has not been measured. Do not quote a speedup from this file.
#
# Run:   sudo sh device/max_performance.sh
#
# WARNING: pinning clocks raises temperature and power draw. Leave the thermal
# daemon running — the AIEye notes are right that a box which thermally trips
# dies mid-demo, which is worse than one that runs a little slower.

set -u
log() { echo "[maxperf] $*"; }

if [ "$(id -u)" != "0" ]; then
  echo "[maxperf] must run as root (sudo sh device/max_performance.sh)" >&2
  exit 1
fi

log "--- 1. CPU governor = performance, min = max ---"
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  [ -d "$p" ] || continue
  MAXF=$(cat "$p/cpuinfo_max_freq" 2>/dev/null) || continue
  echo performance > "$p/scaling_governor" 2>/dev/null
  echo "$MAXF"     > "$p/scaling_max_freq" 2>/dev/null
  echo "$MAXF"     > "$p/scaling_min_freq" 2>/dev/null
  log "$(basename "$p"): gov=$(cat "$p/scaling_governor" 2>/dev/null) cur=$(cat "$p/scaling_cur_freq" 2>/dev/null)"
done

log "--- 2. all cores online ---"
for c in /sys/devices/system/cpu/cpu[0-9]*/online; do
  echo 1 > "$c" 2>/dev/null
done
log "online: $(cat /sys/devices/system/cpu/online 2>/dev/null)"

log "--- 3. disable deep CPU idle states ---"
# Exit latency from a deep C-state is hundreds of microseconds. Our workload is
# bursty by construction: one VAD-closed turn wakes ASR, MT and TTS back to back.
N=0
for s in /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state[1-9]*/disable; do
  [ -f "$s" ] || continue
  echo 1 > "$s" 2>/dev/null && N=$((N+1))
done
log "disabled $N idle states"

log "--- 4. DDR/bus devfreq to max ---"
for d in /sys/class/devfreq/*; do
  [ -d "$d" ] || continue
  case "$(basename "$d")" in *kgsl*) continue ;; esac
  if [ -f "$d/available_governors" ] && grep -q performance "$d/available_governors" 2>/dev/null; then
    echo performance > "$d/governor" 2>/dev/null && log "$(basename "$d"): gov=performance"
  elif [ -f "$d/max_freq" ] && [ -f "$d/min_freq" ]; then
    cat "$d/max_freq" > "$d/min_freq" 2>/dev/null && log "$(basename "$d"): pinned $(cat "$d/cur_freq" 2>/dev/null)"
  fi
done

log "--- 5. raise supervisor priority ---"
# CTranslate2 spawns cpu_count-2 intra-op threads (pipeline_fast.py
# FastPipelineConfig.ct2_intra_threads). We renice, but we do NOT put them on a
# realtime FIFO class: a runaway RT thread on a single-board computer with no
# console is unrecoverable, and the win over nice -10 is small for a
# throughput-bound GEMM.
for pid in $(pgrep -f "device/supervisor.py" 2>/dev/null); do
  renice -n -10 -p "$pid" >/dev/null 2>&1 && log "supervisor(pid $pid): nice=-10"
done

log "--- 6. swap ---"
# A swap-out of a 620 MB resident CT2 model during a pause is exactly the class
# of failure the AIEye keepalive was fighting. Discourage it, do not disable
# swap outright (that turns memory pressure into an OOM kill instead).
if [ -f /proc/sys/vm/swappiness ]; then
  echo 10 > /proc/sys/vm/swappiness 2>/dev/null && log "vm.swappiness=10"
fi

log "DONE. Check temperature: cat /sys/class/thermal/thermal_zone*/temp"
