# device/ — turning the prototype into something that survives a live demo

This directory adds the operational layer the project did not have: one resident
always-warm service, a health check that actually probes the models, a watchdog
that collects evidence before it restarts anything, launchers for both
platforms, a hashed model manifest, and an operator runbook.

It adds **no new models and no new pipeline code**. Everything here calls
`prototype/pipeline_fast.py` and reuses that module's own prewarm paths
(`_asr_prewarm`, `TranscriptCorrector.prewarm`, the first `translate` /
`synthesize` calls that `prototype/live_translate.py` already performs at
startup).

| File | What it is |
|---|---|
| `supervisor.py` | Resident service. Loads every model once, never unloads, prewarms each stage with a REAL inference, then serves `/healthz`, `/probe`, `/metrics`, `POST /translate` on 127.0.0.1. |
| `healthcheck.py` | Operator/CI client. `--level status` (no inference), `light` (real MT+TTS), `full` (real ASR+MT+TTS on a pinned VIVOS wav). |
| `acceptance.py` | Bounded acceptance/soak gate. Samples readiness plus real probes, computes p50/p95/p99, restart and RSS-growth signals, writes atomic JSON evidence, and returns a CI-safe exit code. |
| `watchdog.py` | Probes on an interval; on failure writes an evidence bundle to `evidence/` **before** restarting, then restarts and waits for a real re-warm. |
| `provenance.py` | Generates and verifies `models.manifest.json` (SHA-256 + size for every artifact the pipeline loads). |
| `models.manifest.json` | 46 assets, 2087.7 MiB, hashed against the real files on this dev box. |
| `run_device.sh` | Linux/ARM launcher. **Untested** — no ARM box was available. |
| `run_device.ps1` | Windows dev-box launcher. Same order, same pass/fail line. Run. |
| `max_performance.sh` | CPU governor / C-state / devfreq pinning for the Linux target. **Untested.** |
| `systemd/` | Boot persistence + watchdog timer for the Linux target. **Untested** (Windows has no systemd). |
| `DEMO_RUNBOOK.md` | Operator pre-flight with numeric pass criteria, plus triage per failure mode. |

## Quick start (Windows dev box — this is the path that has been run)

```powershell
python device\provenance.py verify              # 46/46 assets
powershell -ExecutionPolicy Bypass -File device\run_device.ps1
python device\healthcheck.py --level full
```

Run a short operational gate after startup:

```powershell
python device\acceptance.py --probe-level light --iterations 10 --interval-s 2
```

Run a six-hour soak with strict availability, no degraded voice stage, a 2 s
p95 budget, no supervisor restart, and at most 256 MB of observed RSS/peak-RSS
growth:

```powershell
python device\acceptance.py --probe-level light --iterations 360 `
  --interval-s 60 --max-probe-p95-ms 2000 --max-rss-growth-mb 256
```

Every run writes `device/evidence/acceptance-*.json`, including partial evidence
on Ctrl+C. Exit `0` is pass, `2` is a failed gate, `3` means no HTTP response was
received, and `130` is interrupted. A pass certifies only that observed service
session; it is not Snapdragon/QNN/Hexagon, thermal, power, audio-hardware, or
translation-quality validation.

Across five measured starts, warm readiness took 50.4–96.8 s and the full
acceptance probe took 11.2–44.9 s on the contended Windows dev box (see
`DEMO_RUNBOOK.md`). That is why the service is resident: the cost is paid once,
before anyone is watching.

## The one property that matters

Model residency. Every measured latency in `Results/kaggle_t4/README.md`
(end-to-end p50 400.7 ms) is a **warm** number; the same pipeline cold costs
13.5 s of model loading on a T4 and far more on CPU. Any architecture where a
click spawns a process is a demo that stutters at the worst moment. The
supervisor exists to make "warm" the only state the device is ever in.

## Microphone

Run the live mic loop inside this process so it uses the models already warmed
here:

```
python device/supervisor.py --mic --pair vi,en --source vi --multilingual-asr
```

The source must be pinned before ASR. The T4 bidirectional benchmark has a known
source label per utterance; it does not prove automatic first-turn routing on a
live microphone. A physical push-to-talk control should update the source pin
before it forwards audio.

GTCRN and both experimental caches are off by default. Real DEMAND evidence found
significant GTCRN harm in 5/12 cells, and the legacy cache paths are not proven
safe for tonal minimal pairs/negation/numerals. See `docs/SETUP_SUBMISSION.md`.

Do **not** start `prototype/live_translate.py mic` alongside the supervisor: it
loads a second full copy of every model. Running the loop on the resident
pipeline costs +1 MB RSS instead (measured, same pipeline object). If no audio
device can be opened the loop reports the error in `/healthz` under `mic.error`
and HTTP translation keeps working.
