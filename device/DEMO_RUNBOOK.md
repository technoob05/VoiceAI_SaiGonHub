# Demo runbook — VoiceAI on the portable device

Operator document. Follow it in order; every check has a **numeric pass
criterion**. Where a number comes from a measurement, the source is named. Where
a number is a target rather than an observation, it says so.

Judging weights this document serves: **Technical Performance 50%** — accuracy,
latency, speed, **stability** (`CLAUDE.md` §1).

---

## 0. What must be true before the room fills

The device runs **one** resident process that owns every model
(`device/supervisor.py`). Nothing else may be started: not `desktop_app.py`, not
`web_demo_edge.py`, not a second `live_translate.py`. Each of those loads its own
copy of ~1.6 GB of models, and the box measured here had **355 MB of free RAM at
its worst moment** (evidence bundle `evidence-20260820-073329.txt`). Two copies
do not fit. The microphone is the one exception, and only because it now runs
inside the supervisor: `python device/supervisor.py --mic` (§4.11).

**Start the service at least 10 minutes before the demo.** Cold start is
minutes, not seconds (§1).

---

## 1. Boot and warm — numeric pass criteria

```powershell
# Windows dev box / rehearsal
powershell -ExecutionPolicy Bypass -File device\run_device.ps1 -Target en
```
```sh
# Linux/ARM target (UNTESTED — see §6)
sh device/run_device.sh --target en --watchdog
```

The launcher prints one of two lines, and only one of them means go:

```
DEVICE READY - PASS.  http://127.0.0.1:8731/healthz   target=en
DEVICE NOT READY - FAIL (rc=N). Do not start the demo.
```

### What the device is measured to do (Kaggle T4, not this box, not the target device)

These come from `../Results/kaggle_t4/README.md`. Quote them as *development GPU*
numbers; the Snapdragon figure does not exist yet and must not be implied.

| | |
|---|---|
| Directions measured | **all six** — vi↔en, vi↔zh, vi↔ko |
| RTF < 1.0 | **360/360** utterances |
| End-to-end < 2 s | **360/360** utterances |
| End-to-end p50 | 443 ms (vi→en) to 1108 ms (vi→zh) |
| ASR, Vietnamese | WER 0.1259 (PhoWhisper-small INT8) |
| ASR, English | WER 0.0663 |
| ASR, Korean | CER 0.0732 (read CER, not WER — Korean spacing is optional) |
| ASR, Mandarin | CER 0.1229 with the Simplified prompt (0.2216 without) |
| Peak GPU | 2897 MiB with both ASR models resident |

If a judge asks which direction is weakest, the honest answer is **vi→zh**: it is
the slowest (Mandarin TTS is ~6× the cost per second of audio of the other
voices) and Mandarin ASR has the highest error rate of the four languages.

### Pass criteria at boot

| Check | Pass | Measured on the dev box (x86-64, 8 GB RAM, contended) | Source |
|---|---|---|---|
| Manifest verification | 46/46 assets, no WARN lines | 2.6–5.7 s for ~2.1 GiB (285–633 MiB/s) | `provenance.py verify`, this box |
| All five stages `OK` | asr, mt, tts, frontend `OK`; corrector `OK` or `SKIP` | all OK | `healthcheck.py --level status` |
| `warm_total` | < 300 s on this class of machine | **50.4 – 96.8 s** across five successful cold starts (57.2 s most recent) | supervisor log |
| Acceptance probe (full) | `ok=True` | 11.2 / 14.9 / 16.2 / 26.2 / 44.9 s (dominated by the first real ASR decode, §4.6) | supervisor log |
| Peak RSS after ready | 1.2–1.8 GB | **1385–1743 MB** (`rss_peak_mb`, all five stages warm) | `/healthz` |

> **Do not alarm on `rss_mb` alone.** Two minutes idle, `/healthz` reported
> `rss=12.3MB` while the same process still owned 1.6 GB of models — Windows had
> trimmed the working set. That is why `/healthz` also reports `rss_peak_mb`, and
> why the supervisor runs a keepalive probe every 120 s: trimmed pages have to be
> faulted back in, and that fault lands on whoever speaks next.

Stage load times observed across five cold starts on this box (seconds):

| Stage | A (cold OS cache) | B | C (watchdog restart) | D (`run_device.ps1`) | E (`run_device.ps1`) |
|---|---:|---:|---:|---:|---:|
| ASR (PhoWhisper-small INT8) | 24.9 | 29.8 | 26.7 | 18.8 | 19.7 |
| corrector (vietnamese-correction-v2 INT8) | 22.0 | 17.2 | 18.6 | 11.0 | 7.7 |
| MT (NLLB-600M INT8) incl. phrase pre-tokenize | 98.5 | 35.2 | 25.4 | 28.1 | 17.8 |
| TTS (Piper `en_US-lessac-medium`) | 35.1 | 8.5 | 5.1 | 2.8 | 1.7 |
| frontend (Silero VAD + GTCRN) | — (failed, §4.3) | 0.2 | 0.5 | 0.2 | 0.2 |
| **warm total** | — | 96.8 | 84.7 | 63.8 | **50.4** |

Run A hit a 49.8 s industrial-phrase pre-tokenize; the others hit 12–25 s for
the same step. The spread is OS page cache plus a machine that was 60–75 % busy
with unrelated work. **Treat the high end as the number to plan around.**

For reference, the same cascade on a Kaggle T4 loads in **13.5 s total**
(ASR 2.76 s, corrector 2.72 s, MT 5.82 s, TTS 2.24 s —
`Results/kaggle_t4/README.md`). The dev box is 6–7× slower at loading. The ARM
target is **unmeasured**.

---

## 2. Pre-flight, 3 minutes before you speak

Run these three, in order. Do not skip to the demo if any fails.

```powershell
python device\healthcheck.py --level status      # 1. is it ready at all
python device\healthcheck.py --level light       # 2. real MT + TTS forward pass
python device\healthcheck.py --level full        # 3. real ASR + MT + TTS on a pinned wav
```

| # | Pass criterion | Measured here | Why this number |
|---|---|---|---|
| 1 | `ready=True`, all stages `OK`, `uptime` > 0 | ready=True | `status` alone proves only that the process answers — it is the check that lies. See §3. |
| 2 | `ok=True`; `mt` < 15 000 ms | 494 / 1 338 / 2 053 / 3 072 / 3 592 / 11 031 ms | On T4 the same call is 56 ms p50; on this contended CPU box the spread is 2–11 s. Anything over 15 s means the machine is thrashing. |
| 3 | `ok=True`; `asr` < 40 000 ms for the 4.06 s clip | 10 692 / 12 740 / 13 541 / 13 856 / 22 955 / 26 417 / 26 645 / 32 536 ms | RTF 2.6–8 on this box, i.e. **not real-time here**. T4 warm ASR is 167 ms p50 for 3.23 s mean audio (RTF 0.132). |
| 3b | translation text is non-empty and sane | `"The strange crackling of the iron joints."` | An empty or garbage string means the ASR is alive but decoding noise. |

### One live sentence, out loud

```powershell
python - <<'PY'
import json,urllib.request,sys
sys.stdout.reconfigure(encoding='utf-8')
b=json.dumps({"text":"Dừng dây chuyền ngay lập tức.","target":"en"}).encode("utf-8")
r=urllib.request.Request("http://127.0.0.1:8731/translate",data=b,
                         headers={"Content-Type":"application/json"})
print(urllib.request.urlopen(r,timeout=180).read().decode("utf-8"))
PY
```

Measured on this box: `"Stop the production line immediately."`,
`mt_ms=1295`, `tts_ms=2628`, `audio_s=2.01`.

**Pass:** translation is correct and `audio_s` > 0.5.
Do **not** send Vietnamese through `curl` in Git Bash — it re-encodes the body
and the server answers `UnicodeDecodeError` (observed). Use the snippet above.

### Latency the judges will actually feel

The competition target is **< 1.5 s end-to-end per utterance**, stretch < 800 ms
(`CLAUDE.md` §3). The numbers that back that claim are the warm T4 numbers
(**p50 400.7 ms, p95 563.3 ms, max 751.4 ms, RTF p50 0.129, 300/300 utterances
under 2 s** — `Results/kaggle_t4/README.md`) and the streaming first-audio
numbers on warm Piper CPU (**98 ms vs 876 ms, 8.9×**, later 791 → 627 ms for the
fast first fragment — `prototype/EDGE_UPGRADES.md`).

**This dev box does not reproduce them** (ASR alone is 11–33 s per 4 s clip
here, i.e. RTF 2.6–8). If the demo runs on a machine like this one, say so; do not quote the T4
table as if it were the device.

---

## 3. What "healthy" means, and the check that lies

`healthcheck.py --level status` asks the supervisor whether it *thinks* it is
ready. That is the same class of statement as `systemctl is-active`, which on
the AIBOX reported `active` for a VLM that had been OOM-killed minutes earlier
(`references/Summer_School_AIEye/aibox_eyes/systemd/geniex-keepalive.conf`).

`--level light` and `--level full` run **real forward passes**. Only those two
prove the models are alive. The watchdog uses them, not `status`.

---

## 4. Triage

### 4.1 `REFUSING TO START: port 8731 is already in use` (exit 3)
Verified on this box. A supervisor is already running — that is usually good
news. Confirm with `python device\healthcheck.py --level light`. Only if it is a
stale process: stop that PID, then relaunch. **Never run two supervisors**; they
will fight for CPU threads and RAM, and this box only has room for one.

### 4.2 `*** MANIFEST VERIFICATION FAILED ***` (exit 5)
Verified by tampering with a copy of the manifest:
```
data/raw/models_ct2/phowhisper-small-int8/model.bin: SHA256 25c9e77010ce2237… != 0000000000000000…
```
A model file changed or was truncated in transit. Re-copy `data/raw/` from the
master copy and re-run. Do **not** start with `--skip-verify` to "get through the
demo" — a truncated CT2 model does not fail at load, it fails inside the first
inference, in front of the judges. If you are certain the change is intentional
(you converted a new model), regenerate: `python device/provenance.py generate`.

### 4.3 `DEVICE NOT READY — stage(s) failed to warm: [...]` (exit 4)
Observed for real during development: the frontend stage raised
`AttributeError("'GtcrnDenoiser' object has no attribute 'process'")` and the
supervisor refused to report ready. That is the design working — the failure
surfaced at boot, not on stage. Read `device/logs/supervisor.log`, fix, relaunch.

### 4.4 `corrector` shows `SKIP` / "DEGRADED: corrector unavailable"
Not fatal. The pipeline falls back to the raw ASR transcript, exactly as
`prototype/live_translate.py` does. Expect slightly rougher Vietnamese text. On
the T4 run the corrector changed the transcript in only **6 of 300** utterances
and was rejected by the deletion guard in 5 more
(`Results/kaggle_t4/README.md`), so the demo survives without it.

### 4.5 Probe returns `busy: inference lock timeout`
One inference at a time by design. A long utterance is in flight. The watchdog
tolerates a single busy result before it treats it as a fault. If it repeats,
something is stuck — collect the evidence bundle and restart.

### 4.6 First real utterance is far slower than the rest
Expected. The project's ASR prewarm feeds **silence** (`pipeline_fast._asr_prewarm`,
520–827 ms measured), which pays the load but not the full decode path. The
first real decode measured 23.8 s here versus 23.0–26.6 s afterwards — i.e. the
acceptance probe in step 7 of startup absorbs it. **Never skip the acceptance
probe.** If you restart the service, run `--level full` once before demoing.

### 4.7 Third target language is slow
The pipeline keeps a **2-slot LRU of Piper voices**
(`prototype/EDGE_UPGRADES.md`: "exactly 2 are live per session"). Demo Vi→En and
one other language. Switching to a third evicts a voice and pays a reload
(2.24 s on T4; 5.1–35.1 s measured here).

### 4.8 Everything is suddenly slow / latencies double
Memory pressure. The evidence bundle captured **`load 95%  phys total 7877 MB
avail 355 MB`** at the moment of the fault. Close editors and browsers; on
Linux check `dmesg | grep -i 'killed process'`. `device/max_performance.sh` sets
`vm.swappiness=10` for this reason (untested).

### 4.9 Vietnamese output is mojibake in the terminal
Windows console codepage. All device scripts call `reconfigure(encoding="utf-8")`;
if you pipe them through another tool, that tool is the problem. Use
`--json` output and read the file.

### 4.10 Vi→Zh and Vi→Ko translations are visibly weaker than Vi→En
Known model-capacity limit, documented with examples in
`prototype/EDGE_UPGRADES.md` ("compressor → air temperature", "production line →
*circuit*"). Beam 4 does not fix it and costs 30–40 % latency. If a judge speaks
Chinese or Korean, lead with Vi→En and say this openly.

### 4.11 Run the mic INSIDE the supervisor, never beside it
Start the mic with `python device/supervisor.py --mic`, which runs the live loop
on the models this process already warmed. Measured: constructing the live
translator against the resident pipeline costs **+1 MB** RSS and hands back the
same pipeline object.

Starting `prototype/live_translate.py mic` separately instead loads a **second**
full copy of every model. That is the failure this flag exists to prevent — on
the box measured here it would not fit. If `--mic` cannot open an audio device it
logs the error, sets `mic.error` in `/healthz`, and leaves HTTP translation
working rather than taking the device down.

Useful mic flags, all passed straight through to `live_translate`:
`--pair vi,ko`, `--source vi` (`--manual-source` remains an alias; this is what a physical push-to-talk button sets;
detection can never override it), `--multilingual-asr` (route en/zh/ko turns to
the second ASR — costs another ~486 MB and its zh/ko accuracy is unmeasured),
`--no-play`.

---

## 5. Watchdog and recovery — measured

```sh
python device/watchdog.py --attach --interval 45 --deep-every 10
```

Every interval it runs a **real** light probe; every 10th cycle a full probe
including ASR. On failure it writes an evidence bundle to `device/evidence/`
**before** restarting — process table, global memory, port owner, kernel OOM
evidence (Linux) or a stated Windows equivalent, plus 60 lines of
`supervisor.log`. Then it restarts and waits for a real re-warm.

**Measured recovery, this box:** supervisor killed at 07:33:08 with
`Stop-Process -Force`.

| Phase | Time |
|---|---|
| detection (probe interval 20 s in this test) | 19 s |
| evidence bundle written | 5 s |
| restart + full re-warm + verifying probe | 115 s |
| **total, kill → serving again** | **139.3 s** |

Watchdog log line: `RECOVERED in 139.3s (ready+re-warmed; verifying probe
mt=3591.9ms tts=0.6ms)`.

**What that means for a demo:** a crash costs about **2.5 minutes** on this
hardware, most of it model loading. Have a slide ready to talk over. With the
default 45 s interval, detection alone can add up to 45 s. If you want faster
detection during judging, run with `--interval 20`.

---

## 6. What is NOT tested — read before trusting anything here

| Item | Status |
|---|---|
| `device/run_device.sh` | **Never executed.** No Linux/ARM box available. Passes `sh -n` (syntax only) in Git Bash; that proves nothing about behaviour. |
| `device/max_performance.sh` | **Never executed.** Needs root and Linux `sysfs`. Passes `sh -n`. No speedup is claimed from it. |
| `device/systemd/*` | **Never executed.** Windows has no systemd. Boot persistence on the target is untested. |
| ARM/Linux latency, memory, thermals | **Unmeasured.** Every number in this document is x86-64 Windows or Kaggle T4. |
| Qualcomm Hexagon/QNN path | **Not implemented at all** — documented as future work in `prototype/DEPLOYMENT_ROADMAP.md`. Nothing in `device/` touches the NPU. |
| Microphone path through the supervisor | **Implemented** (`--mic`, §4.11); verified to reuse the resident models (+1 MB RSS, same object). **Not** verified against real microphone hardware in this environment. |
| Half-duplex own-speaker guard | Implemented in `live_translate`; **never verified acoustically** — no measurement that closing the mic stops the device hearing its own speaker. |
| Multi-hour soak | **Runner implemented, not run.** `device/acceptance.py` records bounded p50/p95/p99, restarts, degraded/busy ratios and RSS growth while the watchdog remains active. No multi-hour ARM evidence exists yet. |
| Windows boot persistence | **Not scripted.** Use Task Scheduler manually if you need it. |

Anything in this file that is not marked with a source and a measured value is a
target, not an achievement. That distinction is the difference between a
technical-performance score and a retraction in front of judges.
