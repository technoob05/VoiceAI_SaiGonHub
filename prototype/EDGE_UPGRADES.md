# Edge upgrades — fused from top open-source repos

Comprehensive app-quality upgrade absorbing techniques from 6 reference projects
(see `../references/NOTICE.md` for full attribution + licenses). Everything below
is 100% offline, on-device, and wired into `pipeline_fast.py` behind env flags so
it composes with the existing INT8 cascade + caches.

## New modules

| File | Capability | Absorbed from | Verified |
|---|---|---|---|
| `local_tts_piper.py` | Offline Piper/MeloTTS TTS (sherpa-onnx), all 4 langs incl. zh | my-translator | vi/en/zh/ko synth offline; closes MMS Mandarin gap |
| `streaming_tts.py` | First-audio-fast fragment synth + faded concat + backend fallback | RealtimeTTS, my-translator | **98 ms vs 876 ms first-audio (8.9×)** on warm Piper |
| `streaming_asr.py` | LocalAgreement-2 incremental ASR (stable, non-retracting partials) | whisper_streaming | committed text never retracts; buffer-trim bounds compute |
| `edge_frontend.py` | Silero learned-VAD confirm + GTCRN denoiser (ONNX) | RealtimeSTT, sherpa-onnx | **denoise +10.4 dB SNR** (5→15.4 dB, RTF 0.27); VAD rejects broadband noise |
| `live_translate.py` | Always-warm app: mic/file/sidecar → VAD-segment → translate → stream | RTranslator, my-translator, sherpa | file mode end-to-end OK (vi→en) |

## Env flags (compose freely)

```bash
VOICEAI_TTS_BACKEND=piper     # offline Piper/MeloTTS (else "mms" = MMS-TTS torch testbed)
VOICEAI_STREAMING_TTS=1       # fragment-by-fragment TTS, first-audio-fast
VOICEAI_DENOISE_GTCRN=1       # GTCRN industrial-noise front-end (else spectral subtraction)
VOICEAI_SILERO_CONFIRM=1      # learned-VAD confirm after cheap tone-energy gate
```

## Run the app

```bash
# Live microphone (needs a mic):
python live_translate.py mic --target en --source vi

# File (no mic) — same turn→translate→stream path:
python live_translate.py file --wav <vi.wav> --target en --source vi

# Sidecar: raw s16le/16k/mono PCM on stdin → JSON events on stdout
#   (the seam a Tauri / on-device host process plugs into; models load once)
some_capture | python live_translate.py sidecar --target zh --source vi
```

## Standalone demos

```bash
python streaming_tts.py "Xin chào. Hôm nay trời đẹp." --target vi --backend piper
python streaming_asr.py <vi.wav> --lang vi --chunk 1.0
python edge_frontend.py denoise <noisy.wav> --out clean.wav
python edge_frontend.py vad <speech.wav>
```

## Round 2 additions (turn-taking, streaming-ASR maturity, packaging)

| File / change | What | Absorbed from | Verified |
|---|---|---|---|
| `live_translate.py` — pre-roll + min-speech | Ring buffer splices ~300 ms before each VAD turn (recover clipped VN onsets); drops <200 ms noise blips | RealtimeSTT | clean partials + segmentation intact |
| `streaming_asr.py` — `VACOnlineASR` | VAD-gated decoding: skips Whisper on silence, finalizes on trailing silence, counts decodes; `OnlineASR` untouched (backward-compat) | whisper_streaming VAC | backward-compat clean; VAC decodes=3 on 4 s clip (win is on long silence-heavy audio) |
| `streaming_asr.py` — `stream_wav_adaptive` | self-adaptive chunk pacing (chunk grows under load) | whisper_streaming | — |
| `desktop_app.py` + `run_edge_app.bat` + `edge_app.spec` | native pywebview window (no browser) + one-click `.bat` + PyInstaller spec | — | no-window run HTTP 200 |
| `web_demo_edge.py` — caches OFF + single warm pipeline | fixed "Connection lost" (per-combo 130 s reload) and `tm_full_hit` stale-translation bug | — | combo-switch 0.9 s; launch HTTP 200 |
| `DEPLOYMENT_ROADMAP.md` | honest Hexagon/QNN path + PhoWhisper→ONNX→QNN + "no VN Zipformer" finding | RTranslator, sherpa-onnx | — |

**Run the desktop app:** double-click `run_edge_app.bat` (or `python desktop_app.py`).

## Round 3 additions (punctuation, barge-in, native mic)

| File / change | What | Verified |
|---|---|---|
| `punct_restore.py` + `VOICEAI_PUNCT` + web-demo "Punctuation (VN)" toggle | caps + punctuation + sentence segmentation on the ASR transcript (xlm-roberta ONNX, ~1 GB, optional) | "xin chào tôi là duy minh" → "Xin chào tôi là Duy Minh." |
| `playback.py` + live_translate wiring | queue-drained interruptible playback → **barge-in** in mic mode | self-test: interrupt drops queued audio mid-stream |
| `mic_demo.py` | NATIVE mic translator (sounddevice; no browser, no ffmpeg): ENTER→speak→ENTER→translate+speak | native capture verified (RMS>0, pipeline OK) |
| web demo `_ensure_ffmpeg()` | browser mic records webm/opus → needs ffmpeg; we stage one from `imageio-ffmpeg` onto PATH | `[ui] ffmpeg ready`; upload path unaffected |
| web demo `allowed_paths` | VIVOS example button was hitting InvalidPathError | fixed |

**Mic troubleshooting:** hardware mic verified working via sounddevice (11 input devices). If
browser recording still fails it is browser-side (permission 🔒→Microphone→Allow, use
127.0.0.1 not LAN IP, prefer Chrome/Edge). `mic_demo.py` bypasses the browser entirely.

## Honest caveats

- **Cold-start ≠ steady-state.** First turn pays one-time model loads (MT prewarm
  ~22 s, TTS/denoiser/VAD downloads). The app is designed always-warm; report only
  steady-state per-turn latency.
- **streaming_asr is wired into `demo_edge.py` (option a), not yet into
  `run_streaming`** — `run_streaming` still uses whole-utterance faster-whisper;
  `OnlineASR` drives the demo's live loop. Wiring it into the mic/sidecar app is
  the next step (needs a live frame source).
- **Whisper phantom-tail fixed.** `word_timestamps=True` (required by
  LocalAgreement) makes Whisper append a hallucinated trailing segment on short
  clips. `OnlineASR` now drops segments with an impossible speech rate
  (`max_words_per_sec`, language/noise-robust) or `avg_logprob < logprob_floor`.
  Verified: phantom "tôi muốn giúp bạn…" gone; FINAL = the real sentence only.
- **On-device NPU path is documented, not coded** (see NOTICE.md roadmap):
  model-split + host KV-cache (RTranslator) and single-onnxruntime + QNN backend
  (sherpa-onnx) are the Hexagon shipping plan.
```

## Round 4 additions (2026-08-19) — realtime correctness pass

Driven by a re-audit of `../references/` against our own code. Each item below was
a **verified gap or defect**, not a speculative improvement; the file:line on both
sides was checked before the change.

| File / change | What was actually wrong | Verified |
|---|---|---|
| `streaming_tts.py` — fast first fragment | `min_first_fragment_chars` was declared with a comment claiming it implemented RealtimeTTS's fast-yield, but was **referenced nowhere**. The first fragment was always a whole sentence. | first fragment 53 vs 100 chars; **first-audio 791 → 627 ms (−21%)** measured on warm Piper CPU |
| `streaming_tts.py` — decimal split bug | `(?<=[.])(?=\S)` split inside numbers: `"1.5 do C."` → `['... 1.', '5 do C.']`. Industrial text is full of numerals. | `['Nhiet do la 1.5 do C.', 'Xong.']` |
| `streaming_tts.py` — short-fragment merge | A one-word sentence became its own synth call plus a fade pair. Fragment 0 is exempt so the fast first fragment survives. | 20 tests |
| `streaming_tts.py` — conditional fade | `_fade` ran on **every** fragment edge, attenuating 12 ms of real speech onset even when nothing was trimmed. Vietnamese rime attack carries tone. | fades only edges actually cut |
| `playback.py` — interruptible sink | The docstring claimed tens-of-ms barge-in, but `sd.play()`+`sd.wait()` made the **playing fragment uninterruptible**: real barge-in latency was 0.5–3 s. Replaced with a sub-chunked `OutputStream` write loop + `abort()`. | **≤ 1 sub-chunk (20 ms)** of write progress after the flag, 9 tests |
| `playback.py` — `buffered_seconds()` | No way to read playback depth; needed before fragment sizing can adapt to it. | tested |
| `streaming_asr.py` — `no_speech_prob` gate | We filtered only on `avg_logprob` and words/sec. Whisper hallucinations on factory noise flowed into MT and got **spoken**. | 17 tests |
| `streaming_asr.py` — segment-boundary trim | We cut the buffer at the last committed word; `whisper_streaming` deliberately cuts at the second-to-last segment end (it has the word-cut variant written and commented out as rejected). Opt-in `trim_policy="segment"`, default unchanged. | backward-compat test |
| `streaming_asr.py` — VAC leading silence | The whole pending buffer, silence included, was fed to Whisper on speech start. Now sliced at the VAD onset with `buffer_time_offset` re-based. | absolute-timestamp test |
| `lang_router.py` **(new)** | Nothing existed to decide **which direction** to translate. Stdlib-only 4-class text LID (vi/en/zh/ko) + `DirectionRouter` with a manual pin that detection cannot override. | 48 tests, ~380 µs/utterance |
| `local_tts_piper.py` — `provider` knob | `provider="cpu"` was hardcoded; sherpa-onnx accepts `xnnpack`, which is the one that matters on ARM. Env `VOICEAI_TTS_PROVIDER`, default unchanged. | see caveat below |

Total: **104 tests pass**; all touched modules and their downstream importers
(`live_translate.py`, `pipeline_fast.py`) still import clean.

### Round 4 honest caveats

- **XNNPACK is a no-op on this dev box.** The Windows sherpa-onnx wheel ships only
  `CPUExecutionProvider` and logs `Fallback to cpu!`. Interleaved A/B measured
  736 ms vs 766 ms p50 — inside noise. The knob is correct for the ARM build; it
  is **not** a speedup we have demonstrated.
- **Fast first fragment costs total throughput.** Two synth calls instead of one:
  full-utterance synthesis went 791 → 1269 ms (+60%) from per-call overhead. This
  is hidden by playback overlap — measured headroom is **+3.0 s** (fragment 0 is
  3.53 s of audio, fragment 1 takes 497 ms to synthesize) — so there is no
  underrun risk at this fragment size. It would matter if fragments got much
  shorter.
- **`trim_policy="segment"` is off by default and unvalidated on real audio.**
  So is the `no_speech_prob` ceiling of 0.9. Both need an A/B on noisy VIVOS
  before being flipped on.
- **`_DeviceWriter` is untested against real hardware** — no audio device in the
  test environment. The 20 ms bound is on what the worker hands the device; the
  audible bound adds whatever the driver already buffered.
- **`lang_router` inherits the ASR's language commitment.** It is *text* LID. If a
  Vietnamese-only ASR transcribes Korean speech as Vietnamese-looking tokens, this
  module reports `vi` with high confidence and cannot know better. Acoustic LID or
  a multilingual ASR is the only fix; see the bidirectional gap below.

## Round 5 additions (2026-08-19) — bidirectional + terminology

Round 4 closed the realtime defects; Round 5 closes the two gaps Round 4 exposed.

### The device is now bidirectional for real

| File | What |
|---|---|
| `asr_router.py` **(new)** | Two resident ASR models: PhoWhisper-small INT8 for Vietnamese, `Systran/faster-whisper-small` for en/zh/ko. Lazy-loaded, never unloaded, per-language forced token. A Vietnamese-only session never pays for the second model. |
| `live_translate.py` | `--multilingual-asr` plugs the router into the ASR seam. `AsrRouter.model_for()` returns a faster-whisper model, which is exactly what `OnlineASR` consumes, so the streaming machinery is untouched — only the weights and the language token change per turn. |
| `live_translate.py` | Capture/processing decoupling, speculative final ASR during the endpoint countdown, per-sentence MT→TTS pipelining, `DirectionRouter` wiring with a manual pin, half-duplex mic gate with barge-in as opt-in. |

Measured end-to-end on the reverse direction (English speech in, Vietnamese out):

```
smoke_R002_en.wav  [multilingual] ASR 5597 ms
  heard  (en): The strange cracklay of the iron joints.
  routed : en->vi conf=0.93 (detected)
  spoken : Sự sứt nứt kỳ lạ của các khớp sắt.
  total  : 6716 ms
```

Model budget: PhoWhisper INT8 **251.5 MB** + multilingual **486.2 MB** on disk =
**737.8 MB**. The multilingual figure overstates resident weight roughly 2×
because it ships float16 and is quantized to INT8 at load; RSS was not measured
per model and is not claimed.

### Terminology control — the fix for vi→zh/ko quality

`glossary_mt.py` + `industrial_terms.json` (61 terms × 4 languages). Three
interventions were measured against each other on the same industrial sentences:

| Intervention | Result |
|---|---|
| English pivot vi→en→tgt | fixes degenerate output, does **not** recover the lost noun (the error is already made at vi→en), costs a second decoder pass — **3× slower** |
| inline gloss `"xe nâng (forklift)"` | **worse** than bare; the parenthetical confuses alignment |
| **term injection** `"forklift đang lùi"` | recovers the noun in en, zh **and** ko in a single pass — adopted |

Wired into `pipeline_fast.translate` and `translate_sentences_batch` behind
`VOICEAI_MT_GLOSSARY` (default on); both paths verified byte-identical. Exact
whole-utterance glossary hits skip the decoder entirely (0 MT calls).

Before / after, same model, beam 1:

| Source | Before | After |
|---|---|---|
| `Xe nâng đang lùi` (en) | "The train is coming back" | "Forklift is retreating" |
| `Xe nâng đang lùi` (zh) | 车辆正在向后 | **叉车**正在向后 |
| `Xe nâng đang lùi` (ko) | 승차가 뒤집어지고 있어 | **포크리프트**가 뒤집어지고 있습니다 |
| `Máy nén khí … quá nhiệt` (zh) | 区域B的压缩机太热了 | 区域B的**空气压缩机**正在过热 |
| `đeo găng tay và kính bảo hộ` (zh) | 你必须戴**戴戴戴戴戴**和防护眼镜 | 必须戴**手套**和**安全眼镜** |

### Round 5 honest caveats

- **Mandarin is still untested end-to-end.** The repo's only zh audio file,
  `smoke_R002_zh.wav`, is exact digital silence (peak 0.0) — the MMS-TTS Mandarin
  gap that `local_tts_piper.py` already documents. There is no usable Mandarin
  audio here, so the zh ASR path has no latency number, let alone an accuracy one.
- **zh/ko ASR accuracy is unmeasured.** No labelled reference audio exists in the
  repo. The Korean output is real Hangul and plausible; plausible is not a score.
- **The glossary's zh/ko columns have not been reviewed by a native speaker.**
  Exact hits are emitted verbatim, so an error there is a wrong translation spoken
  with full confidence. Treat the tables as a draft.
- **Injection is a nudge, not a constraint.** CTranslate2 has no hard lexical
  constraint. `unused_preferred_terms` reports when the model chose different
  wording — but most of those are correct synonyms (Korean 포크리프트 vs the house
  term 지게차), so it is a consistency signal, not an error detector.
- **Injection is not uniformly better.** On `"Giàn giáo ở tầng ba … đừng leo lên"`
  the English output degraded from "don't go upstairs" to "don't lie up" even as
  "scaffolding" was recovered. Both baseline and injected output get "tầng ba"
  (third floor) wrong.
- **The speculative-final default of 0.25 s is a guess**, tuned against a fake ASR.
  Its real saving depends on PhoWhisper turn-ASR time vs `min_silence`.
- **The half-duplex guard is unverified acoustically** — no measurement that
  closing the mic actually stops own-speaker re-capture on a real device.
- Latency figures in this section were taken on a contended dev box while other
  work was running; treat them as upper bounds, not steady state.

## Round 6 (2026-08-19) — the device layer

`device/` turns four separate entry points into one resident service. Patterned
on `../references/Summer_School_AIEye`, whose most valuable artifacts turned out
not to be models: a launcher encoding a mandatory startup order, a watchdog that
probes for real instead of trusting the process table, and a keepalive that stops
the runtime unloading a model between demo questions.

| File | Role |
|---|---|
| `supervisor.py` | Claims the port **before** touching RAM, verifies the manifest, loads and prewarms ASR → corrector → MT → TTS → VAD/GTCRN with a real inference each, never unloads. Serves `/healthz`, `/probe`, `/metrics`, `POST /translate`. `--mic` runs the live loop in-process. |
| `healthcheck.py` | Three probe levels; `full` runs real ASR+MT+TTS on a pinned VIVOS clip. |
| `watchdog.py` | On failure writes an evidence bundle *before* restarting, then waits for a real re-warm. |
| `provenance.py` + `models.manifest.json` | 46 assets, SHA-256 generated against the files on disk; verified at startup. |
| `run_device.ps1` / `run_device.sh` | Same order, same PASS/FAIL line on both platforms. |
| `DEMO_RUNBOOK.md` | Operator pre-flight with numeric criteria and 11 triage entries. |

**The mic now runs inside the supervisor.** `prototype/live_translate.py mic`
loads its own `FastCascadeS2ST`, so running it beside the supervisor puts a second
~1.4 GB copy of every model in RAM — which did not fit on the box measured here.
`LiveDeps` exists precisely so the warm pipeline can be handed in: measured
**+1 MB RSS**, and `live.pipe is sup.pipe` is True.

Measured on this dev box (x86-64 Windows, 8 GB, contended — **not** the target device):

| | |
|---|---|
| Warm total, 5 runs | 50.4 – 96.8 s |
| Peak RSS warm | 1385.5 MB |
| Acceptance probe (real ASR+MT+TTS) | 11.2 – 44.9 s |
| ASR on a 4.06 s clip | 10.7 – 32.5 s → **RTF 2.6–8, not realtime on this box** |
| Watchdog recovery after a forced kill | 139.3 s (19 s detect + 5 s evidence + 115 s re-warm) |
| Manifest verify | 46/46 assets, 3.4 s |

For contrast the same cascade on a Kaggle T4 warms in 13.5 s and runs ASR at
167 ms p50 (RTF 0.132) — see `../Results/kaggle_t4/README.md`. The dev box is not
evidence about the device; it is evidence that the supervision layer works.

One finding worth keeping: after 2 minutes idle `/healthz` reported `rss=12.3MB`
while the models were still owned — Windows had trimmed the working set. An
"RSS dropped therefore the model died" alarm would have been wrong, so the health
report carries `rss_peak_mb` alongside `rss_mb`.

### Round 6 caveats

- `run_device.sh`, `max_performance.sh` and all four systemd units were **never
  executed** — no Linux/ARM box here. They pass `sh -n` and nothing more.
- ARM latency, memory, thermals, boot persistence and multi-hour soak are
  **unmeasured**.
- Nothing in `device/` touches the Hexagon NPU. The AIEye HTP-arena ordering
  constraint does not transfer — our ordering constraint is host RAM and threads.
- `--mic` is verified to reuse the resident models; it is **not** verified against
  real microphone hardware in this environment.

## Round 7 (2026-08-20) — production hardening + two measured fixes

### Mandarin transcripts now stay in the script the pipeline expects

The T4 run put Mandarin ASR at CER 0.2216 against English 0.0320. Reading the
transcripts showed 20 of 60 rows were **Traditional** output scored against a
**Simplified** reference — CER 0.395 on those rows versus 0.135 on the rest.

That is a production bug, not only a scoring one: MT is configured `zho_Hans`,
the glossary's Mandarin column is Simplified, and the Mandarin voice expects
Simplified input, so a Traditional transcript degrades every downstream stage.

`asr_router` now primes the Mandarin decoder with a Simplified-Chinese
`initial_prompt` (`VOICEAI_ASR_PROMPT_ZH`, set to `""` to disable). Measured on
T4, FLEURS `cmn_hans_cn` test, n=60 (`Results/zh_script_ab.json`):

| | CER | Traditional rows | latency median |
|---|---:|---:|---:|
| no prompt | 0.2216 | 20/60 (CER 0.3946 on those) | 267 ms |
| Simplified prompt | **0.1229** | **0/60** | 270 ms |

**−44.5 % relative CER, Traditional output eliminated entirely, for +3 ms.**
A contended CPU box had suggested +655 ms; that was the box, not the prompt.
This also makes Mandarin the second-best direction by CER after English.

### Mandarin TTS: the setting, not the model

`bench_tts_mandarin.py` normalises for output sample rate and shows MeloTTS zh is
~6x the cost per second of audio, not the 2x its 44.1 kHz would explain:

| Voice | ms per second of output audio |
|---|---:|
| MeloTTS zh_en | 730 |
| en_US-lessac-medium | 114 |
| vi_VN-vais1000-medium | 110 |
| vits-mimic3-ko_KO-kss_low | 149 |

A smaller 16 kHz Mandarin voice (`vits-zh-hf-fanchen-C`, 132 MB) was measured and
**rejected**: it drops Latin characters (`Ignore OOV 'B'`), so "khu vực B" loses
the B — disqualifying on a factory floor full of area codes. The thread budget was
the real lever (2 → 1211, 4 → 880, 6 → 922, 8 → 786 ms/audio-s), and the T4 run
had been using the worst setting. `local_tts_piper._threads_for` now gives
Mandarin its own budget instead of raising the default and stealing cores from ASR.

### Device layer hardened

| File | What |
|---|---|
| `device/pipeline_guard.py` | Per-stage locks (asr → corrector → mt → tts, a fixed order so nothing deadlocks). `FastCascadeS2ST` has no locks and the supervisor gives it four callers; `translate()` mutates a **shared** tokenizer's `src_lang`, so two turns with different source languages could interleave into a translation from the wrong language — a bug the bidirectional work created. Every acquire has a timeout and raises `Busy` → HTTP 503 rather than hanging. |
| `device/devlog.py` | Structured JSONL alongside the human console line, with a closed event vocabulary that rejects typo'd event names. |
| `device/device_config.py` | Validates every `VOICEAI_*` the device depends on **before** importing anything, because `pipeline_fast` reads config in dataclass field defaults — `VOICEAI_CACHE=true` used to raise deep inside an import 20 s into a start, and `VOICEAI_TTS_BACKEND=Piper` silently selected the wrong backend. |
| `device/test_device.py` | Integration tests for the device layer with fakes: config validation, health state machine, degradation paths, manifest failure branch, watchdog decisions, concurrency guard. |

Verified after the changes: `supervisor.py --warm-only` reaches **DEVICE READY in
57.2 s, RSS 1743 MB, no degraded stages**, and `--check-config` refuses a bad
value with the variable named (`VOICEAI_TTS_THREADS_ZH='abc' is invalid`).

### Round 7 caveats

- The GTCRN denoiser result below is n=3 and needs the T4 run before it is quoted.
- The Simplified-prompt latency cost is unmeasured on GPU.
- `vits-zh-hf-fanchen-C` remains on disk and in the catalog but is not selected.
- The dev box was contended throughout; ratios and orderings are trustworthy here,
  absolute millisecond values are not.

## Resolved: what the denoiser is actually for

`exp_noise_robustness.py` on T4, 60 VIVOS utterances × 4 synthetic noise types ×
4 SNRs (**synthetic, not DEMAND/MUSAN** — neither is on disk and the kernel has no
Internet). Full table and CIs in `../Results/noise_robustness_README.md`.

The n=3 smoke had said "GTCRN never improves WER". At n=60 that is **overstated**:
the tally still reads hurts-in-12-of-16, but only **two cells have a CI excluding
zero**. What survives is narrower and more useful:

- **Babble at low SNR is where it breaks.** 0 dB: WER **0.276 → 0.560**, CI
  [+0.172, +0.448]. 5 dB: +0.071, CI [+0.015, +0.134]. Several people talking near
  the mic *is* the factory floor.
- **Statistically neutral in the other 14 cells**, for 363 ms p50 / 539 ms p95 —
  about a quarter of the 1.5 s budget.
- **SI-SDR improved in all 16 cells (+3.6 to +21.9 dB) and never predicted WER.**
  The largest gain in the grid (clank 0 dB, +21.9 dB) still lost WER. Enhancement
  work has to be scored on WER; a perceptual metric will approve a regression.
- **Its real win is the VAD, not the recogniser.** Silero false-triggers on
  impulsive metallic noise at 8–12.5 %; after GTCRN, **0 % in all four clank
  cells**. Each false trigger costs a whole wasted ASR pass.

We had it backwards on both counts: the VAD segmented on **raw** audio (where it
false-triggers) and ASR received **denoised** audio (where it is neutral at best).
`LiveConfig.denoise` now defaults to **off** (`VOICEAI_DENOISE=1` re-enables), with
the numbers in the code and a regression test pinning the default.

Moving the denoiser to the capture path where it belongs needs a *streaming*
denoiser; `edge_frontend`'s GTCRN is whole-signal. sherpa-onnx ships
`OnlineSpeechDenoiser` — that is the follow-up, and it is now the highest-value
remaining noise work.

## Round 8 (2026-08-20) — streaming denoiser, real noise, and two corrections

### GTCRN moved to where it was measured to work

`edge_frontend.StreamingGtcrnDenoiser` wraps sherpa-onnx `OnlineSpeechDenoiser`:
frame-by-frame, 256-sample model frames, buffering mic frames that do not divide
them. Verified against the offline wrapper — identical output length, correlation
0.9285, **29 ms per 100 ms of capture (RTF 0.29)**.

`live_translate` now runs two aligned streams when `denoise_vad=1`: the **denoised**
stream drives the VAD (where GTCRN removed 8.3–12.5 % false triggers), the **raw**
stream reaches ASR (where it is neutral-to-harmful). Off by default — the VAD
benefit was measured offline and the live wiring has no microphone verification.

Two bugs the tests caught while wiring this, both mine:

- a local `take` shadowed the existing `take()` closure, which would have crashed
  the live path whenever `denoise_vad` was on;
- **more dangerous:** sherpa's VAD hands back *the samples it was fed*, so feeding
  it denoised audio silently routed denoised audio into ASR — the exact inversion
  the change exists to prevent. The segment is now re-cut from the raw history by
  absolute sample position, and a test pins the direction.

### Real noise corpus staged

DEMAND (Zenodo 1227121, **CC BY 4.0**, redistribution permitted with attribution),
16 kHz official release, `ch01` only: PSTATION, DKITCHEN, PCAFETER — 28.8 MB with
a sha256 manifest that ships with the audio. PCAFETER is real multi-talker babble
and was added deliberately: synthetic babble is the *only* cell where our headline
finding is significant, so it is the cell that decides whether the finding survives.
MUSAN skipped — no usable subset exists, reasons recorded in the script.

Blocked on the Kaggle **weekly** GPU quota (30 h, exhausted). Everything is staged;
the run needs only a re-push after the reset.

### Correction: the glossary expansion measured exactly nothing

61 → 93 terms moved sentence-level recovery by **zero**, and could not have moved
it: the probe set is generated from the glossary, so it is structurally blind to
expansion. A controlled CPU A/B on the same model with both tables produced 36/36
byte-identical rows. `prototype/glossary_probe_sentences.json` (32 held-out
sentences authored independently of the table) is now what experiment D uses.

### Correction: the Mandarin TTS thread fix is still unmeasured

`PiperTTS(num_threads=2)` sets `_thread_override`, which makes `_threads_for`
return early — so the T4 run measured Mandarin with the per-language budget fully
bypassed. The kernel copy now routes the 2 through `VOICEAI_TTS_THREADS`, keeping
en/vi/ko at 2 as a control.

## Gap status after Round 5

| Gap | Status |
|---|---|
| MT covers all 6 directions | **closed** — NLLB is many-to-many, measured on CPU |
| TTS covers all 4 languages | **closed** — voices on disk, 2-slot LRU, exactly 2 live per session |
| ASR accepts en/zh/ko speech | **closed and measured on T4** — FLEURS en WER 0.066, ko CER 0.073; zh still unmeasured |
| vi→zh / vi→ko terminology | **measured**: term recovery 0.167 → 0.441 on translated sentences; coverage is the limit |
| Direction detection | **closed** for text routing inside a configured pair; open acoustically |
| Two ASR models justified | **closed** — PhoWhisper 0.1259 WER vs multilingual 0.2920 at equal latency (T4, n=120) |
| Mandarin end-to-end evidence | **closed** — FLEURS cmn_hans_cn added; zh→vi measured, RTF p50 0.083 |
| Mandarin transcript script | **closed** — Simplified prompt, CER 0.2216 → 0.1229, Traditional 20/60 → 0/60, +3 ms (T4, n=60) |
| Mandarin TTS speed | **partly closed** — thread budget raised; still 6× cost/audio-s, no acceptable replacement voice found |
| Native review of the zh/ko glossary | **open** |
| One resident process owning every model | **closed** — `device/supervisor.py`, mic included |
| Concurrency safety of the shared pipeline | **closed** — per-stage guard; shared-tokenizer race was real |
| Structured logs / config validation / degradation | **closed** — `device/devlog.py`, `device_config.py`, tests |
| Noise robustness evidence | **measured** — GTCRN neutral in 14/16 cells, harmful in babble; off by default |
| Streaming denoiser on the capture path | **built, benefit refuted** — the VAD win was a synthetic artifact; GTCRN *creates* triggers on real PSTATION |
| Real-corpus noise evidence | **measured** — DEMAND worse than synthetic: 5/12 cells harmful, incl. production hall at 0/5/10 dB |
| Glossary coverage priced | **measurable now** — held-out probes give 0.194 → 0.462, in line with the old circular set |
| Boot persistence / ARM launcher / thermals | **open** — scripts written, never executed |
| On-device Snapdragon / Hexagon | **open** — see DEPLOYMENT_ROADMAP.md; nothing here runs on an NPU yet |

Measured on a Kaggle T4 2026-08-20 (`Results/kaggle_t4/README.md`, run v2):
RTF < 1.0 and end-to-end < 2 s on **360/360** utterances across **all six**
directions, p50 443–1108 ms, peak GPU 2897 MiB with both ASR models resident.
Korean (CER 7.3 %) and Mandarin (CER 12.3 % with the Simplified prompt) are no
longer assumptions.

What is still assumed: behaviour under real industrial noise — the n=3 smoke
suggests our own denoiser makes it worse — and behaviour on the actual device.
Every latency number in this file is CPU or T4, never NPU.
