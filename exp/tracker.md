# VoiceAI_SaiGonHub — Experiment Tracker

*Spine (locked 2026-05-26)*: **Tone-aware mixed-precision quantization for Vietnamese on-device S2ST**.
*Paper target*: IP&M MMLowRes (primary) → TASLP (secondary) → INTERSPEECH / IWSLT / ICASSP / EMNLP-Industry (fallback).
*Hardware*: Kaggle T4×2 / P100 (free tier) for ablation grid; local dev machine for prototyping.

---

## Current SOTA / headline

**End-to-end CPU optimization stack** (VIVOSDEV01_R002, target=en, identical output text across regimes):

| Regime | Total | RTF | Model disk | Speedup vs FP32 |
|---|---:|---:|---:|---:|
| 🐢 **FP32 HF baseline (cold)** | **36,500 ms** | **8.98×** | 3,334 MB | 1.0× |
| ⚡ **INT8 cascade warmed** (faster-whisper + CT2-NLLB + dyn-INT8 VITS) | **6,703 ms** | **1.65×** | **861 MB** (3.9× smaller) | **5.4×** |
| 🚀 **INT8 + tonal-perceptual cache HIT** | **25 ms** | **0.01×** | same as above + ~280 KB cache | **1,441×** |

| Bench n=5 warmed | FP32 p50 | INT8 p50 | Speedup |
|---|---:|---:|---:|
| ASR (PhoWhisper-small) | — ms | — ms | 4.16× (p95) |
| MT (NLLB-distilled-600M) | — ms | — ms | 5.57× (p95) |
| TTS (MMS-TTS VITS) | — ms | — ms | 1.85× (p95) |
| **TOTAL** | **24,053 ms** | **6,418 ms** | **3.75×** |
| RTF | 8.00× | 1.90× | 4.21× |
| WER mean | 0.0000 | 0.0000 | (preserved) |
| TER mean | 0.0000 | 0.0000 | (preserved) |

Detailed JSON: [Results/bench_compare_en_n5.json](../Results/bench_compare_en_n5.json).
Full demo log: [prototype/demo_wow_full.log](../prototype/demo_wow_full.log).

| Exp | Method | Notes |
|-----|--------|-------|
| exp00 | TER metric impl + 5 synthetic checks (Results/exp00_ter_metric.json) | metric verified |
| prototype/pipeline.py | FP32 HF cascade baseline (Vi→{En,Zh,Ko}) | reference baseline; web_demo.py :7860 |
| prototype/pipeline_fast.py | INT8 cascade + utterance cache | **3.75× cascade**, **1,441× on cache hit** |
| prototype/bench_compare.py | Warmup-corrected side-by-side bench | reusable for future ablations |
| prototype/utterance_cache.py | **Novel** 272-dim tonal-perceptual fingerprint + LRU | 22 ms lookup; clean threshold at cos≥0.95 |
| prototype/demo_wow.py | 3-regime live demonstration | publishes headline 1441× speedup |

**Headline number we are chasing**: TER stays within +3pp of FP32 baseline while model size ↓ 60–70% and latency p99 < 1.5s end-to-end. **TER, not WER, is the primary metric** — see saturation pivot below.

## Mandatory baselines (from [papers/reading_list_2025_2026.md](../papers/reading_list_2025_2026.md))

Every quantization result in our Table 1 must compare against these 2025–2026 baselines, or reviewers will mark the paper "out of date":

| Baseline | Year | Why mandatory |
|---|---|---|
| **PhoWhisper-small/large** ([arXiv 2406.02555](https://arxiv.org/abs/2406.02555)) | 2024 | Canonical Vi ASR. PhoWhisper-large hits 4.67% WER on VIVOS — defines the saturation ceiling. |
| **VietASR** ([arXiv 2505.21527](https://arxiv.org/abs/2505.21527)) | 2025 | 68M Zipformer beating Whisper-large-v3 on real-world Vi at 22× fewer params. Compact alternative backbone candidate. |
| **SeaLLMs-Audio-7B** ([arXiv 2511.01670](https://arxiv.org/abs/2511.01670)) | 2025 | First multilingual SEA audio-LLM with Vi support. Establishes audio-LLM line. |
| **Qwen3-ASR-1.7B** (Qwen3-Omni branch) | 2026 | Strong recent multilingual ASR including Vi. |
| **Edge-ASR Whisper-INT3** ([arXiv 2507.07877](https://arxiv.org/abs/2507.07877)) | 2025 | Comprehensive PTQ benchmark on Whisper/Moonshine — establishes "what INT3 / INT4 buys" in English; we extend to Vi tone-WER. |
| **FADE** ([arXiv 2601.02455](https://arxiv.org/html/2601.02455v2)) | 2026 | **Most architecturally similar competitor** — per-layer α-coefficient PTQ for encoder-decoder ASR. Our linguistic-probe-driven allocation must beat FADE's statistical-diagnostic allocation at matched compression ratio. **This is exp06.** |

## Saturation pivot (decided 2026-05-26)

The reading list verdict: **VIVOS WER is saturated at 4.67%, AISHELL-1 CER at ~3-4%**. Headline claims must move:

- **Primary metric**: TER (tone-error-rate). No canonical leaderboard exists — that's where our claim must land.
- **Primary WER table**: noisy/conversational subsets where headroom is real (15–30% WER):
  - Noise-augmented VIVOS (DEMAND `PSTATION`+`DKITCHEN` + MUSAN at SNR ∈ {0, 5, 10, 15} dB) → **exp08**
  - FLEURS-vi (accent variance, real WER ~13–15%)
  - VietSuperSpeech conversational ([arXiv 2603.01894](https://arxiv.org/abs/2603.01894), if downloadable)
- **VIVOS-clean is demoted to "no-harm sanity table"** — show our method does not degrade the saturated benchmark.
- **Korean (FLEURS-ko) stays as the non-tonal control** — tone-aware allocation must NOT help Ko at the same magnitude as Vi/Zh.

## Data inventory (Phase A, downloaded 2026-05-26)

Stored in `data/raw/hf_cache/hub/` (HF snapshots) + `data/processed/vivos/` (extracted WAV):

| Dataset | Local path | Size | Format | Status |
|---------|------------|-----:|--------|--------|
| VIVOS (Vi ASR 15h) | `data/processed/vivos/vivos/{train,test}/` | 1.7 GB | WAV + prompts.txt | extracted, exp00 verified |
| FLEURS vi_vn | `…/datasets--google--fleurs/snapshots/<sha>/parquet-data/vi_vn/` | ~2.5 GB | parquet (audio bytes embedded) | ready |
| FLEURS cmn_hans_cn | same parent dir, cmn_hans_cn/ | ~2.5 GB | parquet | ready |
| FLEURS ko_kr | same parent dir, ko_kr/ | ~2.5 GB | parquet | ready (non-tonal CONTROL) |
| FLEURS en_us | same parent dir, en_us/ | ~2.5 GB | parquet | ready |
| InfoRe 25h Vi TTS | `…/datasets--doof-ferb--infore1_25hours/snapshots/<sha>/data/` | 7.5 GB | parquet (15 shards) | ready |

**Skipped from Phase A** (download in Phase B if needed for Mandarin tone eval / Korean control):
- AISHELL-3 (~19 GB) — Mandarin TTS with explicit pinyin tone labels
- Zeroth-Korean (~5.5 GB) — Korean ASR

**Gated, must `huggingface-cli login` to fetch**: openlanguagedata/flores_plus (text MT eval — non-critical, FLEURS sentences cover Vi/En/Zh/Ko already).

**Kaggle bundles built 2026-05-26** (in `kaggle_uploads/`):
- `voiceai-tier1-vi.zip` — 10.02 GB — VIVOS extracted + FLEURS-vi + InfoRe-25h
- `voiceai-tier1-zh.zip` — 2.37 GB — FLEURS cmn_hans_cn
- `voiceai-tier1-ko-en.zip` — 3.42 GB — FLEURS ko_kr + en_us

Each bundle: `LICENSES.md` + `MANIFEST.json` + `README.md` (Kaggle-side loader snippets). Upload via https://www.kaggle.com/datasets/new (drag-drop zip → set slug → Public).

---

## Methodology rules (read before writing any new exp)

### Mandatory measurements per experiment
Every experiment script MUST log to JSON in `Results/`:
- **Tone-Error-Rate (TER)** — tonal phoneme alignment + tone label mismatch rate. Build the metric in `exp00` before anything else.
- **WER, CER** — standard ASR error rates.
- **BLEU, chrF, COMET-22** — MT eval if cascade includes MT step.
- **Latency**: p50 / p95 / p99 per utterance (not mean — long-tail matters on edge).
- **Model size** in MB (weights only, not activations).
- **Peak memory** in MB (activations + weights, simulating NPU).
- **GFLOPs per second of audio** — proxy for energy when we can't measure energy directly.
- **Bootstrap CI (n ≥ 1000 resamples)** for headline numbers.
- **Paired permutation test** vs. baseline for any "we beat baseline" claim.

### Hard constraints
1. **Single-pass inference**: one forward per utterance per stage (ASR / MT / TTS). No TTA, no multi-decode averaging, no beam-N where N > 5 (unless ablation row).
2. **No cloud calls** in any inference path. If a script needs internet to load a model, that script is for *training only*, not for the deployment artifact.
3. **Quantization-realistic eval**: always evaluate the *quantized* model on the *same* test split, never just the FP32 model with quantized weights swapped in at print time.
4. **Layer probing first**: before claiming "layer X encodes tone", run a linear probe and report probing accuracy. Conjecture is not evidence.

### File conventions
- Each exp script is **self-contained**: imports, config dataclass, model load, eval loop, JSON write — all in one file. Copy to Kaggle, run, done.
- Result JSON goes to `Results/expNN_<short_name>.json`.
- Filename: `expNN_<verb>_<object>.py`, NN zero-padded.

### Banned shortcuts
- ❌ Reporting BLEU without TER (BLEU stays high under tone loss — fake signal).
- ❌ Reporting mean latency (mean lies for streaming systems).
- ❌ Tuning quantization on the test set — use a held-out dev split, never the eval set.
- ❌ Comparing against an FP32 baseline that uses different decoding hyperparams than the quantized model. **Same decoding config, same test set, only the quantization config changes.**

---

## Active experiment scripts

| Script | Goal | Status | Result link |
|--------|------|--------|-------------|
| `exp00_ter_metric.py` | Implement Tone-Error-Rate metric on VIVOS / FLEURS-vi | **metric verified** (5 synthetic cases pass; model eval pending Kaggle disk) | [Results/exp00_ter_metric.json](../Results/exp00_ter_metric.json) |
| `exp04_whisper_tone_probe.py` | **GO/NO-GO gate**: linear probe per encoder layer (6-class tone) on PhoWhisper-small. Replicates de la Fuente & Jurafsky (Interspeech 2024) Mandarin result on a different architecture. Three exit gates: peak acc ≥ 0.367 (random+20pp), top-30% layers carry ≥50% of above-random mass, best-vs-worst gap ≥ 0.10. | **script ready** — runs locally CPU (~30 min) or Kaggle T4 (~3 min). Set `VOICEAI_MAX_UTTERANCES` env for fast smoke. | (pending) |
| `exp_t4_bidirectional.py` v4 | Mandarin thread budget with a control group, treatment asserted in the output | **done 2026-08-20** — `tts_threads_per_lang {en:2, vi:2, zh:4, ko:2}`, cpu_count 4; vi→zh TTS 907 → 846 ms (−6.7%) against −0.5…−5.8% drift in five untreated directions → **no detectable effect** | [Results/kaggle_t4/v4_mandarin_threads.md](../Results/kaggle_t4/v4_mandarin_threads.md) |
| `exp_t4_bidirectional.py` v3 | Held-out probe set for the glossary A/B (32 independent industrial sentences) | **done 2026-08-20 (T4 via kit567, n=96 probes / 279 terms)** — recovery **0.194 → 0.462**, within noise of v2's circular-probe 0.167 → 0.441; six directions unchanged. Mandarin thread treatment did NOT apply (staged copy clobbered) → re-run as v4 | [Results/kaggle_t4/v3_heldout_probes.md](../Results/kaggle_t4/v3_heldout_probes.md) |
| `exp_noise_robustness.py` (DEMAND) | Same grid on **real** noise: DEMAND PSTATION / DKITCHEN / PCAFETER, CC BY 4.0, 16 kHz, sha256 manifest | **done 2026-08-20 (T4 via kit567, n=60/cell)** — real noise is **worse** than synthetic: 5/12 DEMAND cells significantly harmful (PCAFETER 0/5 dB, **PSTATION 0/5/10 dB**) vs 2/16 synthetic. VAD benefit did NOT replicate — GTCRN *creates* false triggers on raw-clean PSTATION (0% → 29%). The n=2 smoke was wrong | [Results/noise_robustness_README.md](../Results/noise_robustness_README.md) |
| `exp_noise_robustness.py` | Synthetic industrial noise x SNR x {clean, noisy, noisy+GTCRN}: WER/CER, denoiser SI-SDR gain and latency, VAD false-trigger rate | **done 2026-08-20 (T4, 60×4×4)** — GTCRN raises SI-SDR in 16/16 cells but ΔWER CI excludes zero in only **2/16**, both babble: @0 dB WER **0.276→0.560** [+0.172,+0.448], @5 dB +0.071. Neutral elsewhere, 363 ms p50. **Real win is the VAD**: clank false triggers 8–12.5% → **0%**. Denoiser now off by default | [Results/noise_robustness_README.md](../Results/noise_robustness_README.md) |
| `exp_zh_script_ab.py` | Does a Simplified-Chinese `initial_prompt` stop Whisper emitting Traditional? | **done 2026-08-20 (T4, n=60)** — CER **0.2216 → 0.1229** (−44.5% rel), Traditional rows **20/60 → 0/60**, cost **+3 ms**; shipped as `asr_router.zh_initial_prompt` | [Results/zh_script_ab.json](../Results/zh_script_ab.json) |
| `bench_tts_mandarin.py` | Isolate the Mandarin TTS bottleneck: per-voice ms per second of output audio, thread sweep, interleaved A/B | **done 2026-08-20** — MeloTTS zh 730 vs 110-149 ms/audio-s; thread sweep 2:1211 4:880 6:922 8:786; fanchen-C rejected (drops Latin chars) | [Results/kaggle_t4/README.md](../Results/kaggle_t4/README.md) |
| `exp_t4_bidirectional.py` | T4 benchmark for the bidirectional cascade: ASR A/B on Vietnamese, reverse-direction WER/CER, five measurable directions, glossary A/B, fast-first-fragment A/B | **done 2026-08-20** — PhoWhisper WER **0.1259** vs multilingual **0.2920** (Δ+0.167, 95% CI [+0.116,+0.215], perm p≈1e-4, 120 utts); FLEURS en WER 0.066, ko CER 0.073; RTF<1 on 300/300; peak GPU 2897 MiB; **zh unmeasurable, no Mandarin audio** | [Results/kaggle_t4/README.md](../Results/kaggle_t4/README.md) |
| `exp_t4_s2st_inference.py` | Warm per-utterance latency / RTF / peak VRAM for the offline ASR→corrector→MT→TTS cascade on a Kaggle T4 kernel with Internet disabled | **done 2026-08-19** — n=300 VIVOS, RTF p50 **0.129** / p95 **0.175**, end-to-end p50 **401 ms** / p95 **563 ms**, peak GPU **2.48 GiB**, 300/300 under RTF 1.0 and under 2 s | [Results/kaggle_t4/README.md](../Results/kaggle_t4/README.md) |
| `device/supervisor.py` | One resident always-warm process owning every model, with health probes, watchdog, SHA-256 provenance and an operator runbook | **done 2026-08-19** — warm 50.4–96.8 s, peak RSS 1385.5 MB, watchdog recovery 139.3 s, manifest 46/46; `--mic` reuses the resident pipeline (+1 MB). **ARM/systemd scripts never executed; ASR RTF 2.6–8 on this dev box, not realtime** | [device/DEMO_RUNBOOK.md](../device/DEMO_RUNBOOK.md) |
| `prototype/asr_router.py` | Two resident ASR models — PhoWhisper-small INT8 for vi, faster-whisper-small for en/zh/ko — lazy, never unloaded, per-language forced token | **done 2026-08-19** — 40 tests; reverse direction demonstrated end-to-end (English speech in → `en->vi conf=0.93` → Vietnamese out, 6.7 s total on a contended CPU box); 251.5 + 486.2 MB on disk; **zh/ko accuracy unmeasured, no Mandarin audio in the repo** | [prototype/asr_router.py](../prototype/asr_router.py) |
| `prototype/glossary_mt.py` | Terminology control for MT: English term injection into the source + exact-phrase memory, 61 terms × 4 languages | **done 2026-08-19** — 15 tests; wired into `pipeline_fast` both paths (batch ≡ per-sentence verified); recovers 叉车/포크리프트/空气压缩机 where the baseline said 车辆/승차/压缩机 | [prototype/glossary_mt.py](../prototype/glossary_mt.py) |
| `prototype/lang_router.py` | Decide translation direction inside a configured language pair (stdlib-only vi/en/zh/ko text LID + manual pin that detection cannot override) | **done 2026-08-19** — 48 tests, ~380 µs/utterance; limit: it is *text* LID, so it inherits whatever language the ASR already committed to | [prototype/lang_router.py](../prototype/lang_router.py) |
| `prototype/bench.py` | Batch latency p50/p95/p99 + WER + TER on N VIVOS samples. Output JSON per stage. | **script ready** — `python prototype/bench.py --n 20 --target en` | [Results/bench_*.json](../Results/) |
| (planned) `exp01_baseline_fp32.py` | Establish FP32 baseline: PhoWhisper-small + NLLB-distilled-600M; WER, TER, BLEU, COMET, latency, size | not started | — |
| (planned) `exp02_uniform_int8.py` | Uniform INT8 PTQ (every layer) — compression reference | not started | — |
| (planned) `exp03_uniform_int4.py` | Uniform INT4 PTQ — **must show ≥ +20% absolute tone-WER degradation** vs exp01 to justify the problem | not started | — |
| (planned) `exp04_whisper_tone_probe.py` | **GO/NO-GO**: linear probe per layer for tone classification on Whisper / PhoWhisper. Replicates de la Fuente & Jurafsky 2024 on a different architecture. | not started | — |
| (planned) `exp05_awq_vs_tone_correlation.py` | **GO/NO-GO**: correlate AWQ activation-magnitude salience with tone-probe salience. Pearson r < 0.5 required. | not started | — |
| (planned) `exp06_fade_replication.py` | **Mandatory new ablation** (from reading_list_2025_2026.md). Replicate FADE ([arXiv 2601.02455](https://arxiv.org/html/2601.02455v2)) per-layer α-coefficient PTQ allocation. Our linguistic-probe allocation MUST beat FADE's diagnostic allocation at matched compression. | not started | — |
| (planned) `exp07_selective_quant_v1.py` | First selective config: tone-bearing layers (from exp04) at FP16, rest INT4 | not started | — |
| (planned) `exp08_noise_aug_pipeline.py` | Build DEMAND `PSTATION`+`DKITCHEN` + MUSAN noise augmentation at SNR ∈ {0, 5, 10, 15} dB over VIVOS / FLEURS-vi. Required for non-saturated WER table. | not started | — |

---

## Experiment rounds

### Round 0 — Foundations (target: end of June 2026)
Build the tools we need before any method can be tested. Order matters — exp04 is the **hard go/no-go gate**.

- `exp00_ter_metric.py` — Tone-Error-Rate metric (forced-aligned tonal phoneme comparison)
- `exp01_baseline_fp32.py` — FP32 PhoWhisper-small + NLLB-distilled-600M baseline (WER, TER, BLEU, COMET, latency, size)
- `exp02_uniform_int8.py` — Uniform INT8 PTQ as compression reference
- `exp03_uniform_int4.py` — Uniform INT4 to **show the tone-WER collapse** that the method aims to fix
- **`exp04_whisper_tone_probe.py` — Replicate de la Fuente & Jurafsky (Interspeech 2024) probe on Whisper / PhoWhisper encoders.** The Mandarin layer-8 localization was on wav2vec2/HuBERT/WavLM — NOT Whisper. Must show tone is concentrated in ≤ 30% of layers for the method to hold.
- **`exp05_awq_vs_tone_correlation.py` — Critical defense experiment.** Compute AWQ per-layer activation salience AND our probe-derived tone salience on the same model. If Pearson r > 0.7, our method collapses into AWQ-with-extra-steps and we must pivot. If r < 0.4, the divergence is the *content* of our novelty.

**Hard exit criteria for Round 0** (all must pass before Round 1):
1. exp04: tone-encoding concentrated in ≤ 30% of Whisper encoder layers
2. exp05: AWQ-salience vs tone-probe-salience correlation r < 0.5 (lower is better for our story)
3. exp03 vs exp01: tone-WER degradation under uniform INT4 must be ≥ +20% absolute (otherwise the problem we are solving doesn't exist)

If ANY of these fail → re-read [CLAUDE.md](../CLAUDE.md) §4 backup spines, do not proceed with the tone-anchored method.

### Round 1 — Selective quantization (target: July–August 2026)
Once probing identifies tone-bearing layers, design the selective scheme.

- exp05–10 — sweep selective configs (which subset of layers stays high-precision, what precision)
- Ablations: per-layer importance, per-head importance (finer granularity)
- Mandarin / Thai generalization check

### Round 2 — End-to-end pipeline (target: September 2026)
- Cascade ASR → MT → TTS, each with its own quantization config
- Latency budget allocation across stages
- Pareto curve: model size vs. tone-WER vs. latency

**2026-08-19 — cascade latency measured end-to-end on Kaggle T4** (`exp_t4_s2st_inference.py`,
n=300 VIVOS, Internet off, all weights/wheels shipped as private dataset inputs):

| Stage | p50 | p95 | share of p50 budget |
|---|---:|---:|---:|
| ASR (PhoWhisper-small INT8, CUDA) | 167.3 ms | 204.1 ms | 42% |
| Corrector (vietnamese-correction-v2 INT8, CUDA) | 47.2 ms | 67.2 ms | 12% |
| MT (NLLB-600M INT8, CUDA) | 56.4 ms | 87.1 ms | 14% |
| TTS (Piper, **CPU**) | 131.7 ms | 218.3 ms | 33% |
| **End-to-end** | **400.7 ms** | **563.3 ms** | — |
| **RTF** | **0.129** | **0.175** | — |

RTF < 1.0 and end-to-end < 2 s on 300/300 utterances; peak GPU 2.48 GiB (limit 10 GiB);
cold model load 13.5 s, paid once at prewarm. Full write-up:
[Results/kaggle_t4/README.md](../Results/kaggle_t4/README.md).
T4 is a development stand-in — Round 3 still owes the on-device Snapdragon number.

### Round 3 — Device deployment (target: October–November 2026)
- Compile via Qualcomm AI Hub / QNN SDK
- On-device latency measurement
- Demo-ready artifact for the finale

---

## Known failures (do not repeat)

- **2026-08-19 — corrector silently deleted content words.** On the first 300-sample T4 run the
  safety gate accepted `lính cứu hỏa` → `lính cứu`, `ủy ban châu âu` → `ban châu Âu`, and
  `y tế bình hưng hòa` → `y tế bình hưng`. Character-length ratio (≥0.72) and token edit ratio
  (≤0.35) are both blind to a single dropped word, so an entity reached MT already truncated.
  Fixed by a `deletion_guard` in `prototype/transcript_corrector.py` that rejects any edit
  removing a token without a same-skeleton replacement; 5/300 utterances now fall back to raw ASR.
  Any future gate must be tested for *deletions*, not just substitutions.
- **2026-08-20 — a cold model load charged to an inference timer.** The first bidirectional smoke
  reported vi→zh TTS at 3594 ms (RTF 1.04, apparently breaking the realtime bar). The cause was the
  2-slot Piper LRU evicting voices as the benchmark cycled targets, so `tts_ms` included a voice
  load. Pre-warming all four voices and raising the cap to 4 gave 877 ms. **Any per-stage timer in a
  benchmark that cycles configurations must prove the resource was warm** — and the device's real
  2-slot behaviour then has to be reported separately, because a pair change genuinely does pay it.
- **2026-08-19 — config declared but never wired.** `streaming_tts.StreamingTTSConfig.min_first_fragment_chars`
  carried a comment claiming it implemented RealtimeTTS's fast-first-fragment yield; `grep` showed it was
  referenced nowhere, so every first fragment was a whole sentence. `EDGE_UPGRADES.md` had been claiming the
  feature since Round 1. Same class of error in `playback.py`, whose docstring promised tens-of-ms barge-in
  while `sd.play()`/`sd.wait()` made the playing fragment uninterruptible for 0.5–3 s. **A capability claimed
  in a doc is not a capability — grep for the symbol before writing it into a results table.**
- **2026-08-19 — Kaggle input layout and version lag.** Kernels now mount datasets at
  `/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>/`, and auto-extract each
  uploaded ZIP into a same-named subfolder (`wheels/wheels/…`). The Kaggle image is also on
  CPython 3.12, so a cp311 wheel bundle fails to install. Separately, `kaggle datasets version`
  returns before the new version is live — a kernel pushed immediately after will silently run
  the *previous* version (this cost one full 300-sample run). Verify dataset content by
  re-downloading before pushing the kernel.

---

## Key insights

- **The cascade is not latency-bound on a mid-range GPU — it is load-bound and TTS-bound.**
  Warm end-to-end p50 is 401 ms (RTF 0.129), but cold model load is 13.5 s: prewarm is the
  feature, not an optimization. Of the warm budget, ASR (42%) and CPU-side TTS (33%) dominate,
  so the NPU port should target ASR first and the MT stage last.
- **A bad-looking accuracy number is a hypothesis, not a finding — read the transcripts.**
  Confirmed on T4: the Simplified prompt removed Traditional output entirely (20/60 → 0/60) and cut
  Mandarin CER 0.2216 → 0.1229 for +3 ms. Two thirds of the gap to English was an orthography choice,
  not hearing. Corollary worth keeping: a latency estimate from a contended box is not evidence —
  the same measurement read +655 ms on CPU and +3 ms on T4.
  Mandarin CER 0.2216 looked like whisper-small being weak on Chinese. Reading the rows showed
  20/60 were **Traditional** output scored against a **Simplified** reference (CER 0.395 on those
  rows vs 0.135 on the rest). It is also a production bug, not just a scoring one: our MT is
  configured `zho_Hans`, the glossary's zh column is Simplified, and the zh TTS voice expects
  Simplified — a Traditional transcript degrades every stage downstream of the ASR.
  Mandarin WER (0.998) is meaningless outright: FLEURS ships the reference character-spaced.
- **The cheap fix for a slow stage is usually a setting, and the obvious fix is usually wrong.**
  Mandarin TTS costs ~6x per second of output audio than the other voices (730 vs 110-149 ms/audio-s),
  not the 2x its 44.1 kHz rate would explain. Swapping to the smaller 16 kHz `vits-zh-hf-fanchen-C`
  looked attractive until it **dropped Latin characters** (`Ignore OOV 'B'`), which destroys
  "khu vực B" — disqualifying for an industrial device. The thread budget was the real lever:
  2 threads 1211 -> 8 threads 786 ms/audio-s, and the T4 run had been using 2.
- **The Vietnamese specialist is worth 17 WER points at zero latency cost — measured, not assumed.**
  PhoWhisper-small INT8 0.1259 WER vs whisper-small multilingual 0.2920 on the same 120 VIVOS
  utterances, 177 vs 172 ms p50. Paired bootstrap Δ+0.167 [95% CI +0.116, +0.215], permutation
  p≈1e-4, wins 91/120. The tone-quantization spine rests on a model that empirically earns its
  place; the one-multilingual-model simplification would have cost more than it saved.
- **Korean WER is the wrong metric and will mislead a reader who does not know it.**
  FLEURS ko: WER 0.2441 but CER 0.0732. Korean spacing is partly optional, so a perfect transcript
  scores badly on segmentation alone — verified synthetically ("안녕하세요 반갑습니다" vs the unspaced
  form scores WER 1.00 / CER 0.00). Report CER for ko, WER for en/vi, never an average of the two.
- **Reported effect sizes must exclude the paths that bypass the mechanism.**
  Glossary term recovery reads 0.933 if exact whole-utterance hits are counted — but those never
  reach the decoder, so they measure a lookup table. On actually-translated sentences the honest
  number is 0.167 -> 0.441. The smoke run's headline was an artifact of a probe set made mostly of
  bare terms.
- **A small smoke can point the wrong way, and a negative result can get *worse* with better data.**
  An n=2 wiring smoke suggested real babble might not reproduce the synthetic-babble catastrophe. At
  n=60 the real corpus convicted the denoiser harder: 5/12 DEMAND cells significantly harmful vs
  2/16 synthetic, including the production hall at three SNRs. Worse, the *benefit* that justified
  building the streaming denoiser was a synthetic artifact — on raw PSTATION Silero never
  false-triggers, and after GTCRN it fires on 29% of segments. Build the feature, keep the tests,
  ship it off.
- **A control group is what turns a plausible win into a measured one.**
  With Mandarin threads 2 → 4, vi→zh TTS moved −6.7% — which looks like a win until the five
  untreated directions in the same run show −0.5% to −5.8% of drift. Conclusion: no effect this
  measurement can distinguish. The dev-box sweep's real gain was 2 → 8 threads, and a 4-vCPU kernel
  cannot get there.
- **A circular probe set was blind, not inflated — worth knowing which failure it was.**
  Re-running experiment D on 32 held-out industrial sentences (v3, `kit567`) gave recovery
  **0.194 → 0.462** against v2's 0.167 → 0.441 on glossary-derived probes. The effect size was
  never the problem; the probe set simply could not *detect a change* in coverage. Fixing the
  design neither raised nor lowered the headline, it made the measurement capable.
- **A probe set drawn from the artefact under test cannot measure that artefact.**
  Experiment D built its probes out of the glossary, so growing the table 61 → 93 terms moved
  sentence-level term recovery by **exactly zero** — 36/36 rows byte-identical in a controlled CPU
  A/B, because `find_terms` returns the same 34 terms over the 12 fixed sentences either way. The
  bare-term probes were a positional slice (`terms[:24]`), so the expansion also silently swapped
  seven safety terms for seven machinery ones and shifted the all-probes figure as a pure artifact.
  "Coverage is the limit" was never tested and could not be by that design; a held-out set authored
  independently of the table (`prototype/glossary_probe_sentences.json`) is now wired in.
- **Twice in two days a treatment silently failed to apply — assert it in the output.**
  First `PiperTTS(num_threads=2)` bypassed the per-language budget via `_thread_override`. Then,
  having fixed that in the kernel's staged copy, I overwrote the copy from `exp/` while wiring the
  probe set and reverted it — v3 came back with `tts_threads_per_lang: None` and vi→zh TTS at
  907 ms, i.e. unchanged. The fix now lives in the canonical script and the run **emits the applied
  thread counts**, so a silent revert shows up in the results instead of being assumed away.
- **A constructor argument silently disabled the optimisation it was meant to measure.**
  `PiperTTS(num_threads=2)` sets `_thread_override`, which makes `_threads_for` return early — so
  the T4 run measured Mandarin TTS with the per-language budget fully bypassed, at the worst setting
  in the sweep. A re-run keeping that argument would have reproduced the same 824 ms and been read
  as "the fix did nothing". **Before trusting an A/B, assert the treatment is actually applied.**
- **A perceptual metric will approve a change that hurts recognition — score on WER.**
  GTCRN improved SI-SDR in all 16 noise cells, by up to +21.9 dB, and that gain never predicted the
  WER outcome: the single largest SI-SDR gain (clank @0 dB) still lost WER. The denoiser's only
  demonstrated benefit is on the **VAD** (clank false triggers 8–12.5% → 0%), not the recogniser —
  and our code had it backwards, segmenting on raw audio and feeding denoised audio to ASR.
- **Count of cells is not evidence; the confidence interval is.** The n=3 smoke said "GTCRN never
  improves WER" and the 60-sample tally still reads "hurts in 12 of 16" — but only **2 of 16** CIs
  exclude zero. Reporting the sign count would have overstated a real but narrow finding.
- **A resident-process device layer is worth more than another latency optimization.**
  Two independent shipped-device references converge on the same three artifacts, none of which
  are models: a launcher encoding a mandatory startup order, a watchdog that probes for real
  instead of trusting the process table, and a keepalive that stops the runtime unloading a model
  between demo questions. Our own version caught a real one: running the mic loop beside the
  supervisor loads a *second* full copy of every model (~1.4 GB); handing the warm pipeline in via
  `LiveDeps` costs **+1 MB**. On an 8 GB box that is the difference between a device and a crash.
- **The vi→zh/ko failure was terminology, not capacity — and the cheap fix beats the expensive one.**
  Three interventions were measured head to head: an English pivot (vi→en→tgt) fixes degenerate
  output but cannot recover a noun already lost at vi→en, and costs a second decoder pass (3×
  slower); an inline gloss `"xe nâng (forklift)"` is *worse* than bare; substituting the English
  term into the Vietnamese source recovers the noun in en, zh **and** ko in a single pass. NLLB is
  English-centric enough that an English noun embedded in a Vietnamese sentence survives to any
  target. A 61-term glossary bought more quality than doubling the decoding budget did.
- **NLLB-600M is the quality floor on vi→zh and vi→ko, and beam search does not lift it.**
  `"máy nén khí … dừng dây chuyền"` becomes `"停止电路"` (stop the *circuit*) in Chinese and
  `"연결을 중단"` (stop the *connection*) in Korean; beam 4 changes the wording without fixing the
  meaning and costs 30–40% latency. The vi→en direction is fine. Any claim about the three
  language pairs must be qualified by direction, not averaged.
- **A "safe" correction gate is only as safe as the failure mode it was written against.**
  Length and edit-ratio budgets catch rewrites but not deletions; the deletion case only
  surfaced by reading all 300 transcript diffs, not from the summary statistics.

---

## How to add results (workflow)
1. Run experiment script on Kaggle / local; save JSON to `Results/` at project root.
2. Add a row to "Active experiment scripts" table above.
3. If headline-worthy, update "Current SOTA" table at top.
4. Update the relevant Round section below.
5. If a new mechanism, add an entry under "Key insights".
6. If a method failed in an instructive way, add to "Known failures".
7. Commit JSON + tracker update together.
