# VoiceAI_SaiGonHub — Experiment Tracker

*Spine (locked 2026-05-26)*: **Tone-aware mixed-precision quantization for Vietnamese on-device S2ST**.
*Paper target*: IP&M MMLowRes (primary) → TASLP (secondary) → INTERSPEECH / IWSLT / ICASSP / EMNLP-Industry (fallback).
*Hardware*: Kaggle T4×2 / P100 (free tier) for ablation grid; local dev machine for prototyping.

---

## Current SOTA / headline

| Exp | Method | Model size | Tone-WER (TER) | WER | BLEU (Vi→En) | Latency p99 | Notes |
|-----|--------|-----------:|---------------:|----:|-------------:|------------:|-------|
| exp00 | TER metric impl + 5 synthetic checks | — | metric verified | metric verified | — | — | Self-test only; no model loaded (CPU disk-low). All 5 sanity cases passed: tone-only errors isolated correctly from base-segment errors. Local-walker loaders for VIVOS + FLEURS parquet ready. |
| prototype | Cascade FP32 baseline, **all 3 target languages working** | ASR 967 MB + MT 2.4 GB + TTS ~140 MB | — | — | — | per-utterance ~25s (en) / ~41s (ko) / ~23s (zh, text-only) | Vi→{En, Zh, Ko} verified on VIVOSDEV01_R002. CPU fp32 single-pass. Korean TTS requires `unidecode` Hangul→Latin preprocessing. Mandarin TTS: MMS-TTS-cmn doesn't exist on HF — graceful fallback to silence + text-only. Web UI on :7860 (one-shot) + streaming variant :7861 (silero-VAD endpoint detection). |

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

### Round 3 — Device deployment (target: October–November 2026)
- Compile via Qualcomm AI Hub / QNN SDK
- On-device latency measurement
- Demo-ready artifact for the finale

---

## Known failures (do not repeat)
*(empty — populate as we go)*

---

## Key insights
*(empty — populate as we go)*

---

## How to add results (workflow)
1. Run experiment script on Kaggle / local; save JSON to `Results/` at project root.
2. Add a row to "Active experiment scripts" table above.
3. If headline-worthy, update "Current SOTA" table at top.
4. Update the relevant Round section below.
5. If a new mechanism, add an entry under "Key insights".
6. If a method failed in an instructive way, add to "Known failures".
7. Commit JSON + tracker update together.
