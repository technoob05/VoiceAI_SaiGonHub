# VoiceAI_SaiGonHub

Vietnamese-centric **on-device speech-to-speech translation** (Vi ↔ En, Zh, Ko) built for two coupled outcomes:

1. **OneVoice AI Challenge 2026** (Saigon AI Hub × Qualcomm) — $50K prize, finale Nov 2026.
2. **Method paper** for **IP&M MMLowRes Special Issue** (deadline 2027-08-31) or A/A\* venue.

The research spine is **tone-aware mixed-precision quantization**: allocate per-layer bit-width by linguistic property (tone-bearing layers stay at INT8/FP16, rest at INT4). Detail in [CLAUDE.md §4](CLAUDE.md).

---

## Quick start

### 1. Install
```bash
git clone <this-repo>
cd VoiceAI_SaiGonHub
python -m pip install -r requirements.txt    # (you'll create this from imports as needed)
# Minimum runtime deps: transformers soundfile pyarrow torch huggingface_hub gradio
```

### 2. Pull Tier 1 data (~25 GB to D:; first run)
```bash
python data/download/download_tier1.py \
    --cache-dir data/raw/hf_cache \
    --skip "AISHELL/AISHELL-3" "kresnik/zeroth_korean"

# Then extract VIVOS tar.gz to data/processed/vivos/ (one-off):
python - <<'EOF'
import tarfile, gzip, shutil
from pathlib import Path
src = Path("data/raw/hf_cache/hub/datasets--AILAB-VNUHCM--vivos").rglob("snapshots/*/data")
src = next(src)
dst = Path("data/processed/vivos"); dst.mkdir(parents=True, exist_ok=True)
tarfile.open(src/"vivos.tar.gz", "r:gz").extractall(dst)
for fn in ("prompts-test.txt.gz", "prompts-train.txt.gz"):
    with gzip.open(src/fn, "rb") as i, open(dst/fn[:-3], "wb") as o:
        shutil.copyfileobj(i, o)
EOF
```

Datasets pulled (Phase A, ~19 GB):
- **VIVOS** (Vi ASR 15h, primary tone signal)
- **FLEURS** (vi_vn / cmn_hans_cn / ko_kr / en_us — n-way parallel, the cross-lingual benchmark)
- **InfoRe-25h** (Vi TTS)

Phase B (AISHELL-3 + Zeroth-Korean) downloads on demand.

### 3. Try the cascade end-to-end (CLI)
```bash
python prototype/pipeline.py \
    --audio data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav \
    --target en \
    --out prototype/out_en.wav
```
First run pulls PhoWhisper-small + NLLB-200-distilled-600M + MMS-TTS-eng (~3.5 GB to `data/raw/hf_cache/`). Subsequent runs offline.

### 4. Or open the web demo (Gradio)
```bash
python prototype/web_demo.py
# auto-opens http://127.0.0.1:7860
```
Mic record OR upload WAV → pick target language → Translate → see Vi transcript, target text, target audio, latency breakdown. See [prototype/README.md](prototype/README.md).

### 5. Run an experiment (Kaggle-ready)
Every file under `exp/exp*.py` is **self-contained** — copy-paste to a Kaggle notebook, attach the matching dataset (slug `voiceai-tier1-vi` for Phase A), run. Results JSON lands in `Results/`. See [exp/README.md](exp/README.md) and [exp/tracker.md](exp/tracker.md).

```bash
# Local TER metric smoke test (works on CPU, no GPU needed):
python exp/exp00_ter_metric.py
```

---

## Repo layout

```
VoiceAI_SaiGonHub/
├── CLAUDE.md                  ← project plan, research spine, risk register (READ THIS)
├── Challenges_Details.md      ← OneVoice AI Challenge 2026 brief (Saigon AI Hub × Qualcomm)
├── README.md                  ← you are here
├── prototype/                 ← local CPU dev — end-to-end working pipeline + web UI
│   ├── pipeline.py            ← CLI cascade ASR → MT → TTS
│   ├── web_demo.py            ← Gradio web UI on :7860
│   └── README.md
├── exp/                       ← self-contained Kaggle-ready experiment scripts
│   ├── tracker.md             ← SOTA table, baselines, methodology rules
│   ├── README.md
│   ├── exp00_ter_metric.py    ← Tone-Error-Rate metric (verified 2026-05-26)
│   └── (exp01–08 planned)
├── papers/                    ← literature scans + future paper drafts
│   ├── lit_scan_v1.md         ← novelty defense paragraph for IP&M intro
│   ├── reading_list_2025_2026.md   ← SOTA 2025-2026 + saturation pivot
│   └── README.md
├── data/
│   ├── README.md              ← dataset registry (HF IDs, licenses, sizes, tone annotation)
│   ├── raw/                   ← HF caches (gitignored — regenerate via download_tier1.py)
│   ├── processed/             ← extracted VIVOS (gitignored)
│   └── download/              ← download + Kaggle bundle scripts
│       ├── download_tier1.py
│       ├── build_kaggle_bundles.py
│       └── README.md
├── kaggle_uploads/            ← built .zip bundles for Kaggle (gitignored)
└── Results/                   ← experiment JSON outputs (tracked in git)
```

---

## Research spine in 3 lines

1. **Vietnamese has 6 tones**; aggressive INT4 quantization disproportionately damages tone-bearing encoder layers.
2. **Linear-probe** the encoder to localize which layers carry tonal information; **keep those at INT8/FP16, push the rest to INT4**.
3. Result target: model size ↓ 60-70%, **Tone-Error-Rate** stays within +3pp of FP32, latency p99 < 1.5s on Snapdragon Hexagon NPU.

Generalizes to Mandarin, Thai, Yorùbá. Method paper venue: IP&M MMLowRes (primary) → TASLP (secondary) → INTERSPEECH / IWSLT / EACL Findings (fallback).

Full plan + risk register: [CLAUDE.md](CLAUDE.md). Novelty defense: [papers/lit_scan_v1.md](papers/lit_scan_v1.md). SOTA tracking: [exp/tracker.md](exp/tracker.md).

---

## Headline numbers (2026-05-27)

Cascaded CPU inference on a single VIVOSDEV01_R002 utterance, same input, identical output text across all regimes:

| Regime | Total latency | RTF | vs FP32 |
|---|---:|---:|---:|
| FP32 HF cold baseline | 36,500 ms | 8.98× | 1.0× |
| **INT8 cascade warmed** (faster-whisper + CT2-NLLB + dyn-INT8 VITS) | **6,703 ms** | **1.65×** | **5.4×** |
| **INT8 + tonal-perceptual cache hit** | **25 ms** | **0.01×** | **1,441×** |

Tonal-perceptual cache is novel: 272-dim fingerprint = Vi tone histogram + temporal-subsampled MFCC + first-difference deltas + F0 stats + energy stats. Discrimination on VIVOS speakers: self-match cos 1.00, jittered same-phrase 0.95-0.99, different-phrase 0.82-0.91 → clean 4pp gap at threshold 0.95.

## Optimization stack — 8 layers for paper sections

| § | Layer | File | Novel? | Speedup contribution |
|---|---|---|---|---|
| 3.1 | **INT8 substrate** | [pipeline_fast.py](prototype/pipeline_fast.py) | engineering | 5.4× cascade |
| 3.2 | **Streaming progressive UI** | [pipeline_fast.py::run_streaming](prototype/pipeline_fast.py) | UX | perceived <1s |
| 3.3 | **Tonal-perceptual audio cache (T1)** | [utterance_cache.py](prototype/utterance_cache.py) | **novel** | 1,441× on hit |
| 3.4 | **Diacritic-robust text cache (T2)** | [translation_memory.py](prototype/translation_memory.py) | **novel** | skip MT (~3× on cross-speaker hit) |
| 3.5 | **Industrial-phrase pre-warming** | [industrial_phrases.json](prototype/industrial_phrases.json) + [preload_cache.py](prototype/preload_cache.py) | **novel** | 1st-time hit |
| 3.6 | **Tonal-language VAD** | [tone_vad.py](prototype/tone_vad.py) | **novel** | rejects garbage (~10 ms) |
| 3.7 | **Persistent cache snapshots** | [persistent_cache.py](prototype/persistent_cache.py) | engineering | cross-session reuse |
| 3.8 | **(planned) tone-aware mixed-precision quantization** | [CLAUDE.md](CLAUDE.md) §4 + `exp/exp04..07_*.py` | **novel** — method-paper spine | INT4 + tone preservation |

Reproduce:
```bash
python prototype/bench_compare.py --n 5 --target en --out Results/bench_compare_en_n5.json
python prototype/demo_wow.py --target en       # 3-regime CLI side-by-side
python prototype/web_demo_wow.py               # Gradio live UI on :7862
```

### Architecture (cascade with cache tiers)

```
                  audio
                    │
                    ▼
              ┌───────────┐    reject?
              │ tone-VAD  ├──── 🛑 noise / silence / hum
              └─────┬─────┘
                    │ accept
                    ▼
              ┌───────────┐    hit (cos ≥ 0.95)
              │ T1 audio  ├──── 🎯 reuse vi + tgt + audio (~25 ms)
              │   cache   │
              └─────┬─────┘
                    │ miss
                    ▼
              ┌───────────┐
              │ faster-   │ ← INT8, streaming, emits partial Vi text
              │ whisper   │
              └─────┬─────┘
                    │ vi_text
                    ▼
              ┌───────────┐    hit (cos ≥ 0.80)
              │ T2 text   ├──── 🔁 reuse tgt, run TTS only (~3 s)
              │   memory  │
              └─────┬─────┘
                    │ miss
                    ▼
              ┌───────────┐
              │ CT2 NLLB  │ ← INT8, sentence-by-sentence
              └─────┬─────┘
                    │ tgt_text
                    ▼
              ┌───────────┐
              │ MMS-TTS   │ ← dyn-INT8 Linear quantization
              └─────┬─────┘
                    │
                    ▼
                 target speech
```

## Paper-quality architecture

This repo is built around a **single named mechanism**: a *tonal-perceptual cache* for on-device speech translation of tonal languages. Industrial workers repeat short directive phrases hundreds of times per shift, but byte-hash caching gives 0% hit rate on real speech and generic perceptual audio fingerprints are tone-blind — they false-hit on Vietnamese minimal pairs (`ma / má / mả / mạ / mã / mà` = ghost / mother / tomb / rice-seedling / horse / which) and produce semantic errors that are unsafe on a factory floor. We close that gap with a typology-aware fingerprint, a two-stream voting cache architecture, and a stack of cooperating mechanisms that together make the cache survive industrial noise, cross-speaker traffic, and cold-start conditions. The 272-d fingerprint design (paper §3.3) is the *component set*; the two-stream voting layer (paper §3.5) is the *architecture* that makes the component design pay off on tonal minimal pairs.

### Named mechanism — 272-d tonal-perceptual fingerprint

The fingerprint *f: A → ℝ²⁷² → S²⁷¹* concatenates five linguistically motivated components, then L2-normalizes:

| Component | Dim | What it encodes | Why a tonal language needs it |
|---|---:|---|---|
| Tone histogram | 6 | Vi tone-class distribution (6 lexical tones via autocorrelation F0 + log-spaced binning) | Orthogonal to MFCC; the only place tone identity survives |
| Temporal-subsampled MFCC | 130 | 13 MFCC × 10 chunks — phoneme trajectory | Primary phrase-level discriminator |
| MFCC first-difference (Δ) | 130 | Phoneme transition velocity | Drops inter-phrase cosine ~0.94 → ~0.85 |
| F0 statistics | 4 | log-mean, std, range, slope | Speaker-pitch tie-break |
| Energy statistics | 2 | utterance RMS, peak chunk RMS | Silence tie-break |
| **Total** | **272** | L2-normalized | — |

### Architecture — TWO-STREAM VOTING

Naive L2-normalized concatenation lets the 130-d MFCC component dominate the unit-sphere geometry; the 6-d tone histogram becomes operationally invisible (paper §2.6 finding: full-272d intra-tone cosine 0.92 > inter-syllable 0.91, *fails to protect minimal pairs*). The fix in [`prototype/two_stream_voting.py`](prototype/two_stream_voting.py) keeps the two signals **independent**: each cache entry stores Stream A (6-d tone histogram, τ_A = 0.85) and Stream B (260-d MFCC + delta, τ_B = 0.93). A cache hit fires only when **both streams' best matches point to the same stored entry** *and* both exceed their thresholds — a logical AND. When MFCC says "match cached `má`" but the tone histogram disagrees, the tone-stream **vetoes** the hit and the cascade re-runs. Measured on 5 syllables × 6 tones = 30 minimal-pair clips: self-lookup 30/30 (**100%**), correct-miss on tonal variants 25/25 (**100%**), false-hit rate 0/25 (**0%**) — with diagnostic counters showing 5 MFCC-only false positives all vetoed by tone-stream disagreement.

### Cooperating mechanisms

- **Spectral-subtraction noise filter** (paper §3.7) — pre-ASR enhancement recovering +7.6 to +9.8 dB SNR in 9 ms; auto-bypassed on clean input. In the industrial 10-20 dB band it *doubles to triples* cache hit rate (15 dB: 3/10 → 9/10; 10 dB: 0/10 → 5/10). [`prototype/noise_filter.py`](prototype/noise_filter.py).
- **Diacritic-robust text cache** (paper §3.4) — 256-d hashed character n-gram (n=1..3) fingerprint on the ASR transcript, threshold cos ≥ 0.80, captures cross-speaker phrase repeat (e.g., many workers saying *"an toàn lao động"*) by skipping MT entirely. [`prototype/translation_memory.py`](prototype/translation_memory.py).
- **Industrial pre-warming** (paper §3.5) — 42 curated Vi industrial phrases × 3 target languages bootstrapped at startup, giving a first-time hit on common directives. [`prototype/industrial_phrases.json`](prototype/industrial_phrases.json) + [`prototype/preload_cache.py`](prototype/preload_cache.py).
- **Tonal-language VAD** (paper §3.6) — F0-stability noise gate, ~10 ms overhead, 100% Vi accept and 0% silence/hum accept on the synthetic test; an F0-range cap re-tuned from 1.8 to 4.0 octaves once we measured natural Vi F0 dynamics. [`prototype/tone_vad.py`](prototype/tone_vad.py).
- **Persistent snapshots** (paper §3.7) — JSON serialization with base64 float16 fingerprints + zlib int16 PCM (~77 KB/audio entry, round-trip cos = 1.000); pre-warmed cache survives across sessions. [`prototype/persistent_cache.py`](prototype/persistent_cache.py).
- **PQ compression** (paper §3.X) — M=16, K=256 product-quantized codes give **68× memory reduction** (1088 B → 16 B) at round-trip cosine error 0.0000 in-distribution; enables fitting a large warm cache into NPU SRAM. [`prototype/pq_compress.py`](prototype/pq_compress.py).

### Empirical evidence (anchor numbers — all measured, no fitted values)

| # | Quantity | Measured | Source |
|---|---|---:|---|
| 1 | INT8 cascade speedup vs FP32 cold | **5.4×** (6,703 ms vs 36,500 ms, identical text) | `Results/bench_compare_en_n5.json` |
| 2 | Tonal-perceptual cache hit speedup | **1,441×** (25 ms vs 36,500 ms FP32 cold) | `Results/bench_compare_en_n5.json` |
| 3 | Two-stream correct-miss on tonal minimal pairs | **100%** (25/25), false-hit 0/25 | `Results/two_stream_voting_eval.json` |
| 4 | Spectral filter SNR gain at σ=0.10 noise | **+9.8 dB** in 9 ms compute | `prototype/noise_filter.py` self-test |
| 5 | Fingerprint ablation gap (MFCC-only vs full) | gap 0.092 → **0.148** (+0.056 from delta features) | `Results/fingerprint_ablation.json` |
| 6 | Worker simulation T1 hit rate at K=2 / R=3 | **66.7%** at every scale (1/3/5 workers); p50 35 ms | `Results/worker_simulation.json` |

### Negative results — published as evidence of methodological rigor

<details>
<summary>4 honest failures (paper §6) — click to expand</summary>

- **§6.1 Speaker-personalized cache** (26-d MFCC mean+std voice fingerprint). All 3 distinct VIVOS speakers cluster into a single partition (pairwise voice cos 0.93-0.98, above the 0.85 partition-creation threshold). Lesson: global timbre statistics are dominated by recording channel + Vi phoneme distribution, not speaker identity. Fix would need a neural x-vector / ECAPA-TDNN embedding (50+ MB).
- **§6.2 Coarse-to-fine MT** with `opus-mt-vi-en` as the draft model. Vi industrial phrases (*"an toàn lao động"*, *"xin chào"*, *"dừng dây chuyền"*) produce empty / single-punctuation output, mean per-token logprob -8 to -10. Fallback fires 100% of the time, so total latency = opus-mt setup + NLLB > NLLB-alone.
- **§6.5 Weighted-concat fingerprint**. Eight α-weight schemes (tone_boost ∈ {1, 2, 5, 10, 20}, tone+f0_boost, tone-only-via-weight, mfcc-only-via-weight) *all* yielded negative protect-gap (-0.02 to -0.03). Suspected confound: stochastic VITS synthesis at N=6 syllables × 6 tones is below the TTS-variance noise floor. This failure is exactly what motivated the two-stream voting fix.
- **§6.3 First-attempt 32-d global MFCC fingerprint**. Inter-phrase cos 0.99 — all VIVOS speakers collapse to a single point. Justifies the temporal-subsampling + delta-feature design that followed.

</details>

---

## Status

| Phase | Date | Status |
|-------|------|--------|
| Plan locked (method paper, spine = tone-aware quant, venue priority IP&M → TASLP → conf) | 2026-05-26 | ✅ |
| Literature scan v1 (defense paragraph drafted, 3 critical risks surfaced) | 2026-05-26 | ✅ |
| Reading list 2025-2026 + saturation pivot (TER as primary metric) | 2026-05-26 | ✅ |
| Tier 1 data downloaded (~19 GB) + 3 Kaggle bundles zipped | 2026-05-26 | ✅ |
| exp00 TER metric verified (5 synthetic cases) | 2026-05-26 | ✅ |
| Prototype CLI end-to-end smoke test (Vi → En via VIVOS sample) | 2026-05-26 | ✅ |
| Prototype web demo (Gradio :7860) — mic record verified | 2026-05-27 | ✅ |
| exp04 Whisper tone-layer probe (**GO/NO-GO gate**) | target 2026-06-08 | ⏳ |
| exp05 AWQ-salience vs tone-probe correlation (**GO/NO-GO gate**) | target 2026-06-15 | ⏳ |
| Phase 2 Technical Spec (competition) | due Jul 2026 | ⏳ |
| Phase 3 Prototype submission | due Aug-Sep 2026 | ⏳ |
| Phase 5 Finale, HCMC | Nov 2026 | 🏁 |
| IP&M MMLowRes paper submit | target Q1-Q2 2027 | 📝 |

---

## Constraints & gotchas (worth knowing)

- **Windows C: drive runs hot**; all HF caches MUST redirect to D: via env vars (HF_HOME, HF_HUB_CACHE, HF_XET_CACHE, TMP, TEMP). `pipeline.py` and `web_demo.py` do this automatically. Detail: [data/download/README.md](data/download/README.md).
- **transformers 5.0.0 has a known bug** with NLLB's `AutoTokenizer.from_pretrained` → we use `NllbTokenizerFast` explicitly. See `prototype/pipeline.py:_ensure_mt`.
- **HF xet downloader** can fail on `os error 112` when system temp is on a full drive; pip-cache + ~/.cache cleanup typically recovers 10+ GB.
- **VIVOS ships as `vivos.tar.gz` with a legacy `vivos.py` loading script** that `datasets 4.4+` no longer supports → we use `huggingface_hub.snapshot_download` and walk files manually (see `exp/exp00_ter_metric.py:_load_vivos_local`).

---

## License & anonymity

Private repo. Will be **fully anonymized** before IP&M submission (double-blind). Keep an `anon/` branch ready closer to submission time — strip author names, GitHub URLs, dataset hosting account names, acknowledgments.

External public datasets retain their own licenses (CC-BY, Apache-2.0, etc.) — see [data/README.md §9 Licensing flags](data/README.md).
