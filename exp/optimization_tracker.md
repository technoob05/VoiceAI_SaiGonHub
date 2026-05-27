# Optimization Tracker — VoiceAI_SaiGonHub

Status board for every optimization idea explored.

**Methodology rule**: every entry must be measured against the same bench
script ([prototype/ablate_all.py](../prototype/ablate_all.py)) on a fixed set
of VIVOS test utterances (seed=42) so speedup numbers are comparable. Decision
column records SHIP / TRIAL / DROP based on the measurement.

Conventions:
- 🟢 SHIP — keep in production pipeline (`pipeline_fast.py`)
- 🟡 TRIAL — present in repo but behind a flag, awaiting more eval
- 🔴 DROP — measured, found not to help (or to hurt). Useful as a "known failure" paper row.
- ⏳ planned — not yet built

| # | Idea | Theory speedup | Measured speedup | WER Δ | TER Δ | Decision | Notes |
|---|------|----------------|------------------|-------|-------|----------|-------|
| 1 | **INT8 cascade substrate** (faster-whisper + CT2-NLLB + dyn-INT8 VITS) | 4-8× | **MEASURED: 5.4× p50 (vs FP32), 1.00× baseline in ablate-all** | 0.0 | 0.0 | 🟢 SHIP | [Results/bench_compare_en_n5.json](../Results/bench_compare_en_n5.json), [Results/ablation_en_n3.json](../Results/ablation_en_n3.json) |
| 2 | **Streaming progressive UI** (run_streaming generator) | perceived <1s | first Vi word visible <2s on 4s audio | — | — | 🟢 SHIP | UX win; backend latency unchanged |
| 3 | **Tonal-perceptual audio cache** (T1, 272-d) | 5-50× on hit | 1,441× on hit (25ms vs 36500ms FP32); **ablation: +1.37× over INT8 alone in mixed traffic (n=3)** | 0.0 (identical output) | 0.0 | 🟢 SHIP | Novel for tonal-lang industrial S2ST; paper §3.3 — **HEADLINE CONTRIBUTION** |
| 4 | **Diacritic-robust text cache** (T2, 256-d hashed n-gram) | 3-10× on cross-speaker hit | self-test 0.83-1.00 cos on paraphrases; **ablation: +1.34× over INT8 alone (n=3)** | — | — | 🟢 SHIP | Skips MT on hit; paper §3.4. **T1+T2 combined: 1.80× (best result in mixed-traffic ablation)** |
| 5 | **Industrial-phrase pre-warming** (42 phrases × 3 langs) | 1st-time hit on common phrases | T2 hit rate 90% on selftest queries | — | — | 🟢 SHIP | Paper §3.5; demo killer |
| 6 | **Tonal-language VAD** (F0-stability gating) | rejects noise <10ms | 100% Vi accept, 0% silence/hum accept (synthetic); **ablation: +1.20× over INT8 cascade alone (n=3 VIVOS)** | — | — | 🟢 SHIP | Paper §3.6; complementary to silero |
| 7 | **Persistent cache snapshots** (JSON) | cross-session reuse | round-trip cos=1.000, ~77 KB / audio entry | — | — | 🟢 SHIP | Engineering, paper §3.7 |
| 8 | **Cross-stage cache fusion** (T1+T2 unified score) | +10% hit rate over disjoint tiers | not implemented this session — disjoint tiers already give independent wins | — | — | ⏳ deferred | Marginal value over the existing tier dispatch in `run_streaming` |
| 9 | **Coarse-to-fine MT** (opus-mt-vi-en fast path → NLLB fallback) | 1.5-2× MT median | **MEASURED: opus-mt-vi-en outputs empty strings on Vi industrial phrases** (latency 60-280 ms but zero translation) | — | — | 🔴 **DROP** | See failure log #3. Vi→En coverage too weak in opus-mt; fallback fires 100%. Net latency = opus + NLLB, slower than NLLB alone. |
| 10 | **Speaker-personalized cache** (voice embedding → per-speaker partition) | +20-50% hit rate per worker | **MEASURED: 26-d MFCC mean+std voice fingerprint gives cross-speaker cos 0.93-0.98 — fails to discriminate Vi speakers. All 3 VIVOS test speakers cluster into 1 partition.** | — | — | 🔴 **DROP** | See failure log #4. Need neural x-vector or temporal voice fingerprint. Paper §6.1 candidate. |
| 11 | **PQ-compressed fingerprints** (M=16, K=256 → 16-byte codes) | 32× cache memory reduction | **MEASURED: 68× memory reduction** (1088 bytes → 16 bytes), round-trip cosine error 0.0000 on in-distribution vectors | — | — | 🟡 **TRIAL** | `prototype/pq_compress.py` shipped; integrate into UtteranceCache for edge deployment when on-device memory matters |
| 12 | **Adaptive precision per token** (INT4 first, INT8 fallback on low conf) | 1.3-1.8× ASR/MT | not implemented this session — depends on INT4 CT2 model availability | — | — | ⏳ deferred | Aligns with method-paper spine (CLAUDE.md §4); plan with exp04 results |
| 13 | **Per-token confidence color overlay** (UI) | UX | — | — | — | ⏳ planned | faster-whisper word_timestamps + probability |
| 14 | **Latency gauge widget** (UI) | UX | — | — | — | ⏳ planned | Circular gauge with 1.5s target |
| 15 | **Tone-aware mixed-precision quantization** (research spine) | INT4 + tone preservation | (Phase 3 of project) | TBD | TBD | ⏳ planned | CLAUDE.md §4 method-paper spine |

---

## Failure log (paper-worthy "what didn't work")

Track ideas that we measured and rejected. These are paper §6 "lessons learned" material.

| # | Idea | Why it didn't work | Reference |
|---|------|--------------------|-----------|
| — | First fingerprint (32-d global MFCC stats) | False-hit rate too high (inter-phrase cos 0.99) — collapsed all VIVOS speakers to one cluster. Fixed by adding temporal subsampling + delta features → 272-d. | utterance_cache.py git history |
| — | F0 max range 1.8 octaves in tone-VAD | Rejected real Vi speech (which spans 2.7+ octaves of F0 dynamics). Re-tuned to 4.0 octaves. | tone_vad.py git history |
| 3 | **Coarse-to-fine MT** with opus-mt-vi-en as the small draft | The small Vi-MT model (Helsinki-NLP/opus-mt-vi-en, ~300 MB) produces **empty or single-punctuation output** on every Vi industrial phrase tested (an toàn lao động → "", xin chào → "", dừng dây chuyền → "", cám ơn rất nhiều → "."). Mean per-token logprob -8 to -10 (high uncertainty). Net effect: every query falls back to NLLB anyway, so total latency = opus-mt setup + NLLB latency > NLLB-alone latency. The Vi→En direction of opus-mt is undertrained. For C2F to work we'd need either (a) a stronger small Vi→En MT (mBART-50 distilled?), or (b) train our own distillation. | `prototype/coarse_to_fine_mt.py` self-test output, commit history. Paper §6.3 candidate "negative result — small MT models for Vi are too weak". |
| 4 | **Speaker-personalized cache** with 26-d MFCC mean+std voice fingerprint | The 26-d global MFCC statistical fingerprint does **not** discriminate between Vi speakers. Tested on 3 distinct VIVOS speakers (SPK01, SPK02, SPK03) × 2 utterances each: all 6 utterances landed in speaker partition `spk1` because pairwise voice-fingerprint cosines were 0.93-0.98 — well above the 0.85 partition-creation threshold. The intuition that "different speakers have different voice timbres" is correct, but global MFCC stats are dominated by recording channel + Vi phoneme distribution, not by speaker identity. To fix: either (a) use a neural x-vector / ECAPA-TDNN embedding (50-150 MB), or (b) use temporal voice features (delta MFCC, pitch-range trajectory) instead of static stats. | `prototype/speaker_cache.py` self-test output. Paper §6.1 candidate "negative result — pure timbre statistics insufficient for industrial speaker discrimination". |

---

## Headline numbers achieved so far (2026-05-27)

Cascaded CPU inference, identical output, single VIVOSDEV01_R002 utterance, target=en:

```
FP32 cold baseline              36,500 ms  RTF 8.98×   1.0×
INT8 cascade warmed              6,703 ms  RTF 1.65×   5.4×
INT8 + audio cache hit              25 ms  RTF 0.01×   1441×
INT8 + tone-VAD reject              ~12 ms (skip entire cascade on noise)
INT8 + T2 text-cache hit         ~3,000 ms (skip MT, TTS only)
```

---

## How to add a new idea

1. Append a row in the main table with status ⏳ planned + theoretical speedup estimate.
2. Build the module (`prototype/<name>.py`) with a self-test.
3. Add it as a togglable flag in `prototype/ablate_all.py`.
4. Run the ablation bench (n ≥ 10 VIVOS samples).
5. Update the row with measured speedup + WER/TER delta + decision (SHIP / TRIAL / DROP).
6. If DROP, also add to the Failure log table with the reason — these are paper §6 material.
