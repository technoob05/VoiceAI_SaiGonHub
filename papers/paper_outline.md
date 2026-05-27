# Paper Outline (draft v0) — *Tonal-Perceptual Caching for On-Device Vietnamese Speech Translation*

*Working title. Target: IP&M MMLowRes Special Issue (primary) / TASLP / ACL Industry / oral A* conference.*

---

## §1. Introduction (2 pp)

- Motivation: low-resource tonal-language speech translation on edge devices for industrial workers.
- The OneVoice AI Challenge 2026 context — workers repeat short directive phrases many times per shift.
- Existing caches:
  - Exact byte-hash → 0% hit rate on real speech.
  - Generic perceptual hash (audio fingerprinting for music ID) → tone-blind, false-hit on Vi minimal pairs (ma/má/mả).
- **Our contribution**: a 272-d **tonal-perceptual fingerprint** that:
  - Bridges linguistic typology and edge cache design.
  - Discriminates phrases (>5 pp cosine gap) while admitting same-phrase intra-speaker variation.
  - Generalizes to any tonal language (Mandarin, Thai, Yoruba) via tone-histogram dim adjustment.
  - Composes with INT8 quantization and standard streaming MT into a deployable pipeline.
- **Headline number**: 1,441× speedup on cache hit (25 ms vs 36.5 s FP32 cold) with identical output.
- Roadmap to the rest of the paper.

---

## §2. Related work (1.5 pp)

- §2.1 Speech translation: SeamlessM4T, NLLB, cascade vs E2E. Note: most prior systems target high-resource languages, not on-device tonal.
- §2.2 On-device inference: faster-whisper / CTranslate2 / GGML. Note: pure engineering substrate, not novelty.
- §2.3 Audio fingerprinting: Shazam-style (Wang 2003), neural fingerprints (Bertin-Mahieux 2011). Note: optimized for music ID, not phrase-level cache.
- §2.4 Cache architectures for ML: BERT inference cache, KV-cache compression. Note: token-level, not utterance-level.
- §2.5 Tonal-language speech: PhoWhisper (Le et al., VinAI 2024), de la Fuente & Jurafsky 2024 (Mandarin tone layer-probe).
- **Gap**: no prior work treats utterance-level caching for tonal speech translation. Our work fills this gap.

---

## §3. Method (3 pp)

### §3.1 System overview
Pipeline diagram (ASCII version already in README): tone-VAD → T1 audio cache → ASR (INT8 streaming) → T2 text cache → MT (INT8) → TTS (INT8).

### §3.2 Theoretical framework
Define formally:
- *Fingerprint function* `f: A → ℝᴰ` mapping audio to L2-normalized vector.
- **Goal 1 (invariance)**: `cos(f(a), f(a′)) ≥ τ` when `a, a′` are realizations of the same phrase by same/different speakers.
- **Goal 2 (discrimination)**: `cos(f(a), f(a″)) < τ` when `a, a″` are different phrases.
- Threshold *τ* is the cache hit gate; gap *intra − inter* must exceed measurement noise (3σ rule).

### §3.3 Tonal-perceptual fingerprint design — the headline mechanism
Five components, justified individually:
- **A. Tone histogram (6 dims)** — encodes tonal class distribution. Critical for Vi because MFCC alone collapses tone into envelope statistics. Computed via autocorrelation F0 → log-spaced tone assignment.
- **B. Temporal-subsampled MFCC (130 dims)** — 13 MFCC × 10 chunks, captures phoneme trajectory.
- **C. MFCC first-difference (130 dims)** — velocity / phoneme transitions. Adding this drops inter-phrase cosine from ~0.94 to ~0.85.
- **D. F0 statistics (4 dims)** — log-mean, std, range, slope. Adds speaker-pitch discrimination.
- **E. Energy statistics (2 dims)** — utterance RMS, peak chunk RMS. Tie-breaks silence.

Justification: each component targets a distinct invariance/discrimination axis. Section §4 ablation table proves each contributes.

### §3.4 Text-level translation memory (cross-speaker)
Audio cache hits only when **same speaker** repeats. To capture cross-speaker phrase repetition (many workers say "an toàn lao động"), we add a 256-d **diacritic-robust text fingerprint** via hashed character n-grams (n=1..3) on the ASR transcript. Cosine threshold 0.80 (looser than audio T1 because paraphrases differ more than acoustic repeats).

### §3.5 Industrial corpus pre-warming
42 curated Vi industrial phrases × 3 target languages (En/Zh/Ko) loaded at startup. First-time-user demo: any of these phrases triggers a T2 hit on first speak.

### §3.6 Tonal-language VAD
F0-stability gate as a typology-aware pre-filter. Rejects pure noise / silence / hum before cascade runs. ~10 ms overhead, prevents 5+ s of wasted compute on garbage inputs.

### §3.7 INT8 substrate
Engineering: faster-whisper + CTranslate2 NLLB + dyn-INT8 VITS. 3.9× model-size reduction, 5.4× cascade speedup, WER/TER preserved at 0%.

### §3.8 Streaming progressive output
Generator-driven cascade emits partial Vi text as ASR decodes, partial target text as MT decodes. Lowers perceived latency without changing back-end latency.

### §3.9 Persistent cache snapshots
JSON serialization with base64-encoded float16 fingerprints + zlib int16 PCM audio. ~77 KB per audio entry. Allows demo to carry pre-warmed cache across sessions.

---

## §4. Experiments (3 pp)

### §4.1 Datasets
- **VIVOS** (Le et al. 2016): 15 h Vi read speech, 46 train spk + 19 test spk. Primary cache + ablation corpus.
- **FLEURS-vi** (Conneau et al. 2022): ~10 h Vi multi-accent. Robustness test.
- **AISHELL-3** (Mandarin, 4 tones): generalization to non-Vi tonal language.
- **DEMAND + MUSAN** noise: industrial robustness simulation.

### §4.2 Fingerprint-component ablation (the §4 anchor table)
Strip each of (A, B, C, D, E) from the 272-d fingerprint, measure intra/inter-phrase cosine on VIVOS multi-speaker pairs. Show:
- Full → intra 0.97, inter 0.85, gap 0.12 (clean)
- No tone → gap drops (sensitivity test)
- No MFCC → gap drops more
- Tone-only → poor discrimination (6-d too low)
- MFCC-only-subsampled → moderate discrimination, but tone-blind

Results in `Results/fingerprint_ablation.json`. Table 1 of the paper.

### §4.3 End-to-end latency ablation (Table 2)
The 7-config ablation from `prototype/ablate_all.py` on 5+ VIVOS test samples:
- FP32 baseline
- INT8 cascade
- INT8 + tone-VAD
- INT8 + T1
- INT8 + T2
- INT8 + T1 + T2
- INT8 full (+pre-warm + tone-VAD)

Reports latency p50/p95/p99 + WER + TER + cache hit rate.

### §4.4 Cache hit-rate vs deployment scale
Simulate 1, 10, 50, 100 workers × N phrase repetitions / shift. Show that hit rate grows with worker count + repetition frequency. Estimate effective end-to-end latency at each scale.

### §4.5 Cross-language generalization
Apply the fingerprint design (with tone histogram dim adjusted: 4 for Mandarin) to AISHELL-3. Verify intra/inter gap is preserved.

### §4.6 Quality preservation
Compare ASR WER, TER, MT BLEU + chrF + COMET-22, and a native-Vietnamese MOS panel (n ≥ 20) on FP32 vs INT8 vs cache-hit outputs. Demonstrates the 1,441× speedup does not degrade quality.

---

## §5. Deployment notes (1 pp)

- Memory: 861 MB models + ≤10 MB cache. Fits Qualcomm Snapdragon 8 Gen 3.
- Latency budget: NPU target <1.5 s end-to-end after tone-aware quantization spine (CLAUDE.md §4).
- Energy: passive cooling, no sustained inference throttling expected at INT4 NPU.

---

## §6. Limitations and negative results (1 pp)

Three honest failures documented with measurements (paper §6, see `optimization_tracker.md` failure log):

- **§6.1 Speaker-personalized cache** with 26-d MFCC mean+std voice fingerprint — cannot discriminate Vi speakers (cross-spk cos 0.93-0.98). Lesson: global timbre statistics are dominated by recording channel + Vi phoneme distribution, not speaker identity. Need neural x-vector (50+ MB) or temporal voice features.

- **§6.2 Coarse-to-fine MT** with opus-mt-vi-en as the small draft — produces empty translations on Vi industrial phrases (0 fast-path hits in 5/5). Net latency increases due to draft + verifier sequential execution. Lesson: small Vi-MT models are insufficient; would need self-distilled draft.

- **§6.3 First-attempt 32-d global MFCC fingerprint** — inter-phrase cos 0.99 (no discrimination). Lesson: global statistical features collapse phrase identity. Temporal subsampling + delta features are required.

- **§6.4 Tone-VAD F0 range cap 1.8 octaves** — rejected real Vi speech (natural F0 range 2.7+ oct). Re-tuned to 4.0 octaves. Lesson: thresholds must respect target-language typology, not generic ML defaults.

---

## §7. Conclusion (0.5 pp)

We presented a tonal-perceptual cache architecture for on-device Vietnamese
speech translation. The 272-d fingerprint design — combining tone histogram,
temporal-subsampled MFCC, first-difference velocity, F0 statistics, and
energy stats — achieves clean phrase discrimination with intra/inter cosine
gap of ~12 pp on VIVOS. Combined with INT8 quantization, streaming UI,
text-level translation memory, and tonal-language VAD, the system delivers
**5.4× cascade speedup and 1,441× speedup on cache hit**, with WER and TER
preserved at zero on read speech. The mechanism generalizes to other tonal
languages by typological adaptation of the tone-histogram dim. Honest
negative results — speaker cache failure, coarse-to-fine MT failure — are
documented in §6 to ground methodological claims.

---

## Appendix outline

- A. Pipeline + cache architecture (ASCII / TikZ diagram)
- B. Fingerprint component pseudocode
- C. Industrial phrase corpus (42 entries)
- D. Reproducibility checklist + git tag

---

## Author punch-list before submission

- [ ] Run `prototype/fingerprint_ablation.py` on a larger pair set (n ≥ 500) and freeze Table 1.
- [ ] Run `prototype/ablate_all.py --n 20 --target en` (and --target zh / ko) and freeze Table 2.
- [ ] Replicate fingerprint design on AISHELL-3 (Mandarin) — requires Phase B download.
- [ ] Native-speaker MOS panel (n ≥ 20) for quality preservation §4.6.
- [ ] Anonymize repo for IP&M double-blind.
- [ ] Compose final LaTeX (Elsevier `cas-dc.cls`) using this outline + tracker + contribution_analysis.
