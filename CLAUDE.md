# VoiceAI_SaiGonHub — Project Instructions

## 1. Dual Mission

This project pursues **two coupled outcomes** from a single research-engineering effort:

### A. Competition: OneVoice AI Challenge 2026 (Saigon AI Hub × Qualcomm)
- **Prize pool**: $50,000 (1st: $20K + partnership; 2nd: $15K; 3rd: $10K; Special Innovation: $5K)
- **Goal**: Standalone **on-device** AI translation device using **Edge AI** on Qualcomm Snapdragon (Hexagon NPU)
- **Language pairs (required)**:
  - Vietnamese ↔ English
  - Vietnamese ↔ Mandarin
  - Vietnamese ↔ Korean
- **Form factor**: Any portable device — pendant, badge, earbud, handheld, etc.
- **Deployment constraint**: 100% on-device. **No cloud dependency.** Optimized for noisy, hands-busy, low-connectivity industrial environments (factories, construction, logistics, hospitals).
- **Judging weights**:
  - Innovation & Novelty — 25%
  - **Technical Performance — 50%** (accuracy, latency, speed, stability)
  - Business Viability — 25%

### B. Academic Publication (parallel target) — **METHOD paper, not benchmark**

Paper contribution type is **locked**: this is a **method paper** with a novel technical mechanism + clean ablations. Not a benchmark/dataset paper. Reviewers at IP&M / TASLP penalize "yet another benchmark" — method-led narrative is mandatory.

Venue priority (decided 2026-05-26):

**Primary (journal — main shot)**:
1. **Information Processing & Management** — Special Issue "VSI: MMLowRes"
   - Submission window: **2026-04-31 → 2027-08-31** (rolling — target submit Q1–Q2 2027)
   - **Double-blind** — anonymize before upload
   - Portal: https://submit.elsevier.com/IPM
   - Managing Guest Editor: Karol Nowakowski (nowakowski@yamanashi.ac.jp)
   - Metrics: CiteScore 18.6, IF 6.9

**Secondary (journal — second shot if rejected, or parallel different framing)**:
2. **IEEE/ACM TASLP** (Transactions on Audio, Speech, and Language Processing)
   - Top journal for speech + language. IF ~4.4 but prestige very high in speech community.
   - Reviewers expect: ablation, WER/BLEU/COMET, latency p99, MOS. Linguistic-property-aware contributions land well.
   - Use if IP&M angle leans "community/ethical" but we also have a sharper systems angle worth its own paper.

**Fallback (conferences, only if both journals reject or for early publicity)**:
3. **INTERSPEECH 2027** (4-page, deadline ~March 2027) — speech community home turf
4. **IWSLT 2027** (workshop, co-located with ACL) — spoken-language-translation niche; shared task entry gives cheap credibility
5. **EMNLP 2027 Industry Track** — applied NLP, deployment-friendly
6. **ICASSP 2027** (deadline ~Sept 2026) — signal/quantization framing
7. **AACL-IJCNLP** — Asia-Pacific friendly for Vietnamese
8. **EACL 2027 Findings**, **ACL 2027 Findings** — last resort for NLP venue stamp

**Framing for the method paper**: position the work as *a [method-specific] technique for on-device speech translation of an under-resourced tonal language, with Mandarin and Korean as cross-lingual transfer testbeds*. The "method-specific" slot is filled by the Research Spine (Section 4).

---

## 2. Where We Are (date-aware)

- **Today**: 2026-05-26
- **Competition phase**: **Phase 1 — Team Registration (May–June 2026)**
- **Next deadline**: Phase 2 Technical Specification submission, **July 2026**
- **Time to finale**: ~6 months
- **Time to IP&M deadline**: ~15 months — aim to **submit early** (target Q1 2027) so reviewer revision cycles complete before journal-final.

---

## 3. Technical Architecture (working hypothesis — refine as we go)

### Pipeline (Speech-to-Speech Translation, S2ST)
```
Mic → VAD → ASR (multilingual) → MT (NMT or direct cascade) → TTS → Speaker
                            ↘ or → end-to-end S2ST model ↗
```

### Candidate model components (all must be quantizable for Hexagon NPU)
- **ASR**: Whisper (small/tiny, INT8 via Qualcomm AI Hub), or PhoWhisper for Vietnamese, or NeMo Canary.
- **MT**: NLLB-200 (distilled-600M), SeamlessM4T (medium), or M2M-100 — quantize aggressively (INT4/INT8).
- **TTS**: VITS, Piper, or XTTS-v2 (multilingual). Voice cloning is **optional** — focus on intelligibility + speed first.
- **End-to-end alternative**: SeamlessM4T-v2 or AudioPaLM-style direct S2ST (if it fits on NPU after quantization).

### Edge constraints to design against
- **Latency budget**: target < **1.5 s** end-to-end per utterance (industry threshold for "feels real-time"). Stretch: < 800 ms.
- **Memory**: typical Snapdragon 8 Gen 3 / X-Elite NPU has 8–16 GB shared. Plan for ≤ 4 GB model footprint.
- **Power**: passive cooling; sustained inference must not thermally throttle.
- **Offline**: zero network calls. All models, vocoders, lexicons bundled.

### Qualcomm AI Hub
- Use **pre-optimized models** from https://aihub.qualcomm.com when available (saves quantization-debug time).
- Anything custom must be exported via QNN SDK or compiled through AI Hub's submission pipeline.

---

## 4. Research Spine — **Method, not Benchmark**

Decided 2026-05-26: paper is a **method paper**. The method must (a) be defendable as novel for IP&M/TASLP, AND (b) be on the critical path to shipping the competition prototype — so research effort and engineering effort are the same effort, not parallel work.

### 🥇 Primary spine (default): **Tone-aware mixed-precision quantization for Vietnamese on-device S2ST**

**Core insight**: Vietnamese has 6 lexical tones; tone loss = semantic loss (BLEU stays high, meaning collapses). Aggressive INT4 quantization disproportionately damages **tone-bearing layers** (early acoustic encoder + decoder cross-attention) more than other layers. We propose **linguistic-property-aware selective quantization**: keep tone-critical layers at INT8/FP16, push the rest to INT4. Result: model size ↓ 60–70%, tone-WER stays within +3% (vs. +47% for uniform INT4).

**Why this wins the competition**:
- We *must* quantize to fit Hexagon NPU. The method IS the engineering work.
- Smaller model → fits latency budget < 1.5s.

**Why this passes journal review**:
- **Novelty defense (critical, must rehearse)**: Mixed-precision quantization exists. **What's new is selecting precision per layer based on the linguistic property the layer encodes** (tone-bearing for tonal languages, formant-bearing for vowel-rich languages, etc.). No prior work selects precision by linguistic typology of the target language.
- **Generalizable**: same method applies to Mandarin, Thai, Yoruba, Igbo (all tonal, all low-resource). This matches IP&M scope directly.
- **TASLP-friendly**: "linguistic-property-aware compression" is exactly the kind of angle TASLP rewards.

**Data needed (low risk, all from [data/README.md](data/README.md))**:
- **Tone-positive signal**: VIVOS + FLEURS-vi (Vietnamese, 6 tones via orthographic diacritics) + AISHELL-3 (Mandarin, 4 tones via explicit pinyin tone marks).
- **Tone-negative CONTROL** ⭐: FLEURS-ko + Zeroth-Korean. **Korean is non-tonal** — this gives a natural control condition: tone-anchored quantization should help Vi/Zh but NOT Ko. If it also helps Ko, our "tone-anchored" interpretation is wrong; if it helps only Vi/Zh, the linguistic-causality story is reinforced. This is a free experimental-design win — exploit it in the paper.
- **Cross-lingual transfer benchmark**: FLEURS is the *only* dataset with n-way parallel speech across Vi/En/Zh/Ko. Use FLEURS as the primary cross-lingual eval table; cascade English-pivot for direct S2ST where no Vi↔{Zh,Ko} corpus exists.
- **Industrial noise gap**: no Vietnamese factory-speech corpus exists publicly. **Synthesize** via DEMAND `PSTATION`+`DKITCHEN` + MUSAN at SNR ∈ {0, 5, 10, 15} dB over VIVOS / FLEURS-vi. Document the recipe as a method-section caveat — reviewers will accept it as long as it's reproducible.

**Experiments**: quantization-config × layer-type grid → metrics (WER, tone-error-rate, BLEU, COMET, latency-p99, model-size-MB). Roughly 2–3 weeks of GPU.

**Hard pre-work (do not skip)**:
- Build **tone-error-rate (TER)** metric — automatic tonal phoneme alignment + tone label comparison. Standard WER alone won't show the contribution. Budget **2 weeks** for this prep.
- Implement **tone-bearing-layer probing** — diagnostic to identify which encoder layers encode tone (linear probing on tone labels). This is the empirical evidence we need to defend the selective scheme.

**Risk**: if tone-bearing layers turn out to be evenly distributed, the "selective" angle weakens. Plan B is Method 2 below.

### ⚠️ Saturation pivot (2026-05-26, from [papers/reading_list_2025_2026.md](papers/reading_list_2025_2026.md))

**VIVOS clean WER is saturated at 4.67%** (PhoWhisper-large). Reviewers will reject "lose only +1% WER under INT4" as within noise floor on a saturated benchmark. So:

1. **Primary metric is TER, not WER.** TER has no canonical leaderboard — that's where our claim must land. Headline number: TER stays within +3pp of FP32 at 60–70% size reduction.
2. **Primary WER table runs on noisy/conversational subsets**, where there is real headroom (15–30% WER range): noise-augmented VIVOS (DEMAND `PSTATION`+`DKITCHEN` + MUSAN at SNR ∈ {0, 5, 10, 15} dB), FLEURS-vi (accent variance), VietSuperSpeech conversational test (arXiv 2603.01894).
3. **VIVOS-clean is demoted to a "no-harm sanity check" table** — show our method does not degrade the saturated benchmark.
4. **Korean (FLEURS-ko) stays as the non-tonal control** — method must NOT help Ko at the same magnitude it helps Vi/Zh. If it does, our linguistic-causality story is wrong.

This is the single most important methodological adjustment before drafting IP&M. Forgetting it produces a paper that looks like a marginal-WER engineering report instead of a method paper.

### 🥈 Backup spine: **Code-switching-aware streaming decoder for industrial Vi-En speech**
Use only if Primary fails. Insight: streaming chunk boundaries cut foreign-language tokens mid-word; predict language-tag in parallel head, adaptively shift chunk boundaries. Higher risk because requires curating Vi-En code-switching corpus from industrial recordings (drifts toward benchmark territory — avoid unless Primary collapses).

### 🥉 Stretch spine: **Cross-lingual phoneme-level distillation for Vi↔{En, Zh, Ko} S2ST**
Distill SeamlessM4T-Large into < 200M student via shared phoneme-space alignment. Defendable but risky — the "shared phoneme space across Vi/Zh/Ko works" claim must be empirically verified and might fail.

### Topics we are NOT pursuing
- Public benchmark / leaderboard release (anh decided 2026-05-26: not a benchmark paper)
- Participatory / community co-design study (would need a full HCI field study — out of scope for one PhD timeline)
- Latency–accuracy Pareto curve as standalone contribution (becomes an *ablation table inside* the method paper, not the spine)

### Method-paper review checklist (read before every IP&M/TASLP draft revision)
- [ ] Single-sentence method statement at start of intro?
- [ ] Clear ablation isolating the proposed mechanism vs every alternative?
- [ ] Comparison to **at least 2** prior compression/quantization papers, with our delta stated explicitly?
- [ ] Generalization experiment to **at least one** non-Vietnamese tonal language (Mandarin or Thai)?
- [ ] Failure-case analysis (where the method does NOT help)?
- [ ] All numbers with **bootstrap CI (≥ 1000 resamples)** + **paired permutation test** vs baseline?
- [ ] Latency reported as **p50/p95/p99**, never mean?
- [ ] Anonymized — no author names, no GitHub URLs, no acknowledgments?

---

## 5. Folder Layout (to be created as we go)

```
VoiceAI_SaiGonHub/
├── CLAUDE.md                  ← this file
├── Challenges_Details.md      ← scraped competition brief
├── docs/
│   ├── proposal/              ← Phase 2 technical spec (July 2026)
│   ├── paper/                 ← IP&M / *ACL manuscript (LaTeX)
│   └── slides/                ← pitch deck, finale Beamer
├── src/
│   ├── asr/                   ← Whisper / PhoWhisper wrappers
│   ├── mt/                    ← NLLB / SeamlessM4T runners
│   ├── tts/                   ← Piper / VITS
│   ├── pipeline/              ← end-to-end glue + VAD
│   ├── quantize/              ← QNN SDK / AI Hub scripts
│   └── eval/                  ← BLEU, COMET, ASR-WER, latency probes
├── data/
│   ├── raw/                   ← recordings, parallel corpora
│   ├── processed/
│   └── benchmarks/            ← BLEU/COMET/MOS test sets
├── experiments/               ← run logs, ablations
├── figures/                   ← paper-ready PDFs (rqX/filename.pdf)
└── device/                    ← Snapdragon device deployment configs
```

---

## 6. How To Work With Me On This Project

(These build on, and do not contradict, my global preferences in `~/.claude/CLAUDE.md`.)

### Language
- **Discussion / explanations**: Vietnamese (technical terms left in English).
- **Code, LaTeX, paper text, commit messages, identifiers**: English.

### Depth & honesty
- I want to understand **why** — connect every choice to theory (e.g., why INT4 specifically hurts code-switching: low-rank attention degradation under cross-lingual token interference).
- Tell me **weaknesses before reviewers do**. If a claim is shaky, flag it inline.
- **Storytelling first** — both the paper and the pitch must have a narrative arc, not a bullet list of metrics.
- **Concrete numbers always** — BLEU score, COMET, WER, latency ms, model size MB, MOS. Never "fast" or "accurate" without a number.

### Code style
- Minimal comments. Code should read like prose; identifiers carry meaning. Only comment **why**, never **what**.
- No premature abstraction. A duplicated 3-line block is fine; a half-finished factory pattern is not.
- No trailing summaries after edits — I read the diff.

### LaTeX workflow
- Springer SVJour3 for IP&M (Elsevier `cas-dc.cls` actually — confirm at submission time).
- Always compile **twice** for TOC/refs.
- Report errors only, suppress chktex warnings in summary.
- Figures path: `../figures/rqX/filename.pdf` (consistent with my AI_Dev_Dataminning project).

### Statistics / evaluation
- Standard MT metrics: **BLEU + chrF + COMET-22** (COMET is now expected by reviewers, BLEU alone is too weak).
- ASR: **WER + CER** (CER matters more for Vietnamese tone marks).
- TTS: **MOS** (subjective, n ≥ 20 raters), plus objective **MCD** if budget tight.
- Latency: report **p50, p95, p99** — never just mean. Median lies less than mean on long-tail systems.
- Always **bootstrap CIs** (≥ 1000 resamples). Significance with **paired permutation test** for MT system comparison (sacreBLEU built-in).

### Anonymization (IP&M is double-blind)
- Keep an `anon/` branch ready. Strip author names, institution names, GitHub URLs, dataset URLs that reveal identity, acknowledgments. Use neutral placeholders.

---

## 7. Risk Register (revisit monthly)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Model can't be quantized to fit Hexagon NPU | High | Critical | Start with AI Hub pre-optimized models; benchmark Day 1 |
| Vietnamese tone errors after INT4 quantization | High | High | Mixed-precision on phoneme-sensitive layers |
| Industrial noise destroys ASR | High | High | Train/fine-tune with augmented factory noise; speech enhancement front-end |
| Latency > 2 s on real device | Medium | Critical | Streaming ASR + chunk-based MT; cache phrase units |
| Code-switching breaks pipeline | High | Medium | Detection + language-tagging head |
| Paper rejected for engineering-only contribution | Medium | High | Method spine locked in Section 4 — defended via linguistic-property-aware framing |
| **Primary method spine collapses** (tone-bearing layers not isolated on Whisper) | **High** | High | de la Fuente & Jurafsky (Interspeech 2024) localized tone to layer ≈8 in wav2vec2/HuBERT/WavLM, **NOT Whisper**. Whisper trains differently (supervised, multilingual, log-Mel). Replicate the probe on Whisper-small / PhoWhisper-small **in the first 2 weeks**. If tone is diffuse → pivot to backup spine. |
| **AWQ already protects the right layers by accident** | Medium | **Critical** | If AWQ activation-magnitude salience correlates >0.7 with tone-probe salience, our method collapses into "AWQ with extra steps." Required exp: measure both, plot correlation, show divergence translates to fewer tone-confusions at matched compression. |
| **IP&M reviewer asks for perceptual eval** | High | High | Tone-WER on ASR transcripts is proxy. Need native-Vietnamese MOS panel (n ≥ 20) comparing FP32 vs quantized output speech with tone-naturalness subscore. Budget this from Phase 3 onward. |
| **Engineering ship vs method novelty tension** | High | High | Method = critical path of engineering. If we ever feel "doing research separate from product," stop and re-align. |
| Xu et al. 2025 (arXiv 2501.03643) is the closest competitor | High (already known) | Medium | They use Gumbel-Softmax DARTS over {2,4,8}-bit; our delta is "predictive interpretability via linguistic probe" — see [papers/lit_scan_v1.md](papers/lit_scan_v1.md) §4 for the rehearsed defense. |
| Conflict between competition IP rules & paper anonymity | Medium | Medium | Read Section 7 of FAQ; pre-clear with organizers |

---

## 8. Active TODO Surface (high-level — concrete tasks live in TodoWrite within sessions)

- [ ] Phase 1 (now): finalize team registration
- [ ] Set up Qualcomm AI Hub account, run Whisper-small + NLLB-distilled INT8 on a Snapdragon dev kit
- [x] Decide research spine — **LOCKED 2026-05-26: tone-aware mixed-precision quantization (Method 1, Section 4)**
- [x] Literature scan v1 — see [papers/lit_scan_v1.md](papers/lit_scan_v1.md). Defense paragraph drafted. 3 critical risks surfaced.
- [x] Reading list 2025-2026 — see [papers/reading_list_2025_2026.md](papers/reading_list_2025_2026.md). Saturation pivot to TER as primary metric. FADE + SeaLLMs-Audio + VietASR + Qwen3-ASR added as mandatory baselines.
- [ ] Implement **noise-augmentation pipeline** (DEMAND `PSTATION`+`DKITCHEN` + MUSAN at SNR ∈ {0, 5, 10, 15} dB over VIVOS / FLEURS-vi) — required to get a real-headroom WER table
- [ ] **exp04: Replicate Mandarin tone-layer probe on Whisper / PhoWhisper** (GO/NO-GO gate — target week of 2026-06-08)
- [ ] **exp05: AWQ-salience vs tone-probe-salience correlation experiment** (GO/NO-GO gate — target week of 2026-06-15)
- [ ] **exp00: Tone-Error-Rate (TER) metric** — forced-aligned tonal phoneme pipeline
- [ ] **exp01–03: FP32 / uniform INT8 / uniform INT4 baselines** — establish the WER/TER collapse the method must fix
- [ ] Draft Phase 2 Technical Specification (due July 2026) — fold method into the spec to satisfy "Innovation 25%" judge weight
- [ ] Begin IP&M paper outline (outline first, results later) — use defense paragraph from lit_scan_v1.md as intro seed
- [ ] **exp06: FADE replication + comparison** — show our linguistic-probe allocation beats FADE's diagnostic-coefficient allocation at matched compression ratio. FADE ([arXiv 2601.02455](https://arxiv.org/html/2601.02455v2)) is the most architecturally similar competitor; this is the single most important new ablation surfaced by the reading list.

---

## 9. Reference Stack

- Qualcomm AI Hub: https://aihub.qualcomm.com
- Qualcomm QNN SDK docs
- SeamlessM4T paper (Meta, 2023): for E2E S2ST baseline framing
- PhoWhisper: VinAI's Vietnamese-optimized Whisper
- IP&M Author Guide: https://www.sciencedirect.com/journal/information-processing-and-management/publish/guide-for-authors

---

*Last updated: 2026-05-26 — locked method-paper spine (Section 4 Method 1) + venue priority IP&M → TASLP → conferences*
