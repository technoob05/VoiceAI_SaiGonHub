# Reading List 2025-2026 — Tone-aware Quantization for Vietnamese S2ST

*Compiled: 2026-05-26. Curated supplement to [lit_scan_v1.md](lit_scan_v1.md) — only NEW 2025/2026 papers and saturation notes. Quality > quantity (28 papers).*

## TL;DR
- **Field is not saturated for our angle, but the moat is shrinking fast.** Generic ASR quantization (Edge-ASR, Quantizing Whisper, Towards-1-bit ASR, Ultra-Low-Bit PTQ Whisper) hit benchmark in mid-2025; *none* segment data by language or evaluate tonal preservation. The tone-aware niche is still open — but two 2025-2026 papers (CantoASR, "Lexical Tone is Hard to Quantize") now occupy adjacent ground.
- **Biggest 2025-2026 development**: SeaLLMs-Audio (arXiv 2511.01670, Mar 2025 release / Nov 2025 paper) is the first multilingual SEA audio-LLM covering Vietnamese — it changes the "no Vietnamese foundation model beyond PhoWhisper" framing in lit_scan_v1. Also Qwen3-ASR (Jan 2026, supports Vi via Qwen3-Omni) becomes a new mandatory baseline.
- **Biggest risk to our method paper**: arxiv 2604.07467 ("Lexical Tone is Hard to Quantize" — Apr 2026, **Mandarin + Yorùbá**) argues that DSU codebook quantization fails to preserve tone. It is *not* model-weight quantization, so our novelty survives — but reviewers will demand explicit framing of *which kind* of quantization we mean. Frame this in the intro to pre-empt.

---

## 1. Vietnamese ASR / S2ST 2025-2026

- [2511.01670] **SeaLLMs-Audio: Large Audio-Language Models for Southeast Asia** (Nov 2025, arXiv; DAMO-NLP-SG). [link](https://arxiv.org/abs/2511.01670)
  - First multilingual audio-LLM tailored for Indonesian/Thai/Vietnamese + En/Zh. Supports ASR + S2TT + speech QA. **Becomes a mandatory baseline for our paper** alongside PhoWhisper. SeaLLMs-Audio-7B released Mar 2025.
- [2505.21527] **VietASR: Industry-level Vietnamese ASR with 50h labeled + Large-Scale Pretraining** (Interspeech 2025; Zhuo et al.). [link](https://arxiv.org/abs/2505.21527)
  - 68M Zipformer trained via ASR-biased SSL on 70k h unlabeled Vi data; **claims to outperform Whisper-large-v3 with 22× fewer params** on real-world Vi. Strong new compact baseline for edge deployment.
- [2603.14779] **Vietnamese ASR: A Revisit** (Mar 2026, arXiv). [link](https://arxiv.org/abs/2603.14779)
  - Unified 500h cleaned Vi corpus + processing pipeline. Useful as data-aggregation reference; complements Bud500.
- [2603.01894] **VietSuperSpeech: Large-Scale Vietnamese Conversational Speech** (Feb 2026, arXiv). [link](https://arxiv.org/abs/2603.01894)
  - 267h conversational Vi (YouTube). Fills the "casual speech" gap that VIVOS/VLSP miss. Worth adding to our eval set as out-of-domain robustness test.
- [2509.05983] **TSPC: Two-Stage Phoneme-Centric Vi-En Code-Switching ASR** (Sep 2025 → Mar 2026 update). [link](https://arxiv.org/abs/2509.05983)
  - 19.06% WER on Vi-En code-switch, beats PhoWhisper-base. Important if we add code-switching robustness section (industrial use case).
- [2406.11546] **GigaSpeech 2** (ACL 2025; 30k h Th/Id/Vi from YouTube). [link](https://arxiv.org/abs/2406.11546)
  - 6000h refined Vi pseudo-labeled corpus → 25-40% WER reduction over Whisper-large-v3 with 10% params. Already mentioned in lit_scan_v1 only briefly; revisit for training-data discussion.
- [2505.07583] **Privacy-Preserving Real-Time Vi-En Translation on iOS Edge** (May 2025). [link](https://arxiv.org/abs/2505.07583)
  - TinyLlama-1.1B GGUF quantized on iPhone — already in lit_scan_v1 §2 but **becomes our most direct mobile-deployment competitor**. Their compression is uniform; ours is tone-aware. Cite as motivation.
- [2603.10211] **ViDia2Std: Vi Dialect-to-Standard Translation** (Mar 2026). [link](https://arxiv.org/pdf/2603.10211)
  - Dialect robustness angle. Optional citation for limitations section.
- **ChunkFormer** (ICASSP 2025; khanld). [github](https://github.com/khanld/chunkformer)
  - Masked-chunk Conformer for long-form Vi ASR (16h audio). Used as WER metric in VLSP 2025 Challenge. Strong streaming-friendly architecture worth benchmarking against.

## 2. Tonal-language ASR & Quantization 2025-2026

- [2511.04139] **CantoASR: Prosody-Aware ASR-LALM Collaboration for Low-Resource Cantonese** (Nov 2025). [link](https://arxiv.org/abs/2511.04139)
  - LoRA-Whisper + Qwen-Audio prosody-aware correction. **Closest neighbor in tonal-language space** — but works at *post-correction* level, not at *quantization-allocation* level. Distinguish in our related-work.
- [2604.07467] **Lexical Tone is Hard to Quantize: Probing DSU in Mandarin and Yorùbá** (Apr 2026). [link](https://arxiv.org/abs/2604.07467) **⚠ RED-FLAG**
  - K-means / VQ codebook quantization prioritizes phones over tones; proposes two-stage residual clustering. **NOT weight quantization** — but reviewers may ask "is this your contribution?". Pre-empt by framing our work as *transformer weight precision* vs *speech-unit codebooks*.
- LoRA-INT8 Whisper Cantonese (MDPI Sensors, Sep 2025). [link](https://www.mdpi.com/1424-8220/25/17/5404)
  - Already in lit_scan_v1; reaffirm — *uniform* INT8, no per-layer tone analysis. Authors explicitly cite "future work: tone-aware loss functions" — leaves the door open for our method.
- [2601.09050] **SITA: Speaker-Invariant Tone-Aware Speech Representations for Low-Resource Tonal Languages** (Jan 2026; Hmong + Mandarin). [link](https://arxiv.org/pdf/2601.09050)
  - Lightweight wav2vec adaptation with tone-repulsive loss. **Not quantization** — but methodologically adjacent: linguistic property steers training. Strong analogy to cite as related work.
- Phoneme-Aware Hierarchical Augmentation for Cantonese ASR (PMC, 2025). [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12298586/)
  - Cantonese data augmentation; supports the "phonological structure matters" motivation but not a competitor.
- [2406.04595] **Pitch-Aware RNN-T for Mandarin MD&D** (2024 still active; cited 2025-2026). [link](https://arxiv.org/html/2406.04595v1)
  - Explicit pitch features improve tonal discrimination. Background citation.

## 3. On-device Speech Translation / Edge Deployment 2025-2026

- [2604.14493] **Pushing the Limits of On-Device Streaming ASR** (Microsoft CoreAI, Apr 2026). [link](https://arxiv.org/abs/2604.14493)
  - Benchmarks 50+ ASR configs incl. Whisper/Nemotron/Parakeet/Canary/Conformer/Qwen3-ASR. **INT4 k-quant** recommended; 2.47 GB → 0.67 GB; 8.20% avg WER; 0.56s algorithmic latency. **English only** — perfect template + clean gap to extend to Vi.
- [2507.07877] **Edge-ASR: Towards Low-Bit Quantization of ASR** (Jul 2025; Whisper + Moonshine; 8 PTQ methods × 7 datasets). [link](https://arxiv.org/abs/2507.07877)
  - Comprehensive PTQ benchmark; finds 3-bit feasible on large models. **No tonal language eval.** Mandatory baseline reference.
- [2508.13358] **Overcoming Latency Bottlenecks in On-Device Speech Translation: Cascaded + Alignment-Based Streaming MT** (Aug 2025). [link](https://arxiv.org/pdf/2508.13358)
  - Cascaded ASR→MT streaming approach. Directly relevant to our pipeline framing.
- [2504.15509] **SimulS2S-LLM: Simultaneous Speech-LLM for S2ST** (ACL 2025). [link](https://arxiv.org/abs/2504.15509)
  - +3 BLEU at matched latency on CVSS. Establishes 2025 simul-S2ST SOTA.
- [2605.14766] **Streaming Speech-to-Text Translation with a SpeechLLM** (May 2026). [link](https://arxiv.org/abs/2605.14766)
  - 1-2s latency streaming S2TT; LLM decides when to emit. Bleeding-edge baseline.
- [2601.22779] **Streaming ASR with Decoder-Only LLMs + Latency Optimization** (Jan 2026). [link](https://arxiv.org/html/2601.22779v1)
  - Latency engineering for LLM-based streaming ASR.
- [2510.06126] **lm-Meter: Runtime Inference Latency for On-Device LMs** (Oct 2025). [link](https://arxiv.org/pdf/2510.06126)
  - Profiling methodology — useful for our p50/p95/p99 reporting protocol on Hexagon NPU.
- **On-Device LLMs: State of the Union 2026** (v-chandra). [link](https://v-chandra.github.io/on-device-llms/)
  - Non-arXiv survey but cited everywhere; benchmarks Snapdragon vs MediaTek vs Apple Silicon; 50-90 GB/s mobile bandwidth vs 2-3 TB/s data-center. Good systems context.

## 4. Mixed-precision Quantization for Speech Foundation Models (post Xu 2025)

- [2511.08093] **Quantizing Whisper: How Design Choices Affect ASR Performance** (Nov 2025). [link](https://arxiv.org/html/2511.08093)
  - Systematic ablation: bit-width × granularity × dynamic/static × library (PyTorch/Quanto/HQQ/bitsandbytes) on Whisper-small. **English only — explicitly flagged as limitation**. Our work fills that void.
- [2601.02455] **FADE: Diagnostic-Driven Layer-Wise Compensation for PTQ of Encoder-Decoder ASR** (Jan 2026). [link](https://arxiv.org/html/2601.02455v2)
  - Per-layer α-coefficient combining weight-geometry + calibration-reliability scores. **Methodologically the most architecturally similar competitor**: assigns per-layer treatment, but criterion is statistical/diagnostic, *not linguistic*. Our delta is preserved (cite directly in defense paragraph).
- Ultra-Low Bit PTQ of Large Speech Models (Gu et al., Interspeech 2025). [link](https://www.isca-archive.org/interspeech_2025/gu25b_interspeech.pdf)
  - Mixed-precision + selective outlier retention + K-means on Whisper. Already partly in lit_scan_v1 as "Towards One-bit ASR" — different paper, similar bit budget. Add as second 2025 mixed-precision speech reference.
- [2511.06516] **TAQ: Per-Task Quantization Using LLMs' Hidden Representations** (Nov 2025, updated May 2026). [link](https://arxiv.org/abs/2511.06516) **⚠ FRAMING-RISK**
  - Task-aware per-layer bit allocation. **A reviewer will ask: "is yours TAQ with task=tone-preservation?"** Defense: TAQ conditions on task *loss*; we condition on a *mechanistic linguistic-probe* signal validated against typology. Add an explicit subsection in related-work distinguishing the two.
- [2504.21553] **Spike-Aware Mixed-Precision Quantization for LLaMA** (Apr 2025). [link](https://arxiv.org/html/2504.21553v1)
  - Outlier/spike-based selection. Generic-LLM angle but cite for completeness as a 2025 mixed-precision criterion.

## 5. Benchmark saturation snapshot

| Benchmark | Best WER/BLEU as of 2026-05 | Paper / source | Headroom for our claim? |
|---|---|---|---|
| **VIVOS test (Vi WER)** | **4.67% WER** — PhoWhisper-large; **~5-7%** Whisper-large-v3; VietASR-68M competitive | PhoWhisper README [link](https://github.com/VinAIResearch/PhoWhisper) | **LOW headroom on WER**. <5% is near read-speech ceiling for Vi; quantization gains will be sub-1pp and noisy. **Pivot to Tone-Error-Rate (TER) and conversational/noisy subsets.** ⚠ |
| **FLEURS-vi (Vi WER)** | No single canonical 2026 number; Whisper-large-v3 reported ~13-15% WER range (HF community), VietASR claims to beat it on real-world data; GigaSpeech-2-trained Zipformer 25-40% lower than Whisper-large-v3 | [Whisper repo](https://github.com/openai/whisper); [VietASR](https://arxiv.org/abs/2505.21527); [GigaSpeech2](https://arxiv.org/abs/2406.11546) | **MODERATE headroom**. FLEURS-vi has accent variation; quantization degradation should be measurable here. Use as primary cross-lingual table. |
| **AISHELL-1 (Zh CER)** | **~4.0-4.5% CER** Paraformer; **~3.5% CER** Zipformer-large; phoneme-level Zipformer 1.92/2.12% (specialized) | [Zipformer 2310.11230](https://arxiv.org/abs/2310.11230); Paraformer 2206.08317 | **VERY LOW headroom on CER**. Use AISHELL-1 only for *tone-error specifically*, not aggregate CER. |
| **AISHELL-3 (Zh tone, TTS-eval reused for ASR-tone)** | Tone-label accuracy >98% on transcripts; per-tone confusion matrix data exists but no canonical leaderboard | [AISHELL-3 2010.11567](https://arxiv.org/pdf/2010.11567) | **OPEN headroom for tone-confusion-rate metric**. Build our TER metric here and show INT4 collapses specific tone pairs (e.g., tone-2/tone-3). |
| **FLEURS S2T BLEU (any → en)** | **~29.7 ASR-BLEU** SeamlessM4T-v2 X→en (S2ST); **+4.2 BLEU over AudioPaLM-2-8B-AST** on S2TT into-en | [SeamlessM4T-v2 / Nature 2024](https://www.nature.com/articles/s41586-024-08359-z) | **MODERATE**. BLEU is noisy enough that ±1 BLEU under quantization is plausible; pair with COMET. |
| **FLEURS-ko (Korean, our CONTROL)** | Whisper-large-v3 ~10-12% CER range; no canonical 2026 SOTA paper; HiKE [2509.24613](https://arxiv.org/pdf/2509.24613) is reference | [HiKE](https://arxiv.org/pdf/2509.24613) | **GOOD as control** — Korean non-tonal so our method should NOT help; this *protects* our causal story. |

### Saturation verdict per benchmark

- **VIVOS WER is effectively saturated** (PhoWhisper-large 4.67%, near-ceiling). A "we lose only +0.5% WER under INT4" claim is statistically uninteresting because **(a)** the baseline noise floor on VIVOS is already ~±0.3% across runs, and **(b)** reviewers can argue WER changes < 1pp are within annotation error.
- **CER on AISHELL-1 similarly saturated** (~3-4% Zipformer SOTA).
- **Tone-Error-Rate (TER) is NOT saturated** — there is no canonical TER leaderboard at all. This is where our claim must live.
- **Conversational / noisy subsets (VietSuperSpeech, noise-augmented VIVOS) have meaningful headroom** (~15-30% WER range), so quantization impact is measurable there.

---

## 6. Implications for our paper

### Should we re-baseline against [paper X]?
- **YES, urgently**: SeaLLMs-Audio-7B (2511.01670), VietASR-68M (2505.21527), Qwen3-ASR-1.7B (2601.21337), Edge-ASR Whisper-INT3 (2507.07877), and FADE (2601.02455) all post-date lit_scan_v1. At least three of these should appear in our Table 1 baselines.
- **Mandatory new comparator**: FADE (2601.02455). It is the most architecturally similar competitor — per-layer treatment of encoder-decoder ASR PTQ. We must show our linguistic-probe-driven allocation beats FADE's diagnostic-coefficient allocation at matched compression ratio. **This is the single most important new ablation.**
- **Mandatory framing move**: TAQ (2511.06516) — pre-empt the "task-aware quantization" question with one explicit related-work paragraph distinguishing task-loss criterion vs linguistic-probe criterion.

### Does any 2025-2026 paper undermine our novelty?
- **No paper does what we propose** (linguistic-property-conditioned per-layer bit allocation for speech foundation models). Novelty intact.
- **Three adjacent papers are dangerously close** and require crisp differentiation:
  1. **CantoASR** (2511.04139) — tone-aware but *post-correction*, not quantization. Differentiation easy.
  2. **Lexical Tone is Hard to Quantize** (2604.07467) — codebook DSU quantization, NOT weight quantization. Differentiation easy but must be stated explicitly in §2.
  3. **FADE** (2601.02455) and **TAQ** (2511.06516) — per-layer / task-aware mixed precision. Differentiation harder; requires our delta to be "**predictive interpretability via linguistic probe**" (already drafted in lit_scan_v1 §4 defense paragraph). Reinforce.

### Is the WER ceiling so low on VIVOS that quantization claims look like noise? **YES.**
- VIVOS 4.67% WER is near read-speech ceiling. Reviewers will reject "WER stays within +1%" as uninteresting.
- **Recommended pivot** (do this BEFORE first IP&M draft):
  1. **Reframe primary metric as Tone-Error-Rate (TER)** with bootstrap CI, computed on tone-bearing syllables only. This is a metric novelty AND avoids the VIVOS ceiling.
  2. **Use noisy/conversational eval sets** as the primary WER comparison: noise-augmented VIVOS (DEMAND + MUSAN, SNR 0-15 dB), VietSuperSpeech conversational test, and FLEURS-vi (accent variance) — all have headroom 15-30% WER where quantization impact is detectable.
  3. **Keep VIVOS-clean only as a "sanity check / saturation table"** — show our method does no harm there.
- This pivot is **the single most important methodological adjustment** suggested by this scan. Without it, the paper's empirical contribution looks negligible on the headline benchmark.

### Other things to track
- **AISHELL-5** (in-car multi-channel Mandarin) — emerging benchmark with real-world noise. Worth a sentence as future-work.
- **Open ASR Leaderboard (HF)** added multilingual tracks Nov 2025; Vietnamese not yet in the top-5 tracked languages but watch for community-PR addition.
- Papers with Code shutdown (Jul 2025) means leaderboards now scattered across HF Spaces, OpenReview, and project READMEs — cite directly to source paper, not PwC.

---

*Sources are inline. No further write-ups needed — this is a triage list, not a full literature review.*
