# Literature Scan v1 — Tone-aware Mixed-precision Quantization for Vietnamese S2ST

*Compiled: 2026-05-26*

## TL;DR (3 bullets)
- Mixed-precision quantization SOTA for transformers/LLMs allocates bits per layer using **purely statistical signals** — activation magnitude (AWQ), Hessian trace / second-order curvature (SqueezeLLM, HAWQ-style), gradient / first-order sensitivity (GWQ, LLM-MQ, GenPTQ), KL-divergence (Xu et al. 2021), or learned DARTS-style search (Xu et al. 2025); **none use linguistic properties of the layer's representation as the criterion**.
- Speech-model quantization (Whisper, Conformer, NLLB inside SeamlessM4T) is now routine down to 4-bit and even 2-/1-bit (Li/Xu et al. Interspeech 2025); for Whisper, INT8 dynamic PTQ gives ~4× size reduction with ~RTF 0.20 on M1 and INT4 is feasible at modest WER cost — but tonal-language WER degradation is **not analyzed by tone vs. segment** in any of the reviewed works.
- **The novelty gap is real**: after searching arXiv, ACL Anthology, ISCA Archive, and Google Scholar, no paper allocates quantization precision to transformer layers based on the *linguistic content* (tone, phoneme, prosody) that the layer encodes. The closest neighbors are (a) ToneUnit (Interspeech 2024) — vector-quantizing speech units, not model weights; (b) Effective and Efficient Mixed Precision Quantization of Speech Foundation Models (Xu et al. 2025) — automated DARTS search over bit-widths but criterion is loss/sensitivity, not linguistic role.

## 1. Mixed-precision Quantization SOTA (7 papers)

| Year | Paper | Venue | Precision-selection criterion | Why relevant |
|---|---|---|---|---|
| 2022 | SmoothQuant (Xiao et al.) | ICML 2023 | Per-channel activation scaling; uniform W8A8 — no mixed precision per layer | Outlier handling; baseline anyone in this space must cite |
| 2023 | AWQ (Lin et al.) | MLSys 2024 (Best Paper) | Activation magnitude → identifies 1% salient weight channels for FP16 protection; rest INT4 | Closest "salience"-based selection; criterion is statistical (activation), not linguistic |
| 2023 | GPTQ (Frantar et al.) | ICLR 2023 | Layer-by-layer Hessian-based reconstruction; uniform bit-width | Standard PTQ baseline |
| 2023 | SqueezeLLM (Kim et al.) | ICML 2024 | Sensitivity = Hessian-weighted k-means; mixed-precision via dense-sparse decomposition | Hessian-trace criterion; non-linguistic |
| 2023 | OmniQuant (Shao et al.) | ICLR 2024 | Learnable equivalent transformations and weight clipping; uniform bits | — |
| 2024 | SpinQuant (Liu et al.) | arXiv 2405.16406 | Learned rotation matrices; uniform W4A4KV4 | Rotation family; non-linguistic |
| 2024 | QuaRot / QServe | NeurIPS 2024 | Hadamard rotations + per-channel reorder | Rotation-based outlier suppression |
| 2024 | GWQ (Shao et al.) | arXiv 2411.00850 | Gradient-aware: top-1% gradient weights FP16, rest INT3/INT4 | First-order sensitivity; non-linguistic |
| 2024 | MixLLM | OpenReview 2024 | Per-layer + per-output-feature precision via loss sensitivity | Mixed precision SOTA; non-linguistic |
| 2024 | ResQ (Saxena et al.) | arXiv 2412.14363 | Low-rank residual mixed-precision | Non-linguistic |
| 2025 | "Mixed-Precision Quantization for Language Models: Techniques and Prospects" | arXiv 2510.16805 | Survey | Anchor citation; confirms criteria taxonomy |

**Synthesis**: every criterion in the literature reduces to one of {weight magnitude, activation magnitude, Hessian/curvature, gradient, KL-divergence, learned DARTS}. None is anchored in what the layer *represents linguistically*.

## 2. Speech-model Quantization (6 papers)

| Year | Paper | Venue | Method | Findings relevant to us |
|---|---|---|---|---|
| 2021 | Xu et al., "Mixed Precision Low-bit Quantization of NN LMs for Speech Recognition" | arXiv 2112.11438 / ICASSP 2022 | KL-div + Hessian + NAS for per-layer bits | First serious mixed-precision for ASR; criterion = sensitivity, not linguistic |
| 2022 | "4-bit Conformer with Native QAT" | Interspeech 2022 / arXiv 2203.15952 | QAT, uniform W4A4 | Establishes 4-bit Conformer is viable |
| 2023 | "Lossless 4-bit Quantization of Architecture-Compressed Conformer ASR" (Xu et al.) | ICASSP/ASRU 2023 | Layer pruning + 4-bit QAT | Lossless on Switchboard 300h |
| 2024 | "Quantization for OpenAI's Whisper Models" | arXiv 2503.09905 | INT8/INT5/INT4 PTQ on whisper.cpp | 19% latency / 45% size; tone-language WER not isolated |
| 2025 | Xu et al., "Effective and Efficient Mixed Precision Quantization of Speech Foundation Models" | arXiv 2501.03643 | Gumbel-Softmax DARTS over {2,4,8}-bit + KL regularization | **Closest competitor**: automatic per-layer bit assignment for speech FMs — but criterion is loss/complexity, not linguistic |
| 2025 | "Towards One-bit ASR" (Li, Xu, et al.) | Interspeech 2025 / arXiv 2505.21245 | 1-/2-bit Conformer via co-training + stochastic precision | Pushes extreme low-bit on Switchboard/LibriSpeech (English only) |
| 2025 | "LoRA-INT8 Whisper: Low-Cost Cantonese ASR for Edge Devices" | Sensors / MDPI 2025 | LoRA fine-tune + INT8 PTQ on Cantonese (9 tones) | Tonal language + edge, but uniform INT8; no tone-specific layer treatment |
| 2025 | "Green Post-Training Quantization for Large-Scale ASR" | Findings EMNLP 2025 | Sensitivity-aware PTQ on Whisper/Moonshine/Qwen3-ASR | Strong baseline; criterion = sensitivity |
| 2025 | "Diagnostic-Driven Layer-Wise Compensation for PTQ of Encoder-Decoder ASR" | arXiv 2601.02455 | Per-layer compensation post-quantization | Layer-wise but not linguistic |
| 2025 | "Efficient Speech Translation through Model Compression and KD" | arXiv 2505.20237 | QLoRA 4-bit + layer pruning + KD on speech translation | Speech translation specifically; uniform 4-bit |

## 3. Tonal-language ASR & Layer Probing (5 papers)

| Year | Paper | Venue | Does it isolate tone-encoding layers? |
|---|---|---|---|
| 2024 | PhoWhisper (Le et al., VinAI) | arXiv 2406.02555 / ICLR Tiny Papers 2024 | No — fine-tunes Whisper on 844h Vietnamese; no per-layer tone analysis. **Our backbone of choice.** |
| 2024 | de la Fuente & Jurafsky, "A layer-wise analysis of Mandarin and English suprasegmentals in SSL speech models" | Interspeech 2024 / arXiv 2408.13678 | **Yes — the key empirical anchor for our claim.** Mandarin tone peaks at **layer 8** in wav2vec2/HuBERT/WavLM; English stress/accent peaks at **layers 8–9**; layer 0 is language-neutral. Mid-layers encode abstract tone; not driven by raw F0. |
| 2023 | "What Can an Accent Identifier Learn? Probing Phonetic and Prosodic Information in Wav2vec2" (Yang et al.) | arXiv 2306.06524 | Yes — fine-tuning shifts phonetic/prosodic info into **layers 9, 11, 12**. |
| 2024 | ToneUnit (Du et al.) | Interspeech 2024 / arXiv 2406.08989 | **Not a quantization paper.** Uses CTC supervision to learn tone-aware *discrete speech units* for Mandarin TTS via Gumbel-VQ / FSQ. Important to disambiguate: their "quantization" = vector quantization of representations, not model weight precision. |
| 2025 | "A Computational Typological Analysis of Tone Function" | SIGTYP 2025 (ACL) | Confirms contour tones collapse toward level tones when F0 is degraded — supports our motivation. |
| 2024 | "Multi-Dialect Vietnamese: Task, Dataset, Baseline Models" | arXiv 2410.03458 | Useful for dialect robustness arguments. |

**Critical takeaway from probing literature**: there is published evidence that tone is concentrated in **mid encoder layers (≈8–9 of 12)** in SSL speech models. This is the empirical foothold for proposing a non-uniform, tone-anchored bit allocation. The Mandarin result must be replicated on Vietnamese / Whisper-family encoders before our paper can claim the same finding — **this replication is the most important experiment in the paper.**

## 4. Linguistic-property-aware Compression — THE NOVELTY GAP

**Verdict: the novelty claim survives, but narrowly.**

After exhaustive searches across arXiv, ACL Anthology, ISCA Archive, and Google Scholar using queries including `"linguistic-aware" quantization`, `"phoneme-aware" quantization`, `"tone-aware" quantization`, `"task-aware quantization" speech`, `"feature-aware" quantization tone`, `selective precision linguistic`, `layer-wise quantization phonetic`, and combinations of {Mandarin, Vietnamese, Thai, Yoruba, Cantonese, tonal} × {quantization, pruning, compression, mixed precision} — **no paper assigns per-layer weight precision based on the linguistic property (tone vs. phoneme vs. prosody) that the layer empirically encodes.**

The two pieces of work that come closest, and why each fails to anticipate us:

1. **Xu et al. 2025, "Effective and Efficient Mixed Precision Quantization of Speech Foundation Models"** (arXiv 2501.03643). They allocate {2, 4, 8}-bit per layer for SSL speech models, but the criterion is **automated Gumbel-Softmax DARTS over a complexity-constrained loss**. They do not interpret *why* layer L gets more bits — they just observe it does. Our delta: we *causally* tie the allocation to a measurable linguistic probe.
2. **ToneUnit (Du et al. 2024)**. "Tone-aware" refers to the *discrete unit codebook* for Mandarin TTS, not transformer weight precision. They quantize *what is spoken*, we quantize *the model that listens/speaks*.

Why this gap existed: the speech-quantization community is dominated by ML-systems researchers who treat the model as a generic transformer; the speech-science community (probing, phonetics) rarely engages with deployment. The bridge — using probing results to drive deployment decisions — is genuinely under-explored.

**Honest caveat #1**: "Task-Aware Quantization" (TAQ, arXiv 2511.06516) and prior task-conditioned PTQ work *do* condition precision on the downstream task. One could argue our work is "task-aware quantization where the task is tone-preservation." If a reviewer raises this, we need a clear distinction: TAQ conditions on task *loss*; we condition on a *mechanistic, probe-validated, language-typological* property. The framing must be precise.

**Honest caveat #2**: in the LLM literature, "salience-aware" quantization (AWQ) protects 0.1–1% of weights identified by activation magnitude. A reviewer can argue this *implicitly* protects whatever is linguistically important. Our defense: AWQ's salience is per-channel and statistical, blind to the linguistic category being protected. Our method names the category, predicts which layers, and verifies via probing. The novelty is **predictive interpretability of the bit allocation**, not just a new allocator.

## 5. Datasets relevant to our experiments

| Dataset | Language(s) | Size | License | HF / URL | Tonal annotation |
|---|---|---|---|---|---|
| VLSP 2020 / 2021 | Vietnamese | ~100h public, ~250h total | Research | vlsp.org.vn | Implicit (orthographic diacritics) |
| Common Voice 17 vi | Vietnamese | ~3–5h validated | CC0 | `mozilla-foundation/common_voice_17_0` | Orthographic |
| FLEURS-vi | Vietnamese | ~12h | CC-BY 4.0 | `google/fleurs` (config `vi_vn`) | Orthographic; aligned multilingual eval |
| VIVOS | Vietnamese | 15h, 65 spk, clean studio | Research | ailab.hcmus.edu.vn/vivos | Orthographic |
| Bud500 / VIET_BUD500 | Vietnamese | 500h | Research | linhtran92/viet_bud500 | Orthographic |
| VietMed | Vietnamese (medical) | ~16h | Research | `leduckhai/VietMed` | Orthographic |
| VinBigdata-VLSP2020 | Vietnamese | 100h | Restricted | — | Orthographic |
| AISHELL-1 / -3 | Mandarin (4 tones) | 170h / 85h | Apache 2.0 / CC-BY | `openslr/aishell` | Pinyin with tone marks |
| BSTC | Mandarin speech translation | 50h | Research | BSTC website | Pinyin tones |
| THCHS-30 | Mandarin | 30h | Apache 2.0 | openslr.org/18 | Tone-marked |
| Common Voice th | Thai (5 tones) | ~150h | CC0 | CV17 | Orthographic, ambiguous |
| Common Voice yo | Yoruba (3 tones) | small | CC0 | CV17 | Diacritic-marked |
| MuST-C v1.2 | EN→{vi, zh} ST | varies | CC-BY-NC-ND | mt.fbk.eu/must-c | n/a (text target) |
| CoVoST-2 | multilingual ST incl. vi-en, en-vi | ~30h vi-en | CC0 | `facebook/covost2` | n/a |
| FLEURS (full) | 102 langs incl. zh, th, yo | ~12h/lang | CC-BY 4.0 | `google/fleurs` | varies |

For the tone-isolation probe we need a corpus with **time-aligned tone labels**. Best candidates: AISHELL-3 (forced-aligned), THCHS-30, and for Vietnamese a custom forced alignment over VLSP/VIVOS using Montreal Forced Aligner with a Vietnamese tone-marked lexicon.

## 6. Defense paragraph (≈190 words) — for IP&M paper intro

> Mixed-precision quantization of transformer-based ASR and translation models has matured rapidly: methods such as AWQ [Lin et al., MLSys 2024], GPTQ [Frantar et al., ICLR 2023], SqueezeLLM [Kim et al., ICML 2024], and recent speech-specific work [Xu et al., arXiv 2501.03643; Li et al., Interspeech 2025] allocate per-layer bit-widths using activation magnitude, Hessian curvature, gradient sensitivity, or learned DARTS-style search. Without exception, these criteria are **statistical and linguistically opaque**: they identify *that* a layer matters, never *what linguistic structure* it carries. In parallel, layer-wise probing studies of speech models [de la Fuente & Jurafsky, Interspeech 2024; Yang et al., 2023] have shown that suprasegmental properties — and Mandarin lexical tone in particular — are concentrated in the middle layers (≈8–9 of 12) of wav2vec2/HuBERT/WavLM, distinct from the layers encoding segmental phonemes. No prior work bridges these two literatures. We propose **tone-anchored mixed-precision quantization**: we first localize the tone-encoding layers of a Vietnamese ASR/S2ST stack via supervised linear probing, then preserve those layers at INT8/FP16 while quantizing tone-irrelevant layers to INT4. The contribution is not a new allocator but a **predictive, mechanistically grounded selection criterion** that — unlike AWQ-style salience or ToneUnit-style speech-unit quantization [Du et al., Interspeech 2024] — names the linguistic category it protects and validates the protection causally on six Vietnamese tones, with cross-lingual transfer to Mandarin, Thai, and Yoruba.

### 6b. Supplementary defense paragraphs (added 2026-05-26 after reading_list_2025_2026.md scan)

**Defense against confusion with "Lexical Tone is Hard to Quantize" (arXiv 2604.07467, Apr 2026):**

> Recent work by Lee et al. ([arXiv 2604.07467](https://arxiv.org/abs/2604.07467), 2026) argues that lexical tone in Mandarin and Yorùbá is poorly preserved by discrete speech-unit (DSU) quantization — i.e., by k-means or vector-quantized codebooks operating in the *representation space* of self-supervised speech models. Our work addresses a categorically different problem: **transformer weight precision** (INT4/INT8 of model parameters) for inference compression, not codebook quantization of intermediate features. We adopt their finding as motivating evidence that "*all* forms of compression carry the risk of tonal information loss," but our method, evaluation protocol, and deployment scenario do not overlap with theirs. We name the distinction explicitly to prevent reader confusion: "quantization" in this paper means model-weight bit-width assignment.

**Defense against confusion with TAQ "Task-Aware Quantization" (arXiv 2511.06516, Nov 2025):**

> Mou et al. ([arXiv 2511.06516](https://arxiv.org/abs/2511.06516), 2025) propose Task-Aware Quantization (TAQ), which assigns per-layer bit-widths in LLMs by conditioning on downstream task-loss sensitivity. Our approach is superficially similar — both depart from uniform precision and exploit per-layer heterogeneity — but the *conditioning signal* differs in a way that matters for interpretability. TAQ uses task-loss as a black-box sensitivity proxy; the layer-level outcome is not predictable in advance from any model-independent property. We condition instead on a **mechanistic linguistic probe** (supervised tonal-classification accuracy per encoder layer), which is (i) validatable independently of any quantization decision, (ii) tied to a typologically meaningful property (lexical tone), and (iii) transferable across tonal languages (Vietnamese, Mandarin, Thai, Yorùbá) by design. Our contribution is therefore not a new allocator but a **predictive, mechanistically grounded selection criterion** — the bit-width assignment can be explained ex ante from typology, not merely measured ex post from a held-out loss.

---

## 7. Top 3 risks for our claim

1. **AWQ already protects the right things by accident.** If activation magnitudes happen to be high in the same mid-encoder layers that probing identifies as tone-bearing, then AWQ's INT4 quantization already preserves tone WER, and our "tone-aware" allocation collapses into a fancier re-derivation of an existing salience pattern. **Required experiment**: measure correlation between AWQ's per-layer salience and our probe-derived tone-saliency; show they diverge meaningfully *and* that the divergence translates to fewer tone-confusion errors at matched compression ratio.
2. **Probing localization is fragile across architectures.** The Interspeech 2024 result is for wav2vec2/HuBERT/WavLM. Whisper's encoder is trained differently (supervised, multilingual, log-Mel input) and may distribute tone information differently — possibly more diffusely across layers, making the "protect layer 8" recipe fail. **Required experiment**: replicate the de la Fuente & Jurafsky probe on Whisper encoders (tiny → large) for Vietnamese tones before committing to the method.
3. **The IP&M reviewer asks "where is the perceptual evaluation?"** Quantizing a speech-to-speech translation system can degrade tone in the *output speech*, not just intermediate representations. We must measure tone-naturalness in synthesized audio (MOS / tone-MOS with native listeners), not only character/tone error rate on ASR transcripts. Without a perceptual eval the contribution looks like an ASR paper in S2ST clothing. **Required experiment**: native-Vietnamese MOS panel comparing baseline vs. tone-anchored quantization at matched bitrate, with tone-error subscore.

---

### Sources / key URLs
- Mixed-precision survey: https://arxiv.org/html/2510.16805v1
- AWQ: https://github.com/mit-han-lab/llm-awq
- Xu et al. 2025 (speech FM mixed precision): https://arxiv.org/html/2501.03643v2
- Xu et al. 2021 (NN LM mixed precision for ASR): https://arxiv.org/abs/2112.11438
- Li/Xu et al. 2025 (1-bit ASR): https://arxiv.org/abs/2505.21245
- Whisper INT8/INT4: https://arxiv.org/pdf/2503.09905
- LoRA-INT8 Whisper Cantonese: https://www.mdpi.com/1424-8220/25/17/5404
- PhoWhisper: https://arxiv.org/abs/2406.02555
- de la Fuente & Jurafsky 2024 (layer-wise suprasegmentals): https://arxiv.org/html/2408.13678v1
- Yang et al. 2023 (accent ID probing): https://arxiv.org/html/2306.06524
- ToneUnit: https://arxiv.org/html/2406.08989v1
- SpinQuant: https://arxiv.org/abs/2405.16406
- GWQ: https://arxiv.org/html/2411.00850v1
- ResQ: https://arxiv.org/html/2412.14363v2
- Per-Task Quantization (TAQ): https://arxiv.org/pdf/2511.06516
- Green PTQ for ASR: https://aclanthology.org/2025.findings-emnlp.566.pdf
- Efficient Speech Translation via Compression+KD: https://arxiv.org/html/2505.20237v1
- Vietnamese iOS edge MT: https://arxiv.org/abs/2505.07583
