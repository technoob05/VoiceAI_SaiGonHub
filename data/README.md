# Dataset Registry — VoiceAI_SaiGonHub

*Compiled: 2026-05-26. All sizes / IDs verified via HuggingFace + paper sources at survey time — re-verify before downloading (especially row counts and total bytes, which change when dataset owners push updates).*

Project context: on-device Vietnamese-centric speech-to-speech translation (S2ST) for Vi↔En, Vi↔Zh, Vi↔Ko, using *tone-aware mixed-precision quantization*. Vietnamese is tonal (6 tones, orthographic diacritics), Mandarin is tonal (4+neutral, pinyin tones), Korean is non-tonal but has pitch accent in some dialects. The "tone" axis is therefore strongest on Vi and Zh.

---

## Priority tiers

**Tier 1 (must have, download immediately — small enough for fast TER / quantization-error experiments):**
1. `AILAB-VNUHCM/vivos` — 15 h Vi ASR, ~1.5 GB, CC-BY-SA-4.0. Tone via Vietnamese orthography.
2. `google/fleurs` (configs `vi_vn`, `cmn_hans_cn`, `ko_kr`, `en_us`) — ~10 h × 4 languages, ~12 GB total, CC-BY-4.0. **Parallel across all 4 of our languages** — best benchmark for cross-lingual transfer + a built-in MT eval (sentences come from FLORES).
3. `kresnik/zeroth_korean` — 51.6 h Ko ASR, ~5.5 GB, CC-BY-4.0.
4. `facebook/flores` / `openlanguagedata/flores_plus` — 3001 sentences × 200 langs (text only), <100 MB, CC-BY-SA-4.0. Anchor MT eval for Vi↔{En,Zh,Ko}.
5. `doof-ferb/infore1_25hours` — 25 h Vi single-speaker TTS, ~3 GB, CC-BY-4.0.
6. `AISHELL/AISHELL-3` — 85 h Mandarin multi-speaker TTS + ASR with **pinyin tone labels**, ~19 GB, Apache-2.0. **Critical for tone-aware evaluation on Zh side.**

**Tier 2 (need for ablations and stronger ASR/MT baselines):**
- `AISHELL/AISHELL-1` — 170 h Mandarin ASR, ~15 GB, Apache-2.0.
- `mozilla-foundation/common_voice_17_0` (config `vi`, `zh-CN`, `ko`, `en`) — Vi ~5 h validated, Zh-CN ~80 h, Ko ~1.5 h, En ~3300 h. Gated (free sign-in), CC0.
- `doof-ferb/vlsp2020_vinai_100h` — 100 h Vi ASR (VLSP 2020 redistribution), ~10 GB, research-only flags apply.
- `facebook/covost2` (config `en_zh-CN`, `zh-CN_en`, `en_ja`, etc.) — multilingual ST. **Note: no Vi pair** — use PhoST instead for Vi-En ST.
- `Helsinki-NLP/opus-100` (configs `en-vi`, `en-zh`, `en-ko`, plus `vi-zh` if available via Tatoeba) — ~1 M sentence pairs per direction, MIT/CC0-ish per-source.
- VinAIResearch/PhoST (off-HF, GitHub) — 508 h En-Vi ST triplets, research-only.

**Tier 3 (nice-to-have, large):**
- `linhtran92/viet_bud500` — 500 h Vi ASR (~100 GB), Apache-2.0. Best large-scale Vi ASR.
- `hynt/ZipVoice-Vietnamese-2500h` — 2500 h Vi TTS-grade audio (huge).
- `wenet-e2e/wenetspeech` — 10,000+ h Mandarin ASR (~1 TB after extraction), CC-BY-4.0 non-commercial.
- `allenai/nllb` — bitext mined from web (Vi-En, Vi-Zh, Vi-Ko subsets), each tens-hundreds GB.

---

## 1. Vietnamese ASR

| Name | HF ID | Size (compressed / extracted) | Hours | License | Tone annot? | Notes |
|---|---|---|---|---|---|---|
| VIVOS | `AILAB-VNUHCM/vivos` (mirror: `vivos`) | ~1.5 GB / ~2.5 GB | 15 h | CC-BY-SA-4.0 | Implicit (orthographic diacritics) | Read speech, 65 speakers (S Vietnamese). Canonical small benchmark. |
| Bud500 | `linhtran92/viet_bud500` | ~50 GB / ~100 GB (649K samples) | ~500 h | Apache-2.0 | Implicit (orthographic) | Multi-domain (podcast/travel/book/food), N/S/Central accents. |
| VLSP 2020 (VinAI redist) | `doof-ferb/vlsp2020_vinai_100h` | ~10 GB | 100 h | Research-only (VLSP terms) | Implicit | Subset of original 250 h VLSP 2020 train set. |
| VLSP 2020 (original) | n/a (Google Drive) | ~25 GB | 250 h | Research-only | Implicit | Manual DL: https://drive.google.com/file/d/1vUSxdORDxk-ePUt-bUVDahpoXiqKchMx |
| VLSP 2021 | n/a (vlsp.org.vn) | ~50+ GB | 280 h transcribed + 400 h untranscribed | Research-only | Implicit | Request via VLSP organizers. |
| Common Voice 17 (vi) | `mozilla-foundation/common_voice_17_0` (config `vi`) | ~0.5 GB / ~1.5 GB | ~5 h validated, ~10 h total | CC0 | Implicit | Gated (HF login + consent). |
| FLEURS (vi) | `google/fleurs` (config `vi_vn`) | ~1 GB | ~10 h train + dev/test | CC-BY-4.0 | Implicit | **Parallel with En/Zh/Ko configs** — same sentences across languages. |
| InfoRe ASR (alt use of TTS data) | `doof-ferb/infore1_25hours` | ~3 GB | 25 h | CC-BY-4.0 | Implicit | Single-speaker, also usable as Vi ASR. |

## 2. Mandarin & Korean ASR (cross-lingual transfer)

| Name | HF ID | Size | Hours | License | Tone annot? | Notes |
|---|---|---|---|---|---|---|
| AISHELL-1 | `AISHELL/AISHELL-1` | ~15 GB | 170 h | Apache-2.0 | Hanzi (no explicit pinyin) | 400 speakers Mandarin. The standard Zh ASR benchmark. |
| AISHELL-3 | `AISHELL/AISHELL-3` | ~19 GB | 85 h | Apache-2.0 | **Yes — pinyin + tone marks** | 218 speakers, designed for TTS but ideal for tone-aware ASR/quant eval. |
| WenetSpeech | `wenet-e2e/wenetspeech` | ~1 TB | 12,483 h | CC-BY-4.0 (NC) | Hanzi only | Multi-domain YouTube + podcast. Non-commercial. |
| FLEURS (cmn) | `google/fleurs` (config `cmn_hans_cn`) | ~1 GB | ~10 h | CC-BY-4.0 | Hanzi only | Parallel with FLEURS vi/ko/en. |
| Common Voice 17 (zh-CN) | `mozilla-foundation/common_voice_17_0` (config `zh-CN`) | ~10 GB | ~80 h validated | CC0 | Hanzi only | Gated. |
| Zeroth-Korean | `kresnik/zeroth_korean` (or `Bingsu/zeroth-korean`) | 2.68 GB / 5.52 GB | 51.6 h train + 1.2 h test | CC-BY-4.0 | n/a (Ko not tonal) | 105 speakers, OpenSLR-40 origin. |
| KsponSpeech (eval only on HF) | `jubang0219/KsponSpeech` | <1 GB | 2.6 h clean + 3.8 h other (eval) | Research-only | n/a | Full 969 h corpus only via AI Hub (Korean gov sign-up). |
| FLEURS (ko) | `google/fleurs` (config `ko_kr`) | ~1 GB | ~10 h | CC-BY-4.0 | n/a | Parallel with FLEURS vi/cmn/en. |
| Common Voice 17 (ko) | `mozilla-foundation/common_voice_17_0` (config `ko`) | ~0.3 GB | ~1.5 h validated | CC0 | n/a | Very small — augment with Zeroth. |

## 3. Parallel Text (MT training/eval)

| Name | HF ID | Pairs | Vi-En | Vi-Zh | Vi-Ko | License | Notes |
|---|---|---|---|---|---|---|---|
| FLORES-200 (devtest) | `facebook/flores` / `openlanguagedata/flores_plus` | 3001 sents × all pairs | yes | yes | yes | CC-BY-SA-4.0 | <100 MB. **Use this as primary MT eval.** Codes: `vie_Latn`, `eng_Latn`, `zho_Hans`, `kor_Hang`. |
| OPUS-100 | `Helsinki-NLP/opus-100` (configs `en-vi`, `en-zh`, `en-ko`) | ~1 M per direction | yes | via En pivot | via En pivot | mixed CC-BY / CC0 | English-centric; no direct Vi-Zh or Vi-Ko. |
| NLLB bitext (web-mined) | `allenai/nllb` | tens of M per pair | yes (vie_Latn-eng_Latn) | yes | yes | per-source mixed; ODC-By for aggregation | Very large gzipped TSVs; needs LASER score filtering. |
| NLLB Seed | `SEACrowd/nllb_seed` (mirror of facebook/flores_plus seed) | ~6K sents × 39 langs | yes | n/a | n/a | CC-BY-SA-4.0 | Tiny, high-quality Wikipedia. |
| Tatoeba MT | `Helsinki-NLP/tatoeba_mt` | varies | yes | yes (small) | yes (small) | CC-BY-2.0 | User-contributed; small for Vi but covers Vi-Zh / Vi-Ko directly. |
| PhoMT | `SEACrowd/phomt` (loader; requires manual DL from VinAIResearch/PhoMT GitHub) | 3.02 M | **yes (3 M)** | no | no | research-only | Largest clean Vi-En text corpus. |
| MTet | n/a (vietai.org) | ~4 M | yes | no | no | research-only | Multi-domain Vi-En, supplements PhoMT. |
| TED2020 | OPUS / mirrored under `Helsinki-NLP/opus-*` | 3.1K Vi-En subtitle pairs | yes | yes (via TED) | yes (via TED) | CC-BY-NC-ND | Small but high quality. |
| OpenSubtitles v2018 | `Helsinki-NLP/open_subtitles` | ~3.9K Vi-En pairs (and Vi-Zh, Vi-Ko) | yes | yes | yes | research/educational | Movie subtitles, noisy alignment. |

## 4. Speech Translation (direct S2ST / S2TT)

| Name | HF ID | Pair coverage | Hours | License | Vi? | Notes |
|---|---|---|---|---|---|---|
| CoVoST-2 | `facebook/covost2` | 21→En, En→15 | 2,900 h total | CC0 | **No Vi** | Use as Zh-En and Ko-pair-not-present baseline. AudioLLMs/covost2_zh_en_test for Zh-En test. |
| CVSS | `google/cvss` (`-c` canonical voice, `-t` transferred voice) | 21→En S2S | derived from CoVoST-2 | CC-BY-4.0 | **No Vi** | Best S2S corpus, but Vi absent. Useful for Zh→En S2S. |
| FLEURS | `google/fleurs` | all 4 langs parallel | ~10 h each | CC-BY-4.0 | yes | The only S2ST resource covering Vi↔{En,Zh,Ko} directly (n-way at sentence level). |
| MuST-C | n/a (ict.fbk.eu sign-up) | En→Vi 441 h + others | 441 h En-Vi | CC-BY-NC-ND-4.0 | yes (En→Vi) | Has known alignment errors in Vi (see PhoST paper) — prefer PhoST. |
| PhoST | GitHub `VinAIResearch/PhoST` (not on HF) | En-Vi ST | 508 h | research-only | yes | 331 K (audio, En text, Vi text) triplets. Best En-Vi ST. |
| GigaST | (FB AI; off-HF) | En→{Zh, others} | ~10 K h | research-only | no | Pseudo-labeled, very large; secondary use. |

## 5. Vietnamese TTS

| Name | HF ID | Size | Hours | Speakers | License | Notes |
|---|---|---|---|---|---|---|
| InfoRe 25h | `doof-ferb/infore1_25hours` | ~3 GB | 25 h | 1 | CC-BY-4.0 | Cleanest small Vi single-speaker TTS. |
| VietBibleVox | `ntt123/VietBibleVox` | ~5 GB | ~80 h (29,185 verses) | 1 (male) | CC-BY-SA-4.0 | Bible readings, narrow domain. |
| viet-tts-dataset (NTT) | `ntt123/viet-tts-dataset` | <1 GB | ~5 h | 1 | CC-BY-4.0 | LJSpeech-style small set. |
| viVoice (1000 h) | `hynt/F5-TTS-Vietnamese-ViVoice` (model card carries dataset link) | ~60 GB | 1,017 h | 186 YouTube channels | YouTube ToS uncertain — research-only safe | Cleaned, music-removed YT speech. |
| ZipVoice-Vietnamese-2500h | `hynt/ZipVoice-Vietnamese-2500h` | ~150 GB | 2,500 h | many | research-only | Very large but YouTube-derived. |
| Bud500 (re-use) | `linhtran92/viet_bud500` | (see ASR) | 500 h | many | Apache-2.0 | Also usable for multi-speaker Vi TTS. |
| VLSP-2020 / VinBigData TTS | n/a | varies | varies | varies | research-only | Mostly ASR-curated; check VLSP archive. |

## 6. Industrial / Noisy Speech (likely scarce)

**Finding: no public Vietnamese industrial/factory speech corpus exists on HuggingFace, OpenSLR, or in the VLSP archive.** This is a known gap. Recommended strategy: synthesize via noise augmentation.

| Name | HF ID | Size | License | Notes |
|---|---|---|---|---|
| MUSAN | `FluidInference/musan` (mirror; original openslr.org/17/) | ~11 GB | CC-BY-4.0 | Music + Speech + Noise (technical/non-technical). Standard for SNR-based augmentation. |
| DEMAND | n/a on HF (download Zenodo: 10.5281/zenodo.1227121) | ~5 GB | CC-BY-SA-3.0 | 18 environments × 16-ch recordings, **includes "PSTATION", "OOFFICE", "TMETRO", "DKITCHEN"** — closest proxy to factory/industrial ambient. |
| FSDnoisy18k | `sps44/fsdnoisy18k` | ~12 GB | CC-BY-4.0 | 42.5 h, 20 sound classes incl. "Engine", "Slam", "Tearing" — useful as industrial-like augment noise. |
| CHiME-6 | `argmaxinc/chime-6` | ~20 GB | CC-BY-NC-4.0 | Dinner-party conversational; not factory but real far-field noise. |
| Noisy LibriSpeech-MUSAN | `zhaoyang9425/NoisyLibriSpeechDataset-MUSAN` | varies | derivative | Ready-made noisy ASR train set (En) — template for our Vi noisy-augmentation pipeline. |
| ambient_noise_audio | `Willy030125/ambient_noise_audio` | ~1 GB | unspecified | Misc ambient sounds, vet license before use. |

**Recommended Vi noisy-eval pipeline:** take VIVOS / FLEURS-vi clean → mix at SNR ∈ {0, 5, 10, 15} dB with DEMAND `PSTATION` + `DKITCHEN` channels + MUSAN noise subset. Document the mix recipe so reviewers can reproduce.

## 7. Download scripts

```python
# pip install -U datasets huggingface_hub
# huggingface-cli login   # needed for Common Voice (gated) and CC0 datasets requiring consent

from datasets import load_dataset
from huggingface_hub import snapshot_download

# ===== Tier 1 =====

# Vietnamese ASR — small canonical
ds_vivos = load_dataset("AILAB-VNUHCM/vivos")                       # ~15 h, 1.5 GB
# FLEURS — n-way parallel benchmark across all 4 languages
ds_fleurs_vi = load_dataset("google/fleurs", "vi_vn")               # ~10 h
ds_fleurs_zh = load_dataset("google/fleurs", "cmn_hans_cn")         # ~10 h
ds_fleurs_ko = load_dataset("google/fleurs", "ko_kr")               # ~10 h
ds_fleurs_en = load_dataset("google/fleurs", "en_us")               # ~10 h

# Korean ASR
ds_zeroth = load_dataset("kresnik/zeroth_korean")                   # 51.6 h, 5.5 GB

# FLORES MT eval (text only) — primary Vi↔{En,Zh,Ko} benchmark
ds_flores = load_dataset("openlanguagedata/flores_plus")            # 3001 sents × 200 langs, <100 MB

# Vietnamese TTS
ds_infore = load_dataset("doof-ferb/infore1_25hours")               # 25 h, 3 GB

# Mandarin TTS / tone-rich ASR
ds_aishell3 = load_dataset("AISHELL/AISHELL-3")                     # 85 h, 19 GB — pinyin+tone labels

# ===== Tier 2 =====

ds_aishell1 = load_dataset("AISHELL/AISHELL-1")                     # 170 h
ds_cv_vi = load_dataset("mozilla-foundation/common_voice_17_0", "vi", trust_remote_code=True)
ds_cv_zh = load_dataset("mozilla-foundation/common_voice_17_0", "zh-CN", trust_remote_code=True)
ds_cv_ko = load_dataset("mozilla-foundation/common_voice_17_0", "ko", trust_remote_code=True)
ds_vlsp = load_dataset("doof-ferb/vlsp2020_vinai_100h")             # 100 h Vi
ds_opus100_envi = load_dataset("Helsinki-NLP/opus-100", "en-vi")
ds_opus100_enzh = load_dataset("Helsinki-NLP/opus-100", "en-zh")
ds_opus100_enko = load_dataset("Helsinki-NLP/opus-100", "en-ko")
ds_covost_zhen = load_dataset("facebook/covost2", "zh-CN_en")

# PhoST (not on HF) — clone separately:
#   git clone https://github.com/VinAIResearch/PhoST && follow PhoST/README

# ===== Tier 3 (large) =====

ds_bud500 = load_dataset("linhtran92/viet_bud500", streaming=True)  # 500 h, ~100 GB — use streaming
snapshot_download(repo_id="wenet-e2e/wenetspeech", repo_type="dataset",
                  allow_patterns=["meta/*", "audio/train_s/*"])      # selective download

# ===== Noise for industrial-robustness augmentation =====

snapshot_download(repo_id="FluidInference/musan", repo_type="dataset")
ds_fsdnoisy = load_dataset("sps44/fsdnoisy18k")
# DEMAND: not on HF, download from https://zenodo.org/records/1227121
```

## 8. Kaggle upload strategy

**Tier 1 total size estimate: ~45 GB compressed (~60 GB extracted)**
- VIVOS: 1.5 GB
- FLEURS (×4 configs Vi/Zh/Ko/En): ~4 GB
- Zeroth-Korean: 2.7 GB
- FLORES-plus: <0.1 GB
- InfoRe 25h: 3 GB
- AISHELL-3: 19 GB
- Plus AISHELL-1 if promoted: +15 GB → 60 GB

**Kaggle dataset limit: 20 GB free, 100 GB with verified phone.** Recommendation:

1. **Split into 3 Kaggle datasets**:
   - `voiceai-tier1-vi` (VIVOS + FLEURS-vi + InfoRe + FLORES) ≈ 8 GB
   - `voiceai-tier1-zh` (AISHELL-3 + FLEURS-cmn) ≈ 20 GB
   - `voiceai-tier1-ko-en` (Zeroth-Korean + FLEURS-ko + FLEURS-en) ≈ 8 GB
2. Use Kaggle's "Add Data" feature to mount all three in any notebook — no need to merge.
3. AISHELL-3 (Apache-2.0), VIVOS (CC-BY-SA), FLEURS (CC-BY-4.0), InfoRe (CC-BY-4.0), Zeroth (CC-BY-4.0), FLORES (CC-BY-SA-4.0) — **all redistributable** with attribution. Keep a `LICENSES.md` in each Kaggle dataset bundling the source attributions.

## 9. Licensing flags to watch — DO NOT redistribute

These can be used in experiments but **must not** be re-uploaded as public Kaggle datasets:

- **Common Voice 17** — CC0 but gated by Mozilla acceptance flow; re-host violates the acceptance terms.
- **VLSP 2020 / 2021** (both `doof-ferb/vlsp2020_vinai_100h` and the original) — research-only, VLSP shared-task terms.
- **KsponSpeech** — Korean gov AI Hub research-only; only eval slice (`jubang0219/KsponSpeech`) is publicly downloadable.
- **WenetSpeech** — CC-BY-4.0 with explicit "non-commercial" rider.
- **MuST-C** — CC-BY-NC-ND-4.0 (no redistribution, no derivatives).
- **PhoST, PhoMT, MTet** — research-only VinAI / VietAI terms; cite + link, don't re-host.
- **CoVoST 2** — CC0 itself, BUT derived from Common Voice; carries CV's gating implicitly.
- **viVoice / ZipVoice-Vietnamese-2500h** — derived from YouTube; copyright status of source clips unclear → safest to treat as research-only.
- **CHiME-6** — CC-BY-NC-4.0.

For the Kaggle bundles in Section 8, **only include**: VIVOS, FLEURS, FLORES, InfoRe, AISHELL-1, AISHELL-3, Zeroth-Korean, MUSAN, FSDnoisy18k.

---

## Quick gap summary for this project

1. **No Vi industrial-noise corpus exists.** Augmentation pipeline (DEMAND PSTATION/DKITCHEN + MUSAN @ 0–15 dB SNR over VIVOS / FLEURS-vi) is the realistic substitute — make this an explicit method-section caveat in the EACL submission.
2. **No Vi↔Ko or Vi↔Zh direct S2ST corpus exists.** FLEURS is the *only* resource with n-way parallel speech across all 4 of our languages. Everything else routes through English (cascade En pivot or two-step S2TT→TTS).
3. **For tone-aware quantization eval**: the cleanest signal is on (Vi: VIVOS + FLEURS-vi, Vi tones via diacritics) and (Zh: AISHELL-3, Zh tones via pinyin tone digits). Korean has no tone axis — treat Ko as the "non-tonal control" condition.
4. **PhoST is the only credible large Vi-En ST corpus** — Mu-ST-C Vi pair has documented alignment errors (4–16% in val/test). Cite PhoST (Interspeech 2022).
