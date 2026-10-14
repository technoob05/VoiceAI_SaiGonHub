# Kaggle T4 benchmark — offline Vi→En cascade

Development benchmark for the full local pipeline
**ASR → Vietnamese corrector → MT → TTS**, run on a Kaggle GPU kernel with
**Internet disabled** and every weight, wheel, and helper module supplied as a
private dataset input. T4 is *not* the competition device — it stands in for a
mid-range GPU while the Snapdragon/Hexagon path is being built.

## Setup

| | |
|---|---|
| Kernel | `daosyduyminh/voiceai-t4-inference-full` (private, script, Internet off) |
| GPU | Tesla T4 (kernel sees 2×; CTranslate2 pinned to device 0) |
| Runtime | Python 3.12.13, torch 2.10.0+cu128, CTranslate2 4.7.2 |
| ASR | PhoWhisper-small, CT2 `int8_float16`, CUDA |
| Corrector | vietnamese-correction-v2, CT2 `int8_float16`, CUDA |
| MT | NLLB-200-distilled-600M, CT2 `int8_float16`, CUDA |
| TTS | Piper `en_US-lessac-medium` via sherpa-onnx, **CPU** |
| Data | VIVOS test, 300 utterances + 5 warm-ups, 969.3 s of audio (mean 3.23 s, range 1.38–6.44 s) |
| Condition | `full-t4-int8-v2` (config dataset v3, srclib v2) |

## Warm per-utterance latency (n=300)

| Stage | mean | p50 | p95 | min | max |
|---|---:|---:|---:|---:|---:|
| ASR | 167.0 ms | 167.3 ms | 204.1 ms | 101.5 ms | 225.2 ms |
| Corrector | 47.4 ms | 47.2 ms | 67.2 ms | 23.1 ms | 80.0 ms |
| MT | 59.4 ms | 56.4 ms | 87.1 ms | 26.0 ms | 132.7 ms |
| TTS (CPU) | 138.1 ms | 131.7 ms | 218.3 ms | 46.8 ms | 319.3 ms |
| **End-to-end** | **411.9 ms** | **400.7 ms** | **563.3 ms** | **208.5 ms** | **751.4 ms** |
| **RTF** | **0.132** | **0.129** | **0.175** | 0.072 | 0.218 |

- **RTF < 1.0 on 300/300 utterances**; worst case 0.218.
- **End-to-end < 2 s on 300/300 utterances**; worst case 751 ms.
- Peak process GPU memory **2537 MiB (2.48 GiB)** — well inside the 10 GiB budget.
- Model load (excluded from the latency numbers above): ASR 2.76 s, corrector
  2.72 s, MT 5.82 s, TTS 2.24 s → **13.5 s cold start**, paid once at prewarm.

Latency is measured warm, after 5 warm-up utterances, and `total_ms` is
conservative: it runs to fully synthesized audio rather than to first audio
chunk. Streaming first-audio latency will be lower and is measured separately.

## Corrector behaviour

| Outcome | n |
|---|---:|
| `unchanged` (model returned the ASR text, casing aside) | 289 |
| `accepted` (edit passed every safety gate) | 6 |
| `deletion_guard` (rejected, fell back to raw ASR) | 5 |

The `deletion_guard` was added *because of* the v1 run on this same 300-sample
set: the corrector was quietly deleting content words while staying inside the
length and edit-ratio budgets — `lính cứu hỏa` → `lính cứu`,
`ủy ban châu âu` → `ban châu Âu`, `y tế bình hưng hòa` → `y tế bình hưng`.
Those five utterances now fall back to the raw ASR text instead of reaching MT
with an entity missing. See `voiceai_full-t4-int8-v1.json` for the pre-fix run.

## Honest limits

- T4 ≠ Snapdragon. These numbers say the cascade and its quantization are sound;
  they say nothing about NPU latency, which still has to be measured on device.
- TTS ran on the kernel's CPU, so its share of the budget is the least
  transferable number here.
- Only Vi→En was measured. Zh and Ko targets are untested at this scale.
- Quality (WER/TER/BLEU) is not in this run — this is a latency and memory
  benchmark only.

## Files

| File | Contents |
|---|---|
| `voiceai_full-t4-int8-v2.json` | Headline run: config, environment, model SHA-256, per-utterance rows, summary |
| `voiceai_full-t4-int8-v2_rows.jsonl` | Same rows, one JSON per line |
| `voiceai_full-t4-int8-v1.json` | Pre-deletion-guard run on the same 300 utterances |
| `voiceai_smoke-t4-int8-v1.json` | 2-sample smoke that unblocked the full run |
| `2026-08-17-smoke-v*/` | Earlier failed smoke attempts, kept for the debugging trail |

## Reproducing

```
kaggle kernels push -p kaggle_uploads/voiceai-t4-kernel-full
kaggle kernels output daosyduyminh/voiceai-t4-inference-full -p Results/kaggle_t4
```

Kaggle mounts dataset inputs at `/kaggle/input/datasets/<owner>/<slug>/` and
auto-extracts uploaded ZIPs into a same-named subfolder, so the kernel resolves
both layouts itself (`dataset_root` / `payload_dir` in
`exp/exp_t4_s2st_inference.py`). Dataset versions propagate a few minutes after
`kaggle datasets version` returns — verify the new content is live before
pushing the kernel, or the run silently uses the previous version.

---

# Bidirectional benchmark — `voiceai_bidir-full-v1.json`

Second T4 run (2026-08-20), covering everything the first one did not: the second
ASR model, all measurable directions, terminology control, and the fast first TTS
fragment. Script: [`exp/exp_t4_bidirectional.py`](../../exp/exp_t4_bidirectional.py);
kernel `daosyduyminh/voiceai-t4-bidirectional`, Internet off, Tesla T4.

Peak GPU **2897 MiB** with both ASR models plus MT and the corrector resident.
Model load: ASR-vi 676 ms, ASR-multilingual 1066 ms, corrector 1002 ms, MT 3490 ms,
all four Piper voices 12879 ms (en 3339 / vi 1690 / zh 6429 / ko 1422).

## A. Does the Vietnamese specialist earn its place? Yes, decisively.

Same 120 VIVOS utterances, same decode settings, both models resident.

| Model | WER | CER | latency p50 | p95 |
|---|---:|---:|---:|---:|
| PhoWhisper-small INT8 | **0.1259** | **0.0718** | 177 ms | 229 ms |
| whisper-small multilingual INT8 | 0.2920 | 0.1761 | 172 ms | 212 ms |

Paired bootstrap over the per-utterance vectors (5000 resamples, seed 20260820):
WER delta **+0.1667, 95% CI [+0.1159, +0.2148]**; CER delta **+0.1044, 95% CI
[+0.0665, +0.1424]**. Paired permutation test p ≈ 1e-4. Per utterance PhoWhisper
wins 91/120, ties 16, loses 13.

The multilingual model is **2.3× worse on Vietnamese at identical latency**. That
settles the architecture question: the device carries two ASR models, not one.
Dropping PhoWhisper to save 251 MB would cost 17 WER points.

Caveat on the absolute number: 12.59 % WER is far above the 4.67 % that
PhoWhisper-**large** reaches on VIVOS. Ours is small + INT8, scored with a
normalization that deliberately preserves Vietnamese diacritics (see
`normalize_for_scoring`), so a tone error counts as an error. This is a
system-level number, not a leaderboard entry.

## B. Reverse-direction ASR (multilingual model)

| Corpus | n | WER | CER | primary | latency p50 |
|---|---:|---:|---:|---:|---:|
| FLEURS en_us test | 60 | **0.0663** | 0.0320 | WER | 229 ms |
| FLEURS ko_kr test | 60 | 0.2441 | **0.0732** | CER | 279 ms |
| Mandarin | — | — | — | — | **not measurable** |

Korean WER is read as CER on purpose. Korean spacing is partly optional, so a
perfect transcript can score a high WER on segmentation alone — verified
synthetically: `"안녕하세요 반갑습니다"` vs `"안녕하세요반갑습니다"` scores WER 1.00 and
CER 0.00. CER 7.3 % is usable Korean ASR; WER 24.4 % on the same rows is an
artifact of the metric, not of the model.

**Mandarin remains unmeasured.** No Mandarin audio exists in any mounted dataset,
and the repo's `smoke_R002_zh.wav` is digital silence. The JSON carries an
explicit reason string rather than a null. This is the largest open hole in the
evidence.

## C. Five of six directions, end to end

| Direction | n | total p50 | total p95 | RTF p50 | RTF p95 | ASR | MT | TTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vi→en | 60 | 451 ms | 639 ms | 0.138 | 0.185 | 191 | 60 | 149 |
| vi→zh | 60 | 1165 ms | 1863 ms | 0.361 | 0.503 | 192 | 56 | **877** |
| vi→ko | 60 | 507 ms | 751 ms | 0.163 | 0.211 | 191 | 55 | 210 |
| en→vi | 60 | 876 ms | 1392 ms | 0.095 | 0.141 | 240 | 159 | 470 |
| ko→vi | 60 | 884 ms | 1476 ms | 0.073 | 0.107 | 294 | 153 | 430 |
| zh→vi | — | — | — | — | — | — | — | — |

RTF < 1.0 on **300/300** measured utterances. Under 2 s: 100 % for vi→en, vi→ko
and en→vi; **96.7 %** for vi→zh and 98.3 % for ko→vi — i.e. 2 of 60 and 1 of 60
utterances crossed the 2-second bar.

**Mandarin TTS is the bottleneck in the whole system.** MeloTTS zh costs 877 ms
p50 against 149–470 ms for the other three, and it is also the slowest voice to
load (6429 ms). It runs at 44.1 kHz where the others are 22.05 kHz. Every vi→zh
number above is dominated by it, and it is the first thing to replace or quantize.

**Correction (2026-08-20): the 824 ms Mandarin figure was measured with the
per-language thread budget bypassed.** The benchmark constructs
`PiperTTS(num_threads=2, ...)`, and `PiperTTS.__init__` sets
`_thread_override = num_threads is not None`, which makes `_threads_for` return
early. Verified: with the explicit argument, all four voices ran at 2 threads —
the worst setting in the dev-box sweep. The kernel copy now passes the 2 through
`VOICEAI_TTS_THREADS` so English, Vietnamese and Korean stay at 2 as a control and
only Mandarin picks up its own budget. **The effect of the thread change is
therefore still unmeasured**; a re-run that had kept the constructor argument would
have reproduced ~824 ms and been misread as "the fix did nothing".

Voices were pre-warmed and the LRU raised to 4 slots for this run so a cold voice
load could not be charged to `tts_ms`. The shipped device uses a 2-slot LRU
because exactly two languages are live per session; a **pair change** therefore
still costs a voice load, up to 6.4 s for Mandarin.

## D. Terminology control — real, and smaller than it first looked

108 probes (61 glossary terms + 12 full industrial sentences) × 3 targets.

| Probe class | probes | terms | recovery OFF | recovery ON |
|---|---:|---:|---:|---:|
| **Translated sentences** (the honest test) | 36 | 102 | 0.167 | **0.441** |
| Exact whole-utterance hits | 72 | 72 | 0.250 | 1.000 |
| All probes | 108 | 174 | 0.201 | 0.672 |

The exact-hit row is a lookup table, not a translation: those never reach the
decoder, so counting them inflates the headline. **Read the sentence row.** On
real sentences, injection lifts term recovery from 16.7 % to 44.1 % — a large
improvement, and nowhere near solved. The first smoke run reported 0.933 because
its probe set was mostly bare terms; that number was an artifact and is not
repeated here.

Recovery counts the glossary's *preferred* surface form appearing in the output,
so a correct synonym scores as a miss. It is a consistency measure, not adequacy —
we have no MT references for these sentences. Representative vi→zh:

| Source | OFF | ON |
|---|---|---|
| Máy nén khí … dừng dây chuyền | 区B的**压缩机**太热了…停止**电路** | 区域B的**空气压缩机**正在过热…停止**生产线** |
| Xe nâng đang lùi | **车辆**正在向后 | **叉车**正在向后 |
| Lô hàng … trước khi xuất kho | 货物需要在**出货**前 | 货物需要在**仓库发送**前 |

## E. Fast first TTS fragment

| Mode | fragments | first audio | total synthesis | playback headroom |
|---|---:|---:|---:|---:|
| fast first fragment | 2 | **234.4 ms** | 475.7 ms | +2729 ms |
| whole sentence | 1 | 485.5 ms | 485.5 ms | n/a |

First audio **halves** (485 → 234 ms, −52 %) for 10 ms of extra total synthesis on
GPU-backed MT with CPU TTS. Headroom is +2.7 s, so the player cannot underrun at
this fragment size. On CPU the same change measured −21 % with +60 % total; the
GPU run is the friendlier case.

## What this run changes

1. **Keep both ASR models.** Settled with a CI and a significance test, not a vibe.
2. **Mandarin TTS is the next optimization target**, not ASR or MT.
3. **The glossary works and is under-covered** — 44 % recovery means most industrial
   nouns are still outside the 61-term table. Coverage is the whole game.
   **Correction (2026-08-20):** this sentence was untestable as written. The probe
   set is generated *from the glossary*, so it is structurally blind to expansion.
   Growing the table to 93 terms changed the sentence-level recovery by **exactly
   nothing** — a controlled CPU A/B on the same model with both tables produced
   36/36 byte-identical rows, because `find_terms` returns the same 34 terms over
   the 12 fixed sentences either way. Pricing coverage needs held-out sentences
   authored independently of the table; `prototype/glossary_probe_sentences.json`
   is that set, and it is wired into the next run. See `v3_vs_v2.md`.
4. **Mandarin is still evidence-free end to end.** Getting Mandarin audio into the
   speech dataset is now the highest-value data task.


---

# Run v2 — Mandarin finally measured (`voiceai_bidir-full-v2.json`)

Same script and config as v1 plus a new input: `daosyduyminh/voiceai-t4-fleurs-zh`,
200 utterances of FLEURS `cmn_hans_cn` test. Mandarin had been the one direction
with **no audio anywhere in our data**; every zh figure before this was a null
with a reason string.

## All six directions now measured

| Direction | n | total p50 | total p95 | RTF p50 | ASR | MT | TTS | under 2 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vi→en | 60 | 443 ms | 619 ms | 0.133 | 183 | 58 | 142 | 100 % |
| vi→zh | 60 | 1108 ms | 1688 ms | 0.340 | 186 | 55 | **824** | 100 % |
| vi→ko | 60 | 471 ms | 696 ms | 0.154 | 188 | 55 | 201 | 100 % |
| en→vi | 60 | 817 ms | 1303 ms | 0.088 | 229 | 156 | 431 | 100 % |
| ko→vi | 60 | 818 ms | 1345 ms | 0.069 | 280 | 147 | 396 | 100 % |
| **zh→vi** | **60** | **806 ms** | **1225 ms** | **0.083** | 277 | 146 | 398 | **100 %** |

RTF < 1.0 and end-to-end < 2 s on **360/360** utterances across all six directions.

## The Mandarin ASR number is mostly a measurement artifact

| Corpus | n | WER | CER |
|---|---:|---:|---:|
| FLEURS en_us | 60 | 0.0663 | 0.0320 |
| FLEURS ko_kr | 60 | 0.2441 | 0.0732 |
| FLEURS cmn_hans_cn | 60 | 0.9982 | 0.2216 |

WER 0.998 for Mandarin is meaningless: FLEURS ships the Mandarin reference
**character-spaced** (`这 并 不 是 告 别`) while Whisper emits unspaced text, so
every whitespace token mismatches. CER is the only metric that means anything —
same reasoning as Korean, and the script now marks both languages `primary=cer`.

CER 0.2216 is still poor next to English 0.0320. Reading the transcripts found
why, and it is **not** that the model mishears:

```
REF  这 并 不 是 告 别 这 是 一 个 篇 章 的 结 束 ...   (Simplified)
HYP  這並不是告別 這是一個偏張的結束 ...              (Traditional)   CER 0.478
```

**20 of 60 rows contain Traditional-only characters**, and those score CER 0.395
against 0.135 for the rest. A third of the corpus is being penalised for an
orthography choice.

This is a production bug, not only a scoring one: our MT stage is configured for
`zho_Hans`, the glossary's Mandarin column is Simplified, and the Mandarin TTS
voice expects Simplified input. A Traditional transcript degrades every stage
downstream. `exp/exp_zh_script_ab.py` measures whether a Simplified-Chinese
`initial_prompt` fixes it.

## Glossary, re-measured with the honest split

| Probe class | probes | terms | recovery OFF | recovery ON |
|---|---:|---:|---:|---:|
| Translated sentences | 36 | 102 | 0.167 | **0.441** |
| Exact whole-utterance hits | 72 | 72 | 0.250 | 1.000 |

## Streaming TTS

| Mode | fragments | first audio | total synthesis | headroom |
|---|---:|---:|---:|---:|
| fast first fragment | 2 | **207.1 ms** | 441.6 ms | +2645 ms |
| whole sentence | 1 | 465.3 ms | 465.3 ms | n/a |

## Mandarin TTS is still the system bottleneck

824 ms p50 against 142–431 ms for the other three voices. Isolated on a dev box
with `exp/bench_tts_mandarin.py`, normalising for the 44.1 kHz output rate:

| Voice | ms per second of output audio |
|---|---:|
| MeloTTS zh_en | **730** |
| en_US-lessac-medium | 114 |
| vi_VN-vais1000-medium | 110 |
| vits-mimic3-ko_KO-kss_low | 149 |

So it is ~6x the cost per unit of audio, not merely 2x from the sample rate.

Two mitigations were tried:

- **A different Mandarin voice.** `vits-zh-hf-fanchen-C` (16 kHz, 132 MB) measured
  531 ms/audio-s versus 698 for MeloTTS in the same session — but it **drops Latin
  characters** (`Ignore OOV 'B'` from sherpa's character lexicon), so `khu vực B`
  loses the B. Disqualifying for an industrial device full of area codes and model
  numbers. Rejected.
- **Thread budget.** A sweep on a 12-core box: 2 threads → 1211 ms/audio-s,
  4 → 880, 6 → 922, 8 → 786. The T4 benchmark ran TTS at `num_threads=2`, the
  worst setting measured. `local_tts_piper._threads_for` now gives Mandarin its
  own budget (env `VOICEAI_TTS_THREADS_ZH`, default `min(8, cpu_count)`) rather
  than raising the default and stealing cores from ASR.

Honest caveat: the dev box was visibly contended during these runs — a later
repeat of the same zh measurement read 1412 ms/audio-s. The *ordering* of the
thread sweep and the 6x ratio between voices are the trustworthy parts; the
absolute values are not.

## Follow-up: the Simplified-Chinese prompt (`../zh_script_ab.json`)

Run on the same T4, same 60 FLEURS `cmn_hans_cn` rows, `exp/exp_zh_script_ab.py`:

| Condition | CER | median CER | Traditional rows | CER on those rows | latency median |
|---|---:|---:|---:|---:|---:|
| no prompt | 0.2216 | 0.1962 | 20/60 | 0.3946 | 267 ms |
| Simplified prompt | **0.1229** | **0.1056** | **0/60** | — | 270 ms |

Priming the decoder with `以下是普通话的句子，请使用简体中文。` removes Traditional
output **completely** and cuts CER by 44.5 % relative, for **+3 ms**. Shipped in
`asr_router` as `zh_initial_prompt` / `VOICEAI_ASR_PROMPT_ZH`.

Worth stating plainly: 0.2216 was never a hearing problem. Two thirds of the gap
to English was an orthography choice that a prompt fixes for free, and the
remaining 0.1229 is the real Mandarin ASR quality of whisper-small INT8.
