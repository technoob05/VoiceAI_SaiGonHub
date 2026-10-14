# Noise robustness — synthetic (`voiceai_noise-t4-v1.json`) and real DEMAND (`voiceai_noise-t4-v3-demand.json`)

Kaggle T4, Internet off. 60 VIVOS utterances × 4 synthetic noise types × 4 SNR
levels × 3 conditions (clean / noisy / noisy+GTCRN), plus a VAD false-trigger
grid. Script: [`exp/exp_noise_robustness.py`](../exp/exp_noise_robustness.py).

**The noise is SYNTHETIC, not DEMAND/MUSAN.** Neither corpus is on disk and the
kernel has no Internet, so the noise is generated deterministically from a seed:
pink (stationary broadband masking), hum (50 Hz + harmonics, concentrated below
700 Hz where F0/F1 and Vietnamese tone contours live), clank (sparse inharmonic
transients), babble (8 talkers with syllabic modulation). Numbers are valid for
comparison **inside this run only** and are not comparable to published DEMAND or
MUSAN results. The paper must rerun the grid on the real corpora.

Clean baseline WER **0.1117** [95% CI 0.0795, 0.1458].

## The grid

| Noise | SNR | noisy WER | +GTCRN WER | ΔWER | 95% CI on Δ | SI-SDR gain |
|---|---:|---:|---:|---:|---|---:|
| pink | 0 | 0.2880 | 0.3442 | +0.0562 | [−0.032, +0.142] | +8.21 dB |
| pink | 5 | 0.1619 | 0.1919 | +0.0300 | [−0.039, +0.096] | +7.08 dB |
| pink | 10 | 0.1260 | 0.1581 | +0.0322 | [−0.020, +0.100] | +5.41 dB |
| pink | 15 | 0.1269 | 0.1474 | +0.0205 | [−0.031, +0.079] | +3.59 dB |
| hum | 0 | 0.1608 | 0.1491 | −0.0117 | [−0.090, +0.043] | +11.81 dB |
| hum | 5 | 0.1080 | 0.1198 | +0.0118 | [−0.014, +0.043] | +9.96 dB |
| hum | 10 | 0.1106 | 0.1365 | +0.0259 | [−0.027, +0.095] | +7.88 dB |
| hum | 15 | 0.1324 | 0.1385 | +0.0061 | [−0.054, +0.069] | +5.45 dB |
| clank | 0 | 0.1155 | 0.1467 | +0.0313 | [−0.007, +0.080] | +21.88 dB |
| clank | 5 | 0.1171 | 0.1407 | +0.0237 | [−0.024, +0.087] | +17.66 dB |
| clank | 10 | 0.1451 | 0.1204 | −0.0248 | [−0.085, +0.018] | +13.18 dB |
| clank | 15 | 0.1394 | 0.1481 | +0.0087 | [−0.068, +0.073] | +9.13 dB |
| **babble** | **0** | 0.2761 | **0.5596** | **+0.2835** | **[+0.172, +0.448]** | +7.44 dB |
| **babble** | **5** | 0.2070 | 0.2779 | **+0.0709** | **[+0.015, +0.134]** | +6.40 dB |
| babble | 10 | 0.2027 | 0.1601 | −0.0426 | [−0.120, +0.018] | +5.15 dB |
| babble | 15 | 0.1691 | 0.1329 | −0.0362 | [−0.117, +0.022] | +3.76 dB |

Positive ΔWER = the denoiser made the transcript worse.

## What this actually says

**Read the CIs, not the sign count.** The raw tally is "hurts in 12 of 16", but
only **two cells have a CI that excludes zero** — both babble, at 0 and 5 dB. In
the other 14 the denoiser's effect on WER is statistically indistinguishable from
nothing. A smoke run at n=3 had suggested a blanket "GTCRN always hurts"; that was
overstated and this run corrects it.

What survives:

1. **In multi-talker babble at low SNR, GTCRN is catastrophic.** At 0 dB WER
   doubles, 0.276 → 0.560, CI [+0.172, +0.448]. That is the factory floor — several
   people talking near the mic — which makes it the worst possible place to fail.
2. **SI-SDR never predicts the WER outcome.** It improved in all 16 cells, by
   +3.6 to +21.9 dB. Clank at 0 dB got the largest gain in the whole grid (+21.9 dB)
   and still lost WER. Any future enhancement work must be scored on WER; a
   perceptual metric will happily approve a change that hurts recognition.
3. **It costs 363 ms p50 / 539 ms p95** — roughly a quarter of the 1.5 s end-to-end
   budget, bought with no measurable recognition benefit.

## The one place it earns its keep — and we cannot use it there yet

The VAD grid is the opposite story. Silero false-triggers on impulsive metallic
noise, and GTCRN removes that completely:

| Noise | false trigger, raw | after GTCRN |
|---|---:|---:|
| clank @ 0 dB | 8.3 % | **0 %** |
| clank @ 5 dB | 12.5 % | **0 %** |
| clank @ 10 dB | 12.5 % | **0 %** |
| clank @ 15 dB | 8.3 % | **0 %** |
| pink / hum / babble, all SNRs | 0 % | 0 % |

Every false trigger costs a whole wasted ASR pass, so this is worth real money on
a device. **The denoiser helps the gate, not the recogniser.**

Capturing that requires a *streaming* denoiser on the capture path;
`edge_frontend`'s GTCRN is whole-signal, so the live loop cannot use it that way
today. sherpa-onnx ships `OnlineSpeechDenoiser` — that is the follow-up.

## Real DEMAND noise — measured, and it is WORSE than synthetic

`voiceai_noise-t4-v3-demand.json`, same T4 protocol, n=60 per cell, run on the
`kit567` account after the primary account's weekly GPU quota ran out.
DEMAND (Thiemann, Ito & Vincent 2013, doi:10.5281/zenodo.1227121, **CC BY 4.0**),
official 16 kHz release, `ch01`, sha256 manifest shipped with the audio.

| Source | Noise | SNR | noisy WER | +GTCRN | ΔWER | 95% CI |
|---|---|---:|---:|---:|---:|---|
| demand | **PSTATION** | 0 | 0.2080 | 0.3843 | **+0.1763** | **[+0.079, +0.298]** |
| demand | **PSTATION** | 5 | 0.1622 | 0.2442 | **+0.0821** | **[+0.008, +0.173]** |
| demand | **PSTATION** | 10 | 0.1037 | 0.1725 | **+0.0688** | **[+0.018, +0.126]** |
| demand | PSTATION | 15 | 0.1283 | 0.1677 | +0.0394 | [−0.005, +0.094] |
| demand | **PCAFETER** | 0 | 0.2930 | 0.4737 | **+0.1807** | **[+0.045, +0.334]** |
| demand | **PCAFETER** | 5 | 0.1496 | 0.2678 | **+0.1182** | **[+0.052, +0.202]** |
| demand | PCAFETER | 10 | 0.1947 | 0.1801 | −0.0146 | [−0.103, +0.049] |
| demand | PCAFETER | 15 | 0.1364 | 0.1308 | −0.0056 | [−0.045, +0.030] |
| demand | DKITCHEN | 0 | 0.1268 | 0.1571 | +0.0303 | [−0.048, +0.123] |
| demand | DKITCHEN | 5–15 | ~0.13 | ~0.12 | −0.010 to −0.022 | all straddle 0 |

Bold rows have a CI excluding zero.

**A 2-utterance wiring smoke had pointed the other way — it was wrong.** At n=60 the
real corpus does not rescue the denoiser; it convicts it harder:

| | significant harmful cells |
|---|---|
| synthetic (16 cells) | 2 — both babble, at 0 and 5 dB |
| **DEMAND (12 cells)** | **5 — PCAFETER 0/5 dB and PSTATION 0/5/10 dB** |

`PSTATION` is a **production hall**: the literal environment this competition names.
GTCRN degrades ASR there significantly at 0, 5 **and** 10 dB — a wider SNR range
than any synthetic noise type showed. Only DKITCHEN is benign, and even there every
CI straddles zero.

## The VAD benefit was a synthetic artifact

This is the finding that costs us something, because the streaming denoiser was
built specifically to capture it.

| Source | Noise | SNR | false trigger, raw | after GTCRN |
|---|---|---:|---:|---:|
| synthetic | clank | 0–15 | 8.3–12.5 % | **0 %** |
| demand | **PSTATION** | 0 | **0 %** | **29.2 %** |
| demand | **PSTATION** | 5 | **0 %** | **16.7 %** |
| demand | DKITCHEN | 0–15 | 8.3–12.5 % | 0–8.3 % |
| demand | PCAFETER | 0–15 | 29–79 % | 92–100 % |

On synthetic impulsive noise GTCRN removed every false trigger. On the **real
production hall it manufactures them**: Silero never fires on raw PSTATION, and
fires on 29 % of denoised segments at 0 dB. DKITCHEN is the only case that improves.

PCAFETER is excluded from this judgement by design — it *is* people talking, so a
trigger there is the VAD working, not failing. That caveat ships in the JSON.

So the justification for `denoise_vad` does not survive contact with recorded noise.
The feature stays in the tree, off by default, with this result recorded next to it;
it should not be enabled without evidence from the actual deployment environment.

MUSAN remains skipped: no usable subset exists (openslr ships one 11 GB non-seekable
archive; every Hub mirror checked was a stub, a pre-mixed derivative, or unlicensed).

## What changed as a result

`LiveConfig.denoise` defaults to **False** (`VOICEAI_DENOISE=1` re-enables), with
the numbers in the code and a regression test pinning it. The real-corpus run only
strengthens that: GTCRN in front of ASR is significantly harmful in five of twelve
DEMAND cells, including the production hall at three separate SNRs.

`LiveConfig.denoise_vad` — the streaming path built to capture the VAD win — also
stays **off**, and now for a different reason than caution. The win itself did not
replicate: on real PSTATION the denoiser *creates* false triggers where raw audio
produced none. The code and its tests remain (the two-aligned-stream machinery is
correct and the ASR-must-see-raw-audio invariant is worth keeping pinned), but there
is currently **no measured benefit to turning it on**.

The one thing the earlier synthetic-only run did get right is the ordering fix: the
original code segmented with the VAD on **raw** audio and fed **denoised** audio to
ASR, which is backwards on both counts regardless of which corpus you measure.

### Honest summary for the pitch

There is no evidence that speech enhancement helps this pipeline, on synthetic or on
recorded industrial noise, and clear evidence it hurts in the loudest conditions.
The remaining mitigation for the "industrial noise destroys ASR" risk is
noise-augmented fine-tuning of the ASR itself — which has not been attempted.
