# Offline Vietnamese-Centric Speech Translation

## Phase-3 Technical Readiness Report for the OneVoice AI Challenge 2026

**Snapshot date:** 20 August 2026  
**System scope:** Vietnamese ↔ English, Mandarin and Korean; local speech-to-speech cascade; no inference API  
**Evidence status:** development GPU and Windows supervisor measured; Snapdragon/Hexagon deployment unmeasured

## Technical summary

The current prototype demonstrates a complete offline cascade across all six required
directions on a Kaggle Tesla T4 development environment. In the latest complete run,
360/360 utterances achieved real-time factor below 1 and end-to-end latency below two
seconds. Direction-level p50 latency ranged from 454 ms to 1,148 ms, p95 from 644 ms
to 1,736 ms, and peak GPU allocation was 2,897 MiB. ASR, the guarded Vietnamese
corrector and MT ran on CUDA; TTS ran on CPU.

This is a development proof, not a Snapdragon result. No QNN/Hexagon inference,
on-device power, thermal, sustained-load or ARM/Linux execution measurement exists.
The correct readiness claim is therefore: **the model cascade meets the challenge
latency bar on a constrained development GPU with substantial memory headroom, while
target-device validation remains the principal deployment gate.**

The system now fails safe on two negative findings. GTCRN enhancement is disabled by
default because it significantly worsened ASR in five of twelve real DEMAND cells.
The experimental audio and fuzzy-text caches are also disabled in the submission
runtime because the legacy matching paths are not proven safe for tonal minimal pairs,
negation, numerals or protected entities.

## All six directions meet the development-GPU latency bar

The latest benchmark used 60 utterances per direction, fixed seeds, warm models,
Internet disabled and versioned private inputs. All directions remained below two
seconds. Mandarin synthesis dominates Vietnamese-to-Mandarin latency; raising its
thread count from two to four did not produce an effect distinguishable from
run-to-run drift.

| Direction | n | p50 total | p95 total | p50 RTF | Under 2 s |
|---|---:|---:|---:|---:|---:|
| Vi → En | 60 | 453.6 ms | 644.4 ms | 0.138 | 100% |
| Vi → Zh | 60 | 1,148.0 ms | 1,736.4 ms | 0.360 | 100% |
| Vi → Ko | 60 | 508.5 ms | 732.9 ms | 0.159 | 100% |
| En → Vi | 60 | 857.0 ms | 1,370.4 ms | 0.093 | 100% |
| Ko → Vi | 60 | 871.4 ms | 1,439.6 ms | 0.072 | 100% |
| Zh → Vi | 60 | 837.3 ms | 1,315.9 ms | 0.088 | 100% |

The warm benchmark excludes model and voice loading from per-turn latency. Loading all
models and voices costs about 19.8 seconds on this T4 condition, including 6.1 seconds
for the Mandarin voice. A resident, prewarmed process is therefore a product
requirement rather than a cosmetic optimization.

## Specialized Vietnamese ASR earns its resident footprint

On the same 120 VIVOS utterances, PhoWhisper-small INT8 produced WER 0.1259 and CER
0.0718, compared with WER 0.2920 and CER 0.1768 for multilingual Whisper-small.
Median latency was effectively equal: 175.8 ms versus 174.5 ms. The paired WER delta
was +0.1667 for the multilingual model, with 95% CI [+0.1159, +0.2148] and paired
permutation p≈1e-4. The system therefore keeps both ASR models resident: the specialist
for Vietnamese and the multilingual model for English, Mandarin and Korean.

For reverse directions, English is reported with WER (0.0663) and Korean with CER
(0.0732), because optional Korean spacing makes WER a misleading primary metric.

## Mandarin script control fixes a production-visible ASR error

FLEURS references use Simplified Chinese, while the unprompted decoder emitted
Traditional-only characters in 20/60 rows. A separate controlled A/B primed the
decoder for Simplified Chinese: CER fell from 0.2216 to 0.1229, Traditional rows fell
from 20/60 to 0/60, and median latency increased by only 3 ms. This is both an
evaluation correction and a pipeline fix because the MT language tag, glossary and
Mandarin TTS are configured for Simplified Chinese.

## Terminology control improves consistency, not translation adequacy

On a held-out set authored independently of the glossary, 96 sentence-target probes
contained 279 expected preferred term occurrences. Enabling the 93-term industrial
glossary increased preferred-term recovery from 0.194 to 0.462. This metric measures
surface-form consistency only: a correct synonym counts as a miss, and the probes do
not contain complete translation references. Exact whole-utterance lookup results are
excluded from the claim because they bypass the decoder.

The next quality gate is reference-based MT evaluation (BLEU, chrF and COMET) plus
native-speaker review of Mandarin and Korean safety terminology.

## Architecture and runtime policy

The measured cascade is:

```text
microphone → Silero VAD → source-pinned ASR router
           → guarded Vietnamese transcript corrector
           → glossary-aware NLLB-200 distilled 600M
           → fragment-streaming Piper/Melo VITS → speaker
```

- Vietnamese ASR: PhoWhisper-small, CTranslate2 INT8.
- Reverse ASR: Whisper-small multilingual, CTranslate2 INT8.
- Corrector: Vietnamese correction model with edit, deletion and protected-token guards;
  it accepted 6/300 proposals in the audited run and rejected five content-deleting edits.
- MT: NLLB-200 distilled 600M, CTranslate2 INT8, greedy decoding and a 93-term glossary.
- TTS: Piper voices for Vietnamese/English/Korean and MeloTTS for Mandarin via sherpa-onnx.
- Operations: one resident supervisor, per-stage concurrency guards, real inference health
  probes, watchdog evidence bundles, bounded acceptance/soak gates and a 46-asset
  SHA-256 manifest.
- Routing: the source is selected before ASR; reverse turns require the resident
  multilingual ASR. Transcript-based post-ASR direction switching is rejected.
- Privacy: persistent telemetry is metadata-only by default; transcript previews require
  explicit `VOICEAI_LOG_CONTENT=1` diagnostic consent.

The full six-direction asset manifest contains 46 files totaling 2,087.7 MiB. The
default-required subset is 1,003.4 MiB but does not represent a full six-direction
artifact. Runtime network download, GTCRN and the research caches are disabled by
default. The standard-library offline preflight additionally rejects incomplete model
downloads, manifest path traversal, missing six-direction components, dependency drift
and unsafe runtime environment flags before launch.

## Robustness evidence changed the product decision

Real DEMAND noise was evaluated with 60 utterances per cell. GTCRN significantly
worsened WER in five of twelve cells, including the production-hall PSTATION condition
at 0, 5 and 10 dB. At PSTATION 0 dB, ΔWER was +0.176 [95% CI +0.079, +0.298]. The
denoiser also created VAD triggers where raw PSTATION produced none: 29.2% at 0 dB and
16.7% at 5 dB.

Accordingly, whole-utterance denoise and streaming denoise-before-VAD are both off.
The next robustness experiment is noise-augmented ASR fine-tuning scored directly on
WER/CER; perceptual enhancement scores alone are not an acceptance metric.

## Deployment readiness matrix

| Surface | Proven | Not proven |
|---|---|---|
| Kaggle T4 | Six directions; warm p50/p95; RTF; peak GPU memory; Internet-off execution | Edge power, thermals, NPU compatibility |
| Windows x86-64 supervisor | Five successful warm starts; 1,385–1,743 MiB peak RSS; manifest, healthcheck and watchdog paths; acceptance/soak runner implemented | Realtime CPU ASR; real acoustic half-duplex; multi-hour soak not yet executed |
| Linux/ARM | Launcher, systemd and performance scripts exist | Scripts have not executed on an ARM target |
| Snapdragon/Hexagon | Deployment roadmap exists | QNN artifacts, HTP execution, device latency, energy and thermal evidence |

The Windows CPU ASR path is not real time: measured RTF was approximately 2.6–8 on a
contended development machine. It is an operational rehearsal surface, not a performance
substitute for the target SoC.

## Reproducible setup and verification

`docs/SETUP_SUBMISSION.md` defines separate Windows, T4 and Snapdragon contracts. The
local runtime is installed from `requirements.txt`; the exact extra CPython-3.12 wheels
used by the T4 logs are recorded in `requirements-t4-py312.txt`. Models are prepared
while online, then the runtime is locked offline and verified with:

```powershell
python scripts/offline_bootstrap.py preflight --profile six-direction
python -m pytest prototype/test_live_translate.py prototype/test_asr_router.py device scripts -q
powershell -ExecutionPolicy Bypass -File device/run_device.ps1 -Target en
python device/healthcheck.py --level full
python device/acceptance.py --probe-level full --iterations 10 --interval-s 2
```

The model manifest and full health probe remain mandatory because unit tests use fakes
and do not establish model or audio-device readiness. The acceptance runner writes
atomic JSON evidence with p50/p95/p99, readiness, degraded/busy ratios, supervisor
restarts and RSS growth; its existence is not evidence that the target passed a soak.
The integrated software suite passed **209 tests with one skip** on 20 August 2026, and
the strict model gate verified all **46 assets / 2,189,076,740 bytes** by SHA-256. The
combined environment preflight still fails closed because `sounddevice==0.5.6` is
missing from the active Python environment.

## Limitations and uncertainty

1. No Snapdragon/QNN/Hexagon, power, thermal or sustained-load measurement exists.
2. Six-direction MT lacks reference-based BLEU, chrF and COMET; glossary recovery is not adequacy.
3. TTS lacks MOS/MCD and native-speaker review, especially for Mandarin and Korean.
4. Real microphone, speaker echo, half-duplex and physical source-button behavior are unverified.
5. Automatic first-turn bidirectional routing is not validated; the source must be pinned before ASR.
6. GTCRN failed on real industrial noise; noise-augmented ASR training has not been attempted.
7. The tonal-cache research has synthetic minimal-pair evidence but is not integrated into the live path and has no human-recorded minimal-pair study.
8. The selective INT4 tone-aware research spine has no complete AWQ comparison or deployed mixed-precision result.
9. The official Phase-3 file format and detailed organizer benchmark were absent from the local challenge brief and must be verified before upload.
10. DEMAND licensing metadata is inconsistent across local documents and must be resolved before public redistribution.
11. Log storage is size-bounded but not yet governed by an enforced time-based retention or application-level encryption policy.

## Recommended next steps

1. Select the exact Snapdragon SKU and execute the QNN/HTP deployment gate with p50/p95/p99, RAM, energy and thermals.
2. Build a reference-based six-direction MT set and report BLEU, chrF, COMET and direction-specific failure cases.
3. Run native-speaker terminology and TTS review, plus human-recorded Vietnamese tonal minimal pairs.
4. Fine-tune ASR with measured industrial-noise augmentation and re-evaluate on held-out DEMAND conditions.
5. Bind the implemented pre-ASR source pin to the physical push-to-talk control, then rehearse real microphone and acoustic half-duplex behavior.
6. Run the new acceptance gate for multiple hours with full probes, watchdog recovery and boot persistence on the target ARM device.
7. Resolve model/dataset licenses, upstream voice hashes, time-based privacy retention and the organizer's final submission format.

## Further questions for the organizer and hardware team

- Which Snapdragon SKU, RAM configuration and QNN SDK version define the official benchmark?
- What file/template/page limits apply to the Phase-3 technical submission?
- Is source selection by physical push-to-talk accepted, or is automatic direction detection required?
- Are text-only degraded-mode outputs permitted when TTS is unavailable?
- What audio retention and privacy policy is expected for on-device cache or diagnostic logs?
