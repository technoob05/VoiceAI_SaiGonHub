# Reproducible setup for the OneVoice Phase-3 prototype

This document separates the three execution surfaces so a development result is
not mistaken for an on-device result.

| Surface | Status | Environment contract |
|---|---|---|
| Windows development supervisor | Measured | Windows 11 x86-64; the current model manifest was generated with Python 3.14; package snapshot in `requirements.txt` |
| Kaggle T4 benchmark | Measured | Python 3.12.13, torch 2.10.0+cu128, CTranslate2 4.7.2, Internet disabled, private versioned inputs |
| Snapdragon / Hexagon NPU | Not implemented or measured | QNN export, ARM launcher execution, energy, thermals and soak testing remain deployment gates |

## 1. Create the local environment

From the repository root on the measured Windows development surface:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an air-gapped installation, stage a wheelhouse while online and use the
idempotent wrapper; it always invokes pip with `--no-index`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_offline.ps1 `
  -Wheelhouse D:\voiceai-wheels -Venv .venv
```

Redirect all caches to the project drive before any model preparation:

```powershell
$voiceAiCache = Join-Path $PWD 'data\raw\hf_cache'
$env:HF_HOME = $voiceAiCache
$env:HF_HUB_CACHE = Join-Path $voiceAiCache 'hub'
$env:HF_XET_CACHE = Join-Path $voiceAiCache 'xet'
$env:TRANSFORMERS_CACHE = Join-Path $voiceAiCache 'hub'
```

## 2. Prepare assets while online

Model acquisition is an explicit setup phase. Runtime auto-download is disabled.

```powershell
python -m ctranslate2.converters.transformers `
  --model vinai/PhoWhisper-small `
  --output_dir data/raw/models_ct2/phowhisper-small-int8 `
  --quantization int8 `
  --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json special_tokens_map.json generation_config.json

python -m ctranslate2.converters.transformers `
  --model openai/whisper-small `
  --output_dir data/raw/models_ct2/whisper-small-multilingual `
  --quantization int8 `
  --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json special_tokens_map.json generation_config.json

python prototype/prepare_corrector.py --ct2

python prototype/local_tts_piper.py download vi_VN-vais1000-medium
python prototype/local_tts_piper.py download en_US-lessac-medium
python prototype/local_tts_piper.py download vits-melo-tts-zh_en
python prototype/local_tts_piper.py download vits-mimic3-ko_KO-kss_low
```

The NLLB INT8 conversion command is maintained in `prototype/README.md`. Do not
replace the shipped model tree during submission packaging: verify it against the
committed manifest instead.

Mandarin and Korean upstream archives do not publish a catalog hash in
`local_tts_piper.py`; the extracted files are therefore guarded by the repository's
46-file SHA-256 manifest. Resolve upstream catalog hashes before a public release.

## 3. Lock the offline runtime policy

These are the submission-safe defaults in code. Setting any value to `1` is an
explicit experimental override and must be recorded in the run evidence.

```powershell
$env:VOICEAI_DENOISE = '0'
$env:VOICEAI_DENOISE_VAD = '0'
$env:VOICEAI_DENOISE_GTCRN = '0'
$env:VOICEAI_CACHE = '0'
$env:VOICEAI_TM = '0'
$env:VOICEAI_TTS_AUTO_DOWNLOAD = '0'
$env:VOICEAI_CORRECTOR_LOCAL_ONLY = '1'
$env:VOICEAI_TTS_BACKEND = 'piper'
$env:VOICEAI_STREAMING_TTS = '1'
$env:VOICEAI_LOG_CONTENT = '0'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
```

Why: GTCRN significantly worsened WER in five of twelve real DEMAND cells. The
legacy T1/T2 caches are not on the live streaming path; T1 single-vector matching
failed tonal minimal-pair safety and T2 v1 strips Vietnamese diacritics. They remain
research artifacts, not submission-runtime features.

`VOICEAI_LOG_CONTENT=0` keeps persistent JSONL telemetry metadata-only: direction,
character counts and latency are retained, while transcript and translation previews
are omitted. The console stream still carries live text to the local UI. Content
previews should only be enabled for an explicitly consented diagnostic session. See
`docs/PRIVACY_AND_DATA_HANDLING.md` for the complete current policy and open gates.

## 4. Verify before starting the demo

```powershell
python scripts/offline_bootstrap.py preflight --profile six-direction
python -m pytest prototype/test_live_translate.py prototype/test_asr_router.py device scripts -q
powershell -ExecutionPolicy Bypass -File device/run_device.ps1 -Target en
python device/healthcheck.py --level full
python device/acceptance.py --probe-level full --iterations 10 --interval-s 2
```

The bootstrap verifies manifest structure and hashes, all six-direction components,
incomplete downloads, exact dependency pins and the fail-closed environment. The
acceptance runner then writes bounded JSON evidence from real supervisor probes. Neither
unit tests nor a host acceptance pass replaces microphone or target-device tests.

## 5. Bidirectional microphone operation

The T4 benchmark knows the source language for every utterance. The current live
microphone path must also receive a source pin before ASR; text language detection
cannot repair a first pass decoded by the wrong ASR model.

```powershell
python device/supervisor.py --mic --pair vi,en --source vi --multilingual-asr
```

For a physical prototype, the push-to-talk control must set `source` before audio
reaches ASR. Missing source is now rejected before ASR, and a reverse source requires
the multilingual model. Automatic first-turn bidirectional routing is intentionally
not claimed.

## 6. Kaggle T4 reproduction contract

The successful kernels were private, Internet-off batch jobs. They used versioned
model, speech, source-library, CPython-3.12 wheel and benchmark-config datasets.
`requirements-t4-py312.txt` records the exact extra wheels printed by the logs;
the base image supplies torch and Transformers. A reproducible rerun must preserve:

- the kernel version and all private dataset versions;
- the model SHA-256 values written into the result JSON;
- the fixed sample seed, warmup count, compute types and per-language TTS threads;
- the separation between CUDA stages (ASR/corrector/MT) and CPU TTS;
- Internet disabled throughout inference.

Primary evidence files are `Results/kaggle_t4/voiceai_bidir-full-v4.json`,
`Results/zh_script_ab.json` and `Results/voiceai_noise-t4-v3-demand.json`.

## 7. Target-device gate

Do not label the prototype “Snapdragon validated” until all of the following exist:

1. QNN/HTP artifacts for the selected Snapdragon SKU and a recorded compiler/runtime version.
2. Cold and warm p50/p95/p99 end-to-end latency on the device.
3. Peak RAM, energy per turn, sustained temperature and throttling measurements.
4. Real microphone, speaker echo/half-duplex and physical push-to-talk validation.
5. Multi-hour soak, watchdog recovery and boot-persistence evidence on Linux/ARM.
