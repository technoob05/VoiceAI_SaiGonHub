# prototype/ — Cascaded S2ST skeleton (local CPU dev)

End-to-end Vietnamese-centric speech-to-speech translation, **public checkpoints only**, runs on CPU. Use this as the engineering baseline; replace components with our tone-aware quantized variants when ready.

For the offline live workflow with final-only Vietnamese correction, see
[REALTIME_TRANSLATION.md](REALTIME_TRANSLATION.md).

Three pipelines + two UIs + one benchmark:

| Entry point | What it is | Quality | Speed (CPU) |
|---|---|---|---|
| [pipeline.py](pipeline.py) | FP32 HF baseline cascade — what reviewers will compare against | reference | RTF ~5x |
| [pipeline_fast.py](pipeline_fast.py) | **INT8 optimized** — faster-whisper + CT2-NLLB + dynamic-quant VITS | ~same | RTF ~1-2x (target) |
| [web_demo.py](web_demo.py) | Gradio one-shot mic / upload UI on :7860 | uses pipeline.py | — |
| [web_demo_streaming.py](web_demo_streaming.py) | Gradio continuous-listening + silero-VAD on :7861 | uses pipeline.py | — |
| [bench.py](bench.py) | Single-pipeline batch latency + WER/TER | — | — |
| [bench_compare.py](bench_compare.py) | **FP32 vs INT8 side-by-side bench**, warmup-corrected, JSON output | — | — |

## Optimization stack (in pipeline_fast.py)

CPU FP32 → CPU INT8 path, ~3.9× model-size reduction + ~5-10× expected inference speedup:

| Stage | FP32 baseline | INT8 optimized | Technique |
|---|---|---|---|
| ASR | HF `WhisperForConditionalGeneration` | **`faster-whisper`** | CTranslate2 INT8 + silero-VAD trim + beam=1 |
| MT | HF `AutoModelForSeq2SeqLM` | **`ctranslate2.Translator`** | CT2 INT8 + beam=1 + dynamic max_decoding_length |
| TTS | HF `VitsModel` FP32 | `torch.quantize_dynamic(..., qint8)` | Dynamic INT8 over Linear layers |
| Decode | beam=5 default | **beam=1** | Greedy decoding everywhere |

**Model size delta** (on disk):
- PhoWhisper-small: 967 MB FP32 → **240 MB INT8** (4.0× smaller)
- NLLB-200-distilled-600M: 2,367 MB FP32 → **621 MB INT8** (3.8× smaller)
- Total: 3,334 MB → **861 MB** (3.9× smaller)

### One-time conversion (run once)
```bash
# CT2 converters output to data/raw/models_ct2/
python -m ctranslate2.converters.transformers \
    --model vinai/PhoWhisper-small \
    --output_dir data/raw/models_ct2/phowhisper-small-int8 \
    --quantization int8 \
    --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json special_tokens_map.json generation_config.json

# NLLB needs the AutoTokenizer monkey-patch (transformers 5.0.0 bug); see commit log.
python -c "import transformers; \
    _o = transformers.AutoTokenizer.from_pretrained; \
    transformers.AutoTokenizer.from_pretrained = lambda m, *a, **k: transformers.NllbTokenizerFast.from_pretrained(m, *a, **k) if 'nllb' in str(m).lower() else _o(m, *a, **k); \
    from ctranslate2.converters import TransformersConverter; \
    TransformersConverter('facebook/nllb-200-distilled-600M', copy_files=['tokenizer.json','tokenizer_config.json','sentencepiece.bpe.model','special_tokens_map.json']).convert('data/raw/models_ct2/nllb-200-distilled-600M-int8', quantization='int8', force=True)"
```

### Running the optimized pipeline
```bash
python prototype/pipeline_fast.py \
    --audio data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav \
    --target en --out prototype/out_fast.wav

# Side-by-side bench (warmup-corrected)
python prototype/bench_compare.py --n 10 --target en --out Results/bench_compare_en_n10.json
```

## Stages

| Stage | Model | Params | Disk | What it does |
|-------|-------|------:|-----:|--------------|
| ASR  | [`vinai/PhoWhisper-small`](https://huggingface.co/vinai/PhoWhisper-small) | 244 M | ~970 MB | Vietnamese audio → Vietnamese text |
| MT   | [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) | 600 M | ~2.4 GB | Vi text → {En, Zh, Ko} text |
| TTS  | [`facebook/mms-tts-{eng,cmn,kor,vie}`](https://huggingface.co/facebook/mms-tts-eng) | ~60–140 M each | ~150 MB / lang | target text → target audio |

First run downloads everything into `<repo>/data/raw/hf_cache/`. Subsequent runs are offline.

## Quick start

```bash
# 1) Install runtime deps (one-time)
pip install transformers soundfile torch

# 2) Pick a Vietnamese WAV from VIVOS test set
ls data/processed/vivos/vivos/test/waves/VIVOSDEV01/ | head -3
# -> VIVOSDEV01_R002.wav  VIVOSDEV01_R003.wav  ...

# 3) Run cascade Vi -> En
python prototype/pipeline.py \
    --audio data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav \
    --target en \
    --out prototype/out_en.wav

# 4) Same input, Mandarin output
python prototype/pipeline.py \
    --audio data/processed/vivos/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav \
    --target zh \
    --out prototype/out_zh.wav
```

Expected output:
```
[input] .../VIVOSDEV01_R002.wav  duration=4.07s  sr=16000
[load] ASR: vinai/PhoWhisper-small
[load] MT:  facebook/nllb-200-distilled-600M
[load] TTS: facebook/mms-tts-eng
[asr]   vi : trong số các quốc gia công nghiệp phát triển
[mt]    en : among the developing industrial countries
[tts]   wrote prototype/out_en.wav  (3.2s @ 16000 Hz)
[latency] asr=2400 ms   mt=4100 ms   tts=900 ms   total=7400 ms
[rtf]   real-time factor = 1.82x  (>1 means slower than real-time on this device)
```

## Web demo (Gradio)

```bash
pip install gradio
python prototype/web_demo.py
# browser auto-opens http://127.0.0.1:7860
```

Features:
- **Mic input** (browser MediaRecorder) OR upload a WAV.
- Target language radio: English / Mandarin / Korean / Vi-round-trip.
- Shows Vi transcript (ASR), target text (MT), playable target audio (TTS).
- Live per-stage latency display: ASR / MT / TTS / total + RTF.
- Pre-loaded examples from VIVOS test set.

Notes:
- First click loads models (~30-60 s after the click). After that, requests are ~100 s on CPU fp32 — that's the **fp32 baseline** the method paper will quantize against.
- Mic captures at 48 kHz from the browser; resampled to 16 kHz inside the cascade.
- Single-user / serialized queue (CPU fp32 OOMs under concurrent requests).
- Stop with `Ctrl+C` in the terminal.

To share the URL on LAN (e.g. judges at the competition demo), edit `web_demo.py` and change `server_name="127.0.0.1"` → `"0.0.0.0"`. To get a public ngrok URL, pass `share=True` to `.launch()`.

## CLI flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--audio` | (required) | Input WAV. Mono or stereo, any sample rate (resampled to 16 kHz). |
| `--target` | `en` | One of `en`, `zh`, `ko`, `vi`. Vi-to-vi acts as voice cloning / round-trip. |
| `--out` | `prototype/out.wav` | Output WAV path. |
| `--device` | `cpu` | `cpu` / `cuda` / `mps`. CPU is the default for portability. |
| `--asr-id` | `vinai/PhoWhisper-small` | Any Whisper-family HF id (e.g. `openai/whisper-small`, `vinai/PhoWhisper-medium`). |
| `--mt-id`  | `facebook/nllb-200-distilled-600M` | Any NLLB-family HF id. |

## What this is NOT

- **Not on-device deployment-ready.** PyTorch CPU fp32 ≈ 1.5–2× real-time. Production target is Hexagon NPU INT4-mixed via Qualcomm AI Hub — that's a later milestone.
- **Not streaming.** Currently transcribes one utterance at a time. Streaming chunked decode will come when we wire to a real microphone.
- **Not robust to noisy input.** Public checkpoints trained mostly on clean speech. Robustness comes from the noise-augmentation pipeline in `exp/exp08_noise_aug_pipeline.py` (planned).
- **No method-novelty yet.** This is just the skeleton — the tone-aware mixed-precision quantization spine (see [../CLAUDE.md](../CLAUDE.md) §4) is implemented in `exp/expNN_*.py`, not here.

## When to extend

| Trigger | Action |
|---------|--------|
| Want to swap the ASR backbone | change `--asr-id`. PhoWhisper-medium (~1.5 GB) is the obvious next step. |
| Want to compare against VietASR | wrap [`khanld/vietasr`](https://huggingface.co/khanld/vietasr) in a sibling `asr_vietasr.py` (NOT a Whisper class — its API is different). |
| Want to record from microphone | add `--mic` flag using `sounddevice`. Trivial loop; do this only once we need a live demo. |
| Want to quantize a stage | apply the selective-precision config from `exp/expNN_*.py` to the loaded model **before** the `.eval()` call. Keep this prototype as the fp32 reference. |

## Disk layout after first run

```
data/raw/hf_cache/hub/
  ├── models--vinai--PhoWhisper-small/
  ├── models--facebook--nllb-200-distilled-600M/
  └── models--facebook--mms-tts-{eng,cmn,kor,vie}/   (one per target lang you used)
```
