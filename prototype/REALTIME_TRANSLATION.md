# Local realtime translation workflow

## Runtime path

```text
microphone (16 kHz mono)
  -> Silero VAD + endpointing (+ optional GTCRN denoise)
  -> PhoWhisper INT8 / LocalAgreement partial ASR
       partial text -> UI only
       immutable final
  -> Vietnamese correction (local mBART, conservative safety gate)
  -> optional punctuation restoration
  -> NLLB-200 INT8 translation
  -> Piper/ONNX streaming TTS
  -> speaker
```

The important contract is that an unstable ASR partial is never translated.
Only the immutable end-of-turn transcript enters correction and MT. This avoids
translation churn and matches the final/partial separation in the OmniGlass
reference architecture.

## Transcript correction policy

The default live model is `bmd1905/vietnamese-correction-v2`, loaded locally via
Transformers. It is a grammar/spelling restoration layer, not a free-form LLM.
Its proposal is accepted only when all guards pass:

- numbers are unchanged;
- negation words (`không`, `chưa`, `đừng`, ...) are unchanged;
- all-uppercase acronyms and protected technical terms are preserved;
- length remains close to the ASR transcript;
- token edit ratio is at most 0.35.

If the checkpoint is missing, inference fails, or a guard rejects the proposal,
the raw ASR final is sent to MT. Both raw and selected transcripts are included
in the JSON events for debugging and evaluation.

This corrector can repair text-form errors. It cannot reliably recover an
arbitrary acoustic substitution while preserving meaning. Improve those cases
with cleaner audio, PhoWhisper adaptation, domain prompts, or ASR N-best
rescoring; do not let a text model invent missing content.

## One-time preparation

Network is used only to acquire the model files:

```powershell
python prototype/prepare_corrector.py --ct2
```

The checkpoint and its CTranslate2 INT8 export are stored under `data/raw/` on drive D. Normal live
runtime uses `local_files_only=True`, so it makes no inference API call and does
not silently download a missing checkpoint.

The live process prewarms ASR, correction, MT, and the selected Piper voice
before it prints `[ready]`. Keep that process alive. `--lazy-load` exists for
debugging, but charges model initialization to the first spoken turn.

## Run

```powershell
# Microphone, Vietnamese -> English, correction enabled by default
python prototype/live_translate.py mic --target en

# Test the exact VAD/live path with a WAV, without audio playback
python prototype/live_translate.py file --wav path/to/input.wav --target en --no-play

# Ablation/baseline without correction
python prototype/live_translate.py mic --target en --no-correct
```

The sidecar emits a `corrected` event between source partials and target partials:

```json
{"type":"corrected","raw":"...","text":"...","correction":{"accepted":true,"reason":"accepted"},"latency_ms":42.1}
```

The local Windows CPU smoke test measured about 1.19 s for the sample correction
after CT2 INT8 export (one sample, not a statistically valid latency claim).
The full current CPU cascade is not realtime yet: a 4.06 s VIVOS cold batch took
57.67 s, and the live LocalAgreement path took 101.39 s before this prewarm
change. These numbers validate wiring, not the latency target. The NPU/QNN
deployment and warm-turn benchmark remain required.
Measure `corrector_ms`, first-audio latency, p50, and p95 on the deployment
device before making a realtime-latency claim. Desktop CPU results do not prove
Snapdragon Hexagon performance; the final deployment still needs model export
and a QNN/QAIRT benchmark.
