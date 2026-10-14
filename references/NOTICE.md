# Third-party attributions — absorbed techniques

The `prototype/` edge-upgrade modules adapt ideas and (where noted) code from the
open-source projects cloned into this `references/` folder. All are permissively
licensed; this file satisfies the attribution requirement of MIT / Apache-2.0.

| Our module | Absorbed from | Upstream | License | What was taken |
|---|---|---|---|---|
| `prototype/local_tts_piper.py` | my-translator `src-tauri/src/commands/local_tts.rs` | github.com/phuc-nt/my-translator | MIT | Offline Piper/sherpa-onnx TTS: catalog, integrity-gated download, engine LRU, WAV out |
| `prototype/streaming_tts.py` | RealtimeTTS (`text_to_stream.py`, `engines/base_engine.py`) + my-translator `src/js/read-chunker.js` | github.com/KoljaB/RealtimeTTS · phuc-nt/my-translator | MIT · MIT | Fast-first-fragment yield, synth→playback queue decoupling, fade-in/out chunk edges, multi-engine fallback, VN-aware sentence chunking |
| `prototype/streaming_asr.py` | whisper_streaming (`whisper_online.py`) | github.com/ufal/whisper_streaming | MIT | LocalAgreement-2 `HypothesisBuffer`, buffer trimming, committed-suffix `initial_prompt` carry-over |
| `prototype/edge_frontend.py` (SileroVAD) | RealtimeSTT two-stage VAD idea + sherpa-onnx VAD runtime/model | github.com/KoljaB/RealtimeSTT · github.com/k2-fsa/sherpa-onnx | MIT · Apache-2.0 | Learned-VAD confirmation stage after cheap tone-energy gate |
| `prototype/edge_frontend.py` (GtcrnDenoiser) | sherpa-onnx `offline-speech-enhancement-gtcrn.py` | github.com/k2-fsa/sherpa-onnx | Apache-2.0 | GTCRN industrial-noise front-end |
| `prototype/live_translate.py` | RTranslator staging + my-translator stdin-PCM sidecar IPC + sherpa VAD segmentation | github.com/niedev/RTranslator · phuc-nt/my-translator · k2-fsa/sherpa-onnx | Apache-2.0 · MIT · Apache-2.0 | Turn-segmented live loop, JSON-line sidecar protocol, model-warm service |

## Round 3 additions (2026-07-14)

| Our module | Absorbed from | Upstream | License | What was taken |
|---|---|---|---|---|
| `prototype/punct_restore.py` | model `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase` (via `punctuators`) | huggingface.co/1-800-BAD-CODE | Apache-2.0 (model) | Vietnamese punctuation + truecase + sentence segmentation of the ASR transcript |
| `prototype/playback.py` | huggingface/speech-to-speech (`local_audio_streamer.py`, `cancel_scope.py`) | github.com/huggingface/speech-to-speech | Apache-2.0 | Queue-drained non-blocking playback + interrupt flag → barge-in |
| `prototype/DEPLOYMENT_ROADMAP.md` (whisper.cpp Hexagon path) | ggml-org/whisper.cpp (`ggml-hexagon` backend, `convert-h5-to-ggml.py`) | github.com/ggml-org/whisper.cpp | MIT | PhoWhisper→GGML→q5→Hexagon NPU on-device ASR path (documented) |

Reference repos this round (gitignored): `references/whisper.cpp` (MIT), `references/hf-speech-to-speech` (Apache-2.0), `references/VFastPunct` (checkpoint link dead — not used), `references/JointCapPunc`.

## Deployment roadmap absorbed (documented, not yet coded)

From **RTranslator** (Apache-2.0) and **sherpa-onnx** (Apache-2.0), for the Qualcomm
Hexagon shipping path:

- **Model-split + host-side external KV-cache** (RTranslator `Recognizer.java`,
  `Translator.java`): re-export PhoWhisper/NLLB into fixed-purpose sub-graphs
  (mel-init / encoder / cache-init / decoder-step / detok) driven by a host decode
  loop — the shape the QNN/HTP compiler converts cleanly.
- **Single-onnxruntime consolidation** (sherpa-onnx): denoiser + Silero VAD +
  streaming Zipformer ASR + Piper TTS all under one onnxruntime; retire
  faster-whisper/CTranslate2. sherpa-onnx ships a first-class **QNN backend**
  (`libQnnHtp.so`, precompiled context binaries, per-SoC HTP config in
  `scripts/qnn/`) that runs Zipformer-transducer / SenseVoice on Hexagon —
  Python bindings do NOT expose QNN, so the NPU runtime is C++/Android, with the
  Python examples as the porting reference.
- **RAM-tiered runtime** (RTranslator): one build, two operating points selected
  at startup by available RAM.
