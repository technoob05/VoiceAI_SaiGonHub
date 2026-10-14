# On-device deployment roadmap — the honest gap to a shipped device

This documents what it takes to close the gap from *"Python pipeline + demo"* to
*"runs on the competition's Qualcomm Snapdragon device"*, and records the
feasibility findings that constrain it. Nothing here is faked as done — it is the
plan and the hard blocks.

## ⭐ Breakthrough finding (2026-07-14, round 2): whisper.cpp has a native Hexagon backend

`references/whisper.cpp` (ggml-org) is the missing on-device ASR runtime — and it
**contains a native Qualcomm Hexagon (HTP) NPU backend**, not just CPU/Vulkan/CoreML:
- `ggml/src/ggml-hexagon/` — full backend (`ggml-hexagon.cpp`, `htp-drv.cpp`) + an
  on-DSP kernel library `htp/` (matmul/flash-attn/softmax/rope/act/argsort/binary…),
  built for HTP **v73/v75/v79/v81** = Snapdragon 8 Gen 2 / Gen 3 / 8 Elite / next-gen.
- Runs quantized weights (`Q4_0`, `Q8_0`, `Q4_1`, `MXFP4`, `IQ4_NL`) **on the NPU**.
- Ships `models/convert-h5-to-ggml.py` — converts a HuggingFace Whisper fine-tune
  (i.e. **our PhoWhisper**) → GGML; then `examples/quantize` → q5_0.

**Why this changes the plan:** it lets us run **our PhoWhisper on the Hexagon NPU**
while keeping Vietnamese quality (it's the same PhoWhisper weights, just reformatted) —
something sherpa-onnx's ONNX-QNN EP does *not* do cleanly for Whisper today. This is a
better ASR-on-Hexagon bet than the QNN-EP path in finding #2 below.

Concrete path (each a real step, none faked):
1. `python models/convert-h5-to-ggml.py ./PhoWhisper-small/ ./whisper .` → GGML f16.
2. `examples/quantize ggml-model.bin ggml-model-q5_0.bin q5_0` → **~170–190 MB** for small.
3. Validate WER on desktop `whisper-cli` vs our CT2 INT8 baseline (de-risk the format).
4. Build `examples/whisper.android` (Kotlin+JNI+CMake) with the q5 model — batch
   transcription of VAD-segmented turns (streaming on Android CPU is ~5× slower than
   realtime; batch-per-turn is the viable mode).
5. Target milestone: `-DGGML_HEXAGON` + Hexagon SDK (signed DSP skels, Snapdragon
   8 Gen 2+) → NPU-accelerated encode/decode. This is the real prize.

Honest caveats: GGML is a *parallel* format — our CT2 INT8 artifact is not reusable,
convert from original HF weights. The Hexagon build needs the proprietary Hexagon SDK
+ signed skels and targets only recent 8-series (v73+); older SoCs get CPU. No turnkey
Android+Hexagon sample — the JNI+Hexagon glue is our integration work. Keep sherpa-onnx
for TTS/VAD/denoise → this becomes a **two-runtime app** (whisper.cpp ASR + sherpa TTS).

## Verified findings (2026-07-14)

1. **No Vietnamese streaming Zipformer exists.** sherpa-onnx ships streaming
   Zipformer for en / zh / **ko** / de / bn only — **not Vietnamese**. Our ASR is
   **PhoWhisper-small** (VinAI, Vietnamese-tuned). Swapping to a non-Vietnamese
   Zipformer to gain speed / a ready QNN export would **degrade Vietnamese WER on
   the core language** — a bad trade. **Decision: keep PhoWhisper.**

2. **The Hexagon (QNN) runtime is C++/Android only.** sherpa-onnx has a first-class
   QNN backend (`libQnnHtp.so`, precompiled context binaries, per-SoC HTP config in
   `scripts/qnn/`) that runs Whisper / Zipformer / SenseVoice on the Hexagon HTP.
   But the **Python bindings do NOT expose the QNN provider** (only cpu/cuda/coreml/
   xnnpack/nnapi/directml/spacemit). So `pipeline_fast.py` can never call Hexagon —
   the NPU runtime must be the C-API / C++ runtime (or the Android AAR). The Python
   code is the *porting reference*, not the shipping runtime.

3. **QNN needs the hardware + SDK.** A real Hexagon PoC requires a Snapdragon dev
   device + the Qualcomm QNN SDK + a C++ build. It cannot be produced or run on a
   Windows/CPU dev box. This is a genuine hardware/tooling block, not a coding gap.

## The path that keeps Vietnamese quality AND reaches the NPU

Because sherpa-onnx's QNN exports **include Whisper**, our PhoWhisper can ride the
same rail without abandoning Vietnamese:

```
PhoWhisper-small (CT2 INT8, today, Python)
      │  export encoder/decoder to ONNX (icefall/whisper → onnx)
      ▼
sherpa-onnx Whisper runner (ONNX, xnnpack CPU)   ← unifies runtime with our TTS/VAD/denoiser
      │  QNN context-binary compile (scripts/qnn/, C++)
      ▼
Hexagon HTP (libQnnHtp.so)  ← the shipped device path
```

Steps, in order (each is a separate work item, none faked as done):

1. **Unify on one onnxruntime (Python, achievable next).** Today denoiser + Silero
   VAD + Piper/MeloTTS TTS already run under onnxruntime; only ASR (CT2) and MT (CT2)
   are outside it. Export PhoWhisper to ONNX and run it via sherpa-onnx's Whisper
   runner with the `xnnpack` provider → deletes the faster-whisper/CTranslate2
   dependency and puts ASR on the same runtime as everything else. **This is the
   next real coding step and it is doable in Python.**
2. **Model-split + host-side external KV-cache** (RTranslator pattern): re-export
   PhoWhisper/NLLB into fixed-purpose sub-graphs (mel-init / encoder / cache-init /
   decoder-step / detok) driven by a host decode loop with an explicit external
   KV-cache and per-step tensor free. This is the shape the QNN/HTP compiler converts
   cleanly (it dislikes one big dynamic graph) and it cuts RAM (RTranslator: NLLB
   2.5→1.3 GB, Whisper 1.4→0.9 GB).
3. **QNN context-binary compile + C++ runtime** (needs device + SDK): compile the
   split graphs to a Hexagon context binary via `scripts/qnn/`, run through the
   sherpa-onnx C-API. TTS/VAD/denoiser stay on ARM CPU (xnnpack); only the heavy
   ASR/MT matmuls go to HTP.
4. **RAM-tiered runtime** (RTranslator pattern): one build, two operating points
   (low-RAM slow path vs full-speed) chosen at startup from available RAM.

## What is a productization gap vs the reference apps (honest)

- **RTranslator** already runs 100% offline **on a phone** (CPU-only ONNX, no NPU).
  We are still Python-on-desktop. Step 1 above (single onnxruntime) is what makes an
  Android AAR / C++ port realistic.
- **my-translator** is a signed, auto-updating **Tauri desktop app**; its offline
  mode is **Apple-Silicon-only (MLX)** and its main engines are cloud. On Windows it
  cannot run offline. We ship a Gradio web demo + a pywebview desktop wrapper
  (`desktop_app.py`) — lighter than a signed installer, but genuinely offline on
  Windows, which my-translator is not.

## Not blocked — shipped this round (see EDGE_UPGRADES.md)

Streaming TTS, LocalAgreement streaming ASR (+ hallucination guard, + VAC/adaptive
pacing), GTCRN denoiser, Silero VAD confirm, pre-roll + min-speech turn-taking,
live app (mic/file/sidecar), web demo, desktop-app wrapper. All Python, all offline,
all verified. The device/NPU port (steps 2–3) is the remaining honest gap.
