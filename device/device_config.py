"""device_config.py — validate everything the device reads from the environment,
at startup, with a message that names the variable and the fix.

The problem this solves is concrete, not hypothetical. `prototype/pipeline_fast.py`
reads its configuration in **dataclass field defaults**:

    ct2_intra_threads: int = int(os.environ.get("VOICEAI_CT2_INTRA_THREADS", 0)) or ...
    cache_enabled:     bool = bool(int(os.environ.get("VOICEAI_CACHE", "0")))

Two consequences:

  1. They are evaluated when `pipeline_fast` is IMPORTED, so anything the device
     wants to set must be set before the import (the supervisor does this).
  2. `VOICEAI_CACHE=true` raises `ValueError: invalid literal for int()` — from
     inside a module import, twenty seconds into a start, with a traceback that
     points at a dataclass and not at the operator's typo. `VOICEAI_TTS_BACKEND=Piper`
     is worse: it does not raise at all, it silently selects the MMS backend and
     the demo has no Mandarin voice.

So the device validates the variables it depends on BEFORE it imports anything
heavy. Every check states the accepted values.

Deliberately NOT validated: variables read only by benchmark/eval scripts that
the device never runs (`VOICEAI_BENCH_JSON`, `VOICEAI_MAX_SAMPLES`,
`VOICEAI_LM_*`, `VOICEAI_RUN_REAL_ASR`, `VOICEAI_MAX_UTTERANCES`). They are
listed at the bottom so the omission is a decision, not an oversight.

Usage
    python device/device_config.py            # validate the current environment
    python device/device_config.py --explain  # ... and print every knob it knows
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEVICE_DIR = Path(__file__).resolve().parent
PROJECT = DEVICE_DIR.parent

TARGETS = ("en", "zh", "ko", "vi")
# ctranslate2 compute types accepted on CPU builds.
CT2_COMPUTE = ("int8", "int8_float32", "int8_float16", "int8_bfloat16",
               "float32", "float16", "bfloat16", "auto", "default")
TTS_BACKENDS = ("piper", "mms")
# sherpa-onnx execution providers.
TTS_PROVIDERS = ("cpu", "cuda", "coreml", "xnnpack", "nnapi", "trt", "directml")
ASR_DEVICES = ("cpu", "cuda", "auto")


class ConfigError(Exception):
    """Raised by require_valid(); the message is operator-facing."""


# ---------------------------------------------------------------------------
# spec: name -> (kind, extra, why-it-matters-to-the-device)
#   kind is one of: intbool, int, float01, float, posfloat, port, enum,
#                   dir_exists, file_exists, str
# ---------------------------------------------------------------------------
SPEC: dict[str, tuple] = {
    # --- device layer's own ---
    "VOICEAI_DEVICE_PORT": ("port", None, "supervisor/healthcheck/watchdog HTTP port"),
    "VOICEAI_TARGET": ("enum", TARGETS, "default translation target"),
    "VOICEAI_KEEPALIVE_S": ("posfloat", None, "seconds between idle keepalive probes; 0 disables"),
    "VOICEAI_MANIFEST": ("file_exists", None, "override path to models.manifest.json"),

    # --- pipeline_fast: threading and load ---
    "VOICEAI_CT2_INTRA_THREADS": ("int", (0, 256), "CTranslate2 intra-op threads (0 = auto)"),
    "VOICEAI_TTS_THREADS": ("int", (1, 256), "TTS (torch / sherpa-onnx) threads"),
    # Per-language override: Mandarin MeloTTS costs ~6x per second of output audio
    # than the other voices, so it gets its own thread budget rather than raising
    # the default and stealing cores from ASR. See local_tts_piper._threads_for.
    "VOICEAI_TTS_THREADS_VI": ("int", (1, 256), "TTS threads for Vietnamese"),
    "VOICEAI_TTS_THREADS_EN": ("int", (1, 256), "TTS threads for English"),
    "VOICEAI_TTS_THREADS_ZH": ("int", (1, 256), "TTS threads for Mandarin"),
    "VOICEAI_TTS_THREADS_KO": ("int", (1, 256), "TTS threads for Korean"),
    "VOICEAI_LOAD_PARALLEL": ("intbool", None, "background-preload TTS while MT decodes"),
    "VOICEAI_GREEDY": ("intbool", None, "beam=1 everywhere"),

    # --- ASR ---
    "VOICEAI_ASR_VI_DIR": ("dir_exists", None, "Vietnamese CT2 ASR model directory"),
    "VOICEAI_ASR_MULTI_DIR": ("dir_exists", None, "multilingual CT2 ASR model directory"),
    "VOICEAI_ASR_DEVICE": ("enum", ASR_DEVICES, "faster-whisper device"),
    "VOICEAI_ASR_COMPUTE_TYPE": ("enum", CT2_COMPUTE, "faster-whisper compute type"),
    "VOICEAI_ASR_PROMPT": ("str", 800, "Whisper initial_prompt (<=224 tokens)"),

    # --- MT ---
    "VOICEAI_MT_GLOSSARY": ("intbool", None, "industrial-term glossary injection"),

    # --- caches ---
    "VOICEAI_CACHE": ("intbool", None, "tonal-perceptual utterance cache"),
    "VOICEAI_CACHE_MAX": ("int", (1, 100000), "utterance cache entries"),
    "VOICEAI_CACHE_COSINE": ("float01", None, "utterance cache hit threshold"),
    "VOICEAI_CACHE_TWO_STREAM": ("intbool", None, "two-stream voting cache"),
    "VOICEAI_CACHE_TS_TONE": ("float01", None, "two-stream tone threshold"),
    "VOICEAI_CACHE_TS_SEG": ("float01", None, "two-stream segment threshold"),
    "VOICEAI_TM": ("intbool", None, "translation memory"),
    "VOICEAI_TM_MAX": ("int", (1, 100000), "translation memory entries"),
    "VOICEAI_TM_COSINE": ("float01", None, "translation memory hit threshold"),

    # --- audio frontend ---
    "VOICEAI_TONE_VAD": ("intbool", None, "tone-energy VAD pre-filter"),
    "VOICEAI_NOISE_FILTER": ("intbool", None, "spectral-subtraction noise filter"),
    "VOICEAI_DENOISE": ("intbool", None, "whole-utterance GTCRN before ASR; research opt-in"),
    "VOICEAI_DENOISE_VAD": ("intbool", None, "streaming GTCRN before VAD; research opt-in"),
    "VOICEAI_DENOISE_GTCRN": ("intbool", None, "GTCRN learned denoiser"),
    "VOICEAI_SILERO_CONFIRM": ("intbool", None, "Silero VAD confirmation stage"),

    # --- privacy ---
    "VOICEAI_LOG_CONTENT": ("intbool", None, "persist transcript previews in structured logs"),

    # --- TTS ---
    "VOICEAI_TTS_BACKEND": ("enum", TTS_BACKENDS, "piper (offline, 4 langs) or mms"),
    "VOICEAI_TTS_AUTO_DOWNLOAD": ("intbool", None, "allow TTS model network download at runtime"),
    "VOICEAI_TTS_PROVIDER": ("enum", TTS_PROVIDERS, "sherpa-onnx execution provider"),
    "VOICEAI_STREAMING_TTS": ("intbool", None, "fragment-streaming TTS (fast first audio)"),
    "VOICEAI_PIPER_MODELS_DIR": ("dir_exists", None, "Piper voice root"),
    "VOICEAI_PIPER_VOICE_EN": ("str", 120, "voice id override for target en"),
    "VOICEAI_PIPER_VOICE_VI": ("str", 120, "voice id override for target vi"),
    "VOICEAI_PIPER_VOICE_ZH": ("str", 120, "voice id override for target zh"),
    "VOICEAI_PIPER_VOICE_KO": ("str", 120, "voice id override for target ko"),

    # --- corrector (optional stage) ---
    "VOICEAI_CORRECTOR": ("intbool", None, "Vietnamese transcript corrector"),
    "VOICEAI_CORRECTOR_MODEL": ("str", 200, "HF id of the corrector"),
    "VOICEAI_CORRECTOR_CT2": ("dir_exists", None, "CT2 export of the corrector"),
    "VOICEAI_CORRECTOR_LOCAL_ONLY": ("intbool", None, "forbid network at runtime"),
    "VOICEAI_CORRECTOR_INT8": ("intbool", None, "torch dynamic INT8 corrector"),
    "VOICEAI_CORRECTOR_MAX_EDIT": ("float01", None, "reject corrections above this edit ratio"),
    "VOICEAI_CORRECTOR_MAX_INPUT": ("int", (1, 4096), "corrector input token cap"),
    "VOICEAI_CORRECTOR_MAX_NEW": ("int", (1, 4096), "corrector output token cap"),
    "VOICEAI_CORRECTOR_MAX_LENGTH": ("int", (1, 4096), "corrector length cap"),
    "VOICEAI_CORRECTOR_MIN_LENGTH": ("int", (0, 4096), "corrector minimum length"),
    "VOICEAI_CORRECTOR_MAX_DROPPED": ("int", (0, 4096), "corrector dropped-token guard"),

    # --- misc ---
    "VOICEAI_PUNCT": ("intbool", None, "punctuation restoration (~1 GB extra model)"),
    "VOICEAI_PROTECTED_TERMS": ("file_exists", None, "protected-terms JSON"),
}

# Read by benchmark/eval scripts only; the device never executes those paths.
NOT_DEVICE_RELEVANT = (
    "VOICEAI_BENCH_JSON", "VOICEAI_MAX_SAMPLES", "VOICEAI_MAX_UTTERANCES",
    "VOICEAI_LM_NUM_EVAL", "VOICEAI_LM_BEAM_K", "VOICEAI_RUN_REAL_ASR",
)

# Model trees the supervisor's required stages load. Absence here is a start
# failure, not a warning — checked only when the manifest verifier is skipped,
# since the manifest is the stronger check when it runs.
REQUIRED_MODEL_DIRS = (
    ("asr", "data/raw/models_ct2/phowhisper-small-int8"),
    ("mt", "data/raw/models_ct2/nllb-200-distilled-600M-int8"),
    ("mt_tokenizer", "data/raw/hf_cache/hub/models--facebook--nllb-200-distilled-600M"),
    ("tts", "data/raw/piper-models"),
    ("frontend", "data/raw/onnx-frontend"),
)


def _fail(name: str, raw: str, expected: str) -> str:
    return f"{name}={raw!r} is invalid — expected {expected}."


def check_var(name: str, raw: str) -> str | None:
    """Return an error string, or None if `raw` is an acceptable value."""
    kind, extra, _ = SPEC[name]
    raw = raw.strip() if isinstance(raw, str) else raw
    if raw == "":
        # An empty string is how a shell "unsets" a variable it exported. Every
        # consumer in prototype/ falls back to its default for "", except int()
        # conversions - which is exactly the crash we are preventing.
        if kind in ("intbool", "int", "float01", "float", "posfloat", "port"):
            return _fail(name, raw, "a number; an empty value reaches int()/float() and raises")
        return None

    if kind == "intbool":
        if raw not in ("0", "1"):
            return _fail(name, raw, "0 or 1 (the code does bool(int(...)) — "
                                    "'true'/'yes'/'on' raise ValueError)")
        return None

    if kind in ("int", "port"):
        try:
            v = int(raw)
        except ValueError:
            return _fail(name, raw, "an integer")
        lo, hi = (1, 65535) if kind == "port" else extra
        if not lo <= v <= hi:
            return _fail(name, raw, f"an integer in [{lo}, {hi}]")
        return None

    if kind in ("float01", "float", "posfloat"):
        try:
            v = float(raw)
        except ValueError:
            return _fail(name, raw, "a number")
        if v != v:
            return _fail(name, raw, "a number (NaN rejected)")
        if kind == "float01" and not 0.0 <= v <= 1.0:
            return _fail(name, raw, "a number in [0.0, 1.0]")
        if kind == "posfloat" and v < 0:
            return _fail(name, raw, "a number >= 0")
        return None

    if kind == "enum":
        if raw.lower() not in extra:
            return _fail(name, raw, "one of " + "|".join(extra))
        if raw != raw.lower():
            return (f"{name}={raw!r} has the wrong case — the code lowercases some "
                    f"of these and not others. Use {raw.lower()!r}.")
        return None

    if kind == "dir_exists":
        if not Path(raw).is_dir():
            return f"{name}={raw!r} does not exist (or is not a directory)."
        return None

    if kind == "file_exists":
        if not Path(raw).is_file():
            return f"{name}={raw!r} does not exist (or is not a file)."
        return None

    if kind == "str":
        if extra and len(raw) > extra:
            return f"{name} is {len(raw)} chars, over the {extra}-char limit."
        return None

    return None


def validate(environ: dict | None = None, *, check_models: bool = False,
             project: Path | None = None) -> tuple[list[str], list[str]]:
    """Validate the environment. Returns (errors, warnings).

    errors   -> the device must refuse to start
    warnings -> the device starts, but the operator should know
    """
    env = os.environ if environ is None else environ
    project = project or PROJECT
    errors: list[str] = []
    warnings: list[str] = []

    for name in sorted(SPEC):
        if name not in env:
            continue
        err = check_var(name, env[name])
        if err:
            errors.append(err)

    # Unknown VOICEAI_* variables: almost always a typo for a real one, and a
    # typo means the setting the operator believes is active is NOT active.
    known = set(SPEC) | set(NOT_DEVICE_RELEVANT)
    for name in sorted(env):
        if name.startswith("VOICEAI_") and name not in known:
            near = _nearest(name, known)
            warnings.append(f"{name} is not a variable any code in this repo reads"
                            + (f" — did you mean {near}?" if near else "."))

    # Cross-variable consistency: the ones where two settings can be individually
    # valid and jointly wrong.
    backend = env.get("VOICEAI_TTS_BACKEND", "").strip().lower()
    if backend == "mms":
        warnings.append("VOICEAI_TTS_BACKEND=mms has no Mandarin voice and is the "
                        "PyTorch quantization testbed, not the demo path. The device "
                        "expects 'piper'.")
    if env.get("VOICEAI_PUNCT") == "1":
        warnings.append("VOICEAI_PUNCT=1 loads an extra ~1 GB model; the measured RAM "
                        "headroom on the reference box does not allow it alongside the "
                        "cascade.")
    if env.get("VOICEAI_CORRECTOR_LOCAL_ONLY") == "0":
        warnings.append("VOICEAI_CORRECTOR_LOCAL_ONLY=0 permits a network fetch at "
                        "runtime. The device is offline by contract; a DNS timeout here "
                        "stalls the first transcript.")
    ka = env.get("VOICEAI_KEEPALIVE_S")
    if ka not in (None, "") and check_var("VOICEAI_KEEPALIVE_S", ka) is None:
        if 0 < float(ka) < 10:
            warnings.append(f"VOICEAI_KEEPALIVE_S={ka} runs a real MT+TTS forward pass "
                            f"that often, which will contend with live speech.")

    if check_models:
        for stage, rel in REQUIRED_MODEL_DIRS:
            d = project / rel
            if not d.is_dir():
                errors.append(f"required {stage} model tree missing: {rel} "
                              f"(expected under {project}).")

    return errors, warnings


def _nearest(name: str, known) -> str | None:
    """Cheap prefix/suffix similarity — no difflib import needed for this."""
    best, best_score = None, 0
    for k in known:
        common = 0
        for a, b in zip(name, k):
            if a != b:
                break
            common += 1
        score = common - abs(len(name) - len(k))
        if score > best_score and common >= 12:
            best, best_score = k, score
    return best


def require_valid(environ: dict | None = None, *, check_models: bool = False,
                  logger=None) -> list[str]:
    """Validate or raise ConfigError. Returns the warning list."""
    errors, warnings = validate(environ, check_models=check_models)
    if logger is not None:
        import devlog
        for w in warnings:
            logger.warn(devlog.E_CONFIG_WARN, w)
    if errors:
        raise ConfigError(
            "invalid device configuration — refusing to load models:\n  "
            + "\n  ".join(errors)
            + "\n\nEvery one of these is read by prototype/ code that would otherwise "
              "fail deep inside a model load. Fix the variable and restart.")
    return warnings


def explain() -> str:
    rows = ["Environment variables the device reads and validates:", ""]
    for name, (kind, extra, why) in sorted(SPEC.items()):
        dom = {"intbool": "0|1", "port": "1-65535", "float01": "0.0-1.0",
               "posfloat": ">=0", "float": "number", "dir_exists": "existing dir",
               "file_exists": "existing file", "str": "text"}.get(kind)
        if kind == "enum":
            dom = "|".join(extra)
        elif kind == "int":
            dom = f"{extra[0]}-{extra[1]}"
        cur = os.environ.get(name)
        rows.append(f"  {name:<32} {dom:<28} {why}"
                    + (f"   [set: {cur!r}]" if cur is not None else ""))
    rows += ["", "Read only by benchmark/eval scripts, not validated: "
                 + ", ".join(NOT_DEVICE_RELEVANT)]
    return "\n".join(rows)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="validate VoiceAI device configuration")
    ap.add_argument("--explain", action="store_true", help="list every knob")
    ap.add_argument("--check-models", action="store_true",
                    help="also require the model trees to exist")
    args = ap.parse_args()
    if args.explain:
        print(explain())
    errors, warnings = validate(check_models=args.check_models)
    for w in warnings:
        print(f"[config] WARN  {w}")
    if errors:
        print("[config] *** INVALID CONFIGURATION ***", file=sys.stderr)
        for e in errors:
            print(f"[config]   {e}", file=sys.stderr)
        return 2
    print(f"[config] ok — {sum(1 for k in SPEC if k in os.environ)} of "
          f"{len(SPEC)} known variables set, {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
