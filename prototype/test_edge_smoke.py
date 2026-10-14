"""Headless end-to-end smoke / integration test for the on-device S2ST edge stack.

Runs a sequence of PASS/FAIL checks against the real modules (no mocks) and prints
a summary table. Exit 0 iff all run checks pass, else 1.

    python test_edge_smoke.py            # full run (slow: loads ASR/MT/TTS models)
    python test_edge_smoke.py --fast     # only the fast pure/edge checks (1-4)

Checks
------
  1. fragment_text        pure splitter: 3-sentence VN → ≥2 fragments, no words lost
  2. denoiser (GTCRN)     5 dB-SNR noisy input → output SNR improves ≥5 dB vs clean
  3. Silero VAD           is_speech(clean)==True and is_speech(gaussian noise)==False
  4. PiperTTS             short vi synth → len>0, sr>0, RMS>0.01 (not silence)
  5. streaming ASR        LocalAgreement-2 final text non-empty and not a run-on
  6. full pipeline        run(sample, "en") → non-empty vi_text, tgt_text, tgt_audio
  7. streaming TTS        first-fragment time <= whole-utterance time * 1.1 (raw Piper)

Checks 5-6 share one FastCascadeS2ST (loaded once). Checks 4/7 share one warm PiperTTS.
Each check is isolated: a failure records and continues; it never aborts the rest.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# Force UTF-8 stdout (Windows cp1252 would choke on VN/CJK text).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Make sibling prototype modules importable regardless of CWD.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SAMPLE_WAV = (HERE.parent / "data" / "processed" / "vivos" / "vivos" /
              "test" / "waves" / "VIVOSDEV01" / "VIVOSDEV01_R002.wav")

# 3-sentence Vietnamese text reused by checks 1 and 7.
VN_3SENT = ("Xin chào mọi người. Hôm nay trời rất đẹp. "
            "Chúng ta hãy bắt đầu cuộc họp ngay bây giờ.")


# --------------------------------------------------------------------------- #
# Tiny test runner
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            ok, msg = True, (detail or "ok")
        except AssertionError as e:
            ok, msg = False, f"assert: {e}"
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        dt = (time.perf_counter() - t0) * 1000
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name} — {msg}  ({dt:.0f} ms)")
        self.results.append((name, ok, msg))

    def summary_and_exit(self):
        print("\n" + "=" * 72)
        print(f"{'CHECK':<26}{'RESULT':<8}DETAIL")
        print("-" * 72)
        n_pass = 0
        for name, ok, msg in self.results:
            if ok:
                n_pass += 1
            tag = "PASS" if ok else "FAIL"
            det = msg if len(msg) <= 40 else msg[:37] + "..."
            print(f"{name:<26}{tag:<8}{det}")
        print("-" * 72)
        total = len(self.results)
        print(f"{n_pass}/{total} passed")
        print("=" * 72)
        sys.exit(0 if n_pass == total else 1)


# --------------------------------------------------------------------------- #
# Shared lazy resources (load once)
# --------------------------------------------------------------------------- #
_STATE: dict = {"clean": None, "sr": None, "piper": None, "pipe": None}


def load_clean():
    if _STATE["clean"] is None:
        audio, sr = sf.read(str(SAMPLE_WAV), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        _STATE["clean"], _STATE["sr"] = np.ascontiguousarray(audio), int(sr)
    return _STATE["clean"], _STATE["sr"]


def get_warm_piper():
    """One PiperTTS instance, warmed on a short vi phrase (shared by checks 4 & 7)."""
    if _STATE["piper"] is None:
        from local_tts_piper import PiperTTS
        p = PiperTTS()
        p.synthesize("Khởi động.", "vi")  # warm: download (if needed) + build engine
        _STATE["piper"] = p
    return _STATE["piper"]


def get_pipe():
    """One FastCascadeS2ST, ASR+MT+TTS loaded once, shared by checks 5 & 6."""
    if _STATE["pipe"] is None:
        from pipeline_fast import FastCascadeS2ST, FastPipelineConfig
        pipe = FastCascadeS2ST(FastPipelineConfig())
        pipe._ensure_asr()  # load whisper for streaming-ASR check
        _STATE["pipe"] = pipe
    return _STATE["pipe"]


def _snr_db(clean: np.ndarray, other: np.ndarray) -> float:
    """SNR of `clean` vs the residual (clean - other), length-aligned."""
    n = min(len(clean), len(other))
    c = clean[:n].astype(np.float64)
    resid = c - other[:n].astype(np.float64)
    sig = float(np.mean(c ** 2))
    noise = float(np.mean(resid ** 2)) + 1e-12
    return 10.0 * np.log10((sig + 1e-12) / noise)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_fragment_text() -> str:
    from streaming_tts import fragment_text, StreamingTTSConfig
    frags = fragment_text(VN_3SENT, StreamingTTSConfig())
    assert len(frags) >= 2, f"expected >=2 fragments, got {len(frags)}: {frags}"
    # No word lost: every sentence must survive intact in the concatenation.
    joined = " ".join(frags)
    for sent in ("Xin chào mọi người", "Hôm nay trời rất đẹp",
                 "Chúng ta hãy bắt đầu cuộc họp ngay bây giờ"):
        assert sent in joined, f"lost text: {sent!r} not in fragments {frags}"
    return f"{len(frags)} fragments, no words lost"


def check_denoiser() -> str:
    from edge_frontend import get_denoiser
    clean, sr = load_clean()
    rng = np.random.default_rng(0)
    sig_pow = float(np.mean(clean.astype(np.float64) ** 2))
    target_snr_db = 5.0
    noise_pow = sig_pow / (10 ** (target_snr_db / 10))
    noise = rng.standard_normal(len(clean)).astype(np.float32) * np.sqrt(noise_pow)
    noisy = (clean + noise).astype(np.float32)

    in_snr = _snr_db(clean, noisy)
    out, osr = get_denoiser().denoise(noisy, sr)
    # Denoiser may resample; bring clean to output rate for a fair residual.
    ref = clean
    if osr != sr:
        m = int(round(len(clean) * osr / sr))
        ref = np.interp(np.linspace(0, len(clean), m, endpoint=False),
                        np.arange(len(clean)), clean).astype(np.float32)
    out_snr = _snr_db(ref, out)
    gain = out_snr - in_snr
    assert gain >= 5.0, (f"SNR gain {gain:.2f} dB < 5 dB "
                         f"(in={in_snr:.2f}, out={out_snr:.2f})")
    return f"SNR {in_snr:.1f}->{out_snr:.1f} dB (+{gain:.1f} dB)"


def check_vad() -> str:
    from edge_frontend import get_vad
    clean, sr = load_clean()
    vad = get_vad()
    speech_on_clean = vad.is_speech(clean, sr)
    assert speech_on_clean is True, "is_speech(clean sample) returned False"
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(sr * 3).astype(np.float32) * 0.05)
    speech_on_noise = vad.is_speech(noise, sr)
    assert speech_on_noise is False, "is_speech(pure gaussian noise) returned True"
    return "speech=True on clean, False on noise"


def check_piper_tts() -> str:
    piper = get_warm_piper()
    audio, sr = piper.synthesize("Chào buổi sáng.", "vi")
    assert len(audio) > 0, "empty audio"
    assert sr > 0, f"bad sample_rate {sr}"
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert rms > 0.01, f"RMS {rms:.4f} <= 0.01 (silence?)"
    extra = ""
    # Optional (guarded): zh voice is installed locally, so exercise it too.
    try:
        za, zsr = piper.synthesize("你好世界。", "zh")
        zrms = float(np.sqrt(np.mean(za.astype(np.float64) ** 2)))
        if len(za) > 0 and zrms > 0.01:
            extra = f"; zh ok (RMS {zrms:.3f})"
    except Exception as e:  # noqa: BLE001
        extra = f"; zh skipped ({type(e).__name__})"
    return f"vi audio {len(audio)/sr:.2f}s @ {sr}Hz RMS={rms:.3f}{extra}"


def check_streaming_asr() -> str:
    from streaming_asr import stream_wav
    clean, sr = load_clean()
    pipe = get_pipe()
    final_text = ""
    for ev, committed, _vol in stream_wav(pipe._asr, clean, sr, "vi", 1.5):
        if ev == "final":
            final_text = committed
    assert final_text.strip(), "streaming ASR produced empty final text"
    nwords = len(final_text.split())
    # Cheap hallucination/run-on guard: a ~4s clip shouldn't yield a 40+ word wall.
    assert nwords < 40, f"implausible run-on ({nwords} words): {final_text!r}"
    return f"{nwords} words: {final_text!r}"


def check_full_pipeline() -> str:
    clean, sr = load_clean()
    pipe = get_pipe()
    res = pipe.run(clean, sr, "en")
    vi, tgt = res["vi_text"], res["tgt_text"]
    audio = res["tgt_audio"]
    assert vi and vi.strip(), "empty vi_text"
    assert tgt and tgt.strip(), "empty tgt_text (en)"
    assert audio is not None and len(audio) > 0, "empty tgt_audio"
    return f"vi={vi[:24]!r}... en={tgt[:24]!r}... audio={len(audio)/max(res['sr'],1):.2f}s"


def check_streaming_tts_first_audio() -> str:
    from streaming_tts import StreamingTTS
    piper = get_warm_piper()  # warm: no download/build cost skews the timing

    # Whole-utterance: synth the entire 3-sentence text in one raw Piper call.
    t0 = time.perf_counter()
    piper.synthesize(VN_3SENT, "vi")
    whole_ms = (time.perf_counter() - t0) * 1000

    # Streaming: time until the FIRST fragment is ready (raw uncached Piper backend).
    stts = StreamingTTS(piper.synthesize)
    t1 = time.perf_counter()
    first = next(stts.stream(VN_3SENT, "vi"))
    first_ms = (time.perf_counter() - t1) * 1000

    ratio = first_ms / max(whole_ms, 1e-6)
    assert first_ms <= whole_ms * 1.1, (
        f"first-fragment {first_ms:.0f} ms > whole {whole_ms:.0f} ms * 1.1")
    return (f"first={first_ms:.0f}ms whole={whole_ms:.0f}ms "
            f"ratio={ratio:.2f} (frag0={first.text[:16]!r})")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Edge S2ST smoke test")
    ap.add_argument("--fast", action="store_true",
                    help="run only the fast pure/edge checks (1-4)")
    args = ap.parse_args()

    if not SAMPLE_WAV.exists():
        print(f"[FATAL] sample wav not found: {SAMPLE_WAV}")
        sys.exit(2)

    r = Runner()
    # Fast, pure-first ordering.
    r.check("1.fragment_text", check_fragment_text)
    r.check("2.denoiser_gtcrn", check_denoiser)
    r.check("3.silero_vad", check_vad)
    r.check("4.piper_tts", check_piper_tts)

    if not args.fast:
        r.check("5.streaming_asr", check_streaming_asr)
        r.check("6.full_pipeline", check_full_pipeline)
        r.check("7.streaming_tts_first", check_streaming_tts_first_audio)
    else:
        print("[skip] --fast: skipping model-heavy checks 5-7")

    r.summary_and_exit()


if __name__ == "__main__":
    main()
