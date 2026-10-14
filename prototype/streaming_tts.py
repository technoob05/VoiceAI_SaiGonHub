"""Streaming TTS — first-audio-fast fragment synthesis + gap-free concatenation.

Fused from references:
  * RealtimeTTS (KoljaB, MIT) — fast-first-fragment yield, producer/consumer
    synth→playback decoupling, 10-15ms fade-in/out at chunk edges, multi-engine
    fallback. (RealtimeTTS/text_to_stream.py, engines/base_engine.py)
  * my-translator (phuc-nt, MIT) — VN-punctuation-aware sentence chunker that
    never loses words. (src/js/read-chunker.js splitIntoChunks)

Why: our old path synthesized the WHOLE translation then played it — first audio
only after the entire sentence was vocoded. Here we split the target text into
speakable fragments, synth the FIRST fragment immediately, and yield audio while
the tail is still being produced. With the in-process sherpa-onnx Piper backend
(prototype/local_tts_piper.py) each fragment is cheap, so perceived first-audio
latency drops well under the 1.5 s budget.

Backend-agnostic: takes any `synth_fn(text, target) -> (np.ndarray float32, sr)`,
so it drives either the MMS-TTS or the Piper backend of pipeline_fast.py.
"""

from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

SynthFn = Callable[[str, str], "tuple[np.ndarray, int]"]

# Sentence terminators incl. Vietnamese/CJK punctuation (from read-chunker.js).
_TERMINATORS = ".!?…。！？;；"
_CJK_TERMINATORS = "…。！？；"
# Space-separated break, CJK break (no space after CJK punctuation), and ASCII
# break with no space -- the last one must not fire inside a decimal ("1.5"),
# hence the digit+terminator lookbehind. Numerals are everywhere in industrial text.
_SENT_SPLIT = re.compile(
    rf"(?<=[{_TERMINATORS}])\s+"
    rf"|(?<=[{_CJK_TERMINATORS}])(?=\S)"
    rf"|(?<![0-9][.!?;])(?<=[.!?;])(?=\S)"
)
_CLAUSE = re.compile(r"[,，、:：]\s*")


@dataclass
class StreamingTTSConfig:
    # First fragment goes out as soon as it has this many chars (fast first-audio),
    # cut at the first clause delimiter at or past it instead of waiting for the
    # sentence terminator. RealtimeTTS minimum_first_fragment_length.
    min_first_fragment_chars: int = 12
    # Force-flush a pending fragment once it exceeds this, so a comma-less run-on
    # sentence still starts speaking. RealtimeTTS force_first_fragment_after_words.
    max_fragment_chars: int = 180
    # A too-short fragment is merged into the next one: a one-word synthesis call
    # sounds clipped and costs a whole call. RealtimeTTS minimum_sentence_length.
    # Fragment 0 is exempt -- it uses min_first_fragment_chars so the fast first
    # fragment is never merged away.
    min_fragment_chars: int = 25
    fade_ms: int = 12                 # click-free concatenation at fragment edges
    trim_silence: bool = True
    silence_threshold: float = 0.01
    queue_max: int = 32               # bounded synth→consumer queue (drop-oldest)


# ---------------------------------------------------------------------------
# Fragmentation (VN/CJK-aware, never loses words)
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENT_SPLIT.split(text.strip())) if s]


def _fast_first_cut(sent: str, min_chars: int) -> "Optional[tuple[str, str]]":
    """Cut the first sentence at the earliest clause delimiter at or after
    `min_chars` so audio starts before the whole sentence is synthesized."""
    if len(sent) <= min_chars:
        return None
    for m in _CLAUSE.finditer(sent):
        if m.end() >= min_chars:
            head, tail = sent[:m.end()].strip(), sent[m.end():].strip()
            return (head, tail) if head and tail else None
    return None


def _merge_short(frags: list[str], cfg: StreamingTTSConfig) -> list[str]:
    out: list[str] = []
    for frag in frags:
        if out:
            floor = cfg.min_first_fragment_chars if len(out) == 1 else cfg.min_fragment_chars
            if len(out[-1]) < floor and len(out[-1]) + len(frag) + 1 <= cfg.max_fragment_chars:
                out[-1] = f"{out[-1]} {frag}"
                continue
        out.append(frag)
    # >2 only: merging a short tail back into a 2-fragment list would swallow the
    # fast first fragment.
    if len(out) > 2 and len(out[-1]) < cfg.min_fragment_chars:
        if len(out[-2]) + len(out[-1]) + 1 <= cfg.max_fragment_chars:
            tail = out.pop()
            out[-1] = f"{out[-1]} {tail}"
    return out


def fragment_text(text: str, cfg: StreamingTTSConfig) -> list[str]:
    """Split into speakable fragments: sentences first, then pack clauses so the
    FIRST fragment is short (fast) but later fragments coalesce for smoothness."""
    sents = split_sentences(text)
    lead = ""
    if sents:
        cut = _fast_first_cut(sents[0], cfg.min_first_fragment_chars)
        if cut:
            lead, sents[0] = cut
    frags: list[str] = [lead] if lead else []
    for sent in sents:
        if len(sent) <= cfg.max_fragment_chars:
            frags.append(sent)
            continue
        # Long sentence: break on clause delimiters, packing up to max_fragment_chars.
        buf = ""
        for piece in _CLAUSE.split(sent):
            piece = piece.strip()
            if not piece:
                continue
            if buf and len(buf) + len(piece) + 1 > cfg.max_fragment_chars:
                frags.append(buf)
                buf = piece
            else:
                buf = f"{buf} {piece}".strip()
        if buf:
            frags.append(buf)
    frags = _merge_short(frags, cfg)
    return frags or ([text.strip()] if text.strip() else [])


# ---------------------------------------------------------------------------
# Audio edge shaping (from RealtimeTTS base_engine.py)
# ---------------------------------------------------------------------------
def _fade(audio: np.ndarray, sr: int, fade_ms: int,
          head: bool = True, tail: bool = True) -> np.ndarray:
    n = int(sr * fade_ms / 1000)
    if n <= 0 or len(audio) < 2 * n or not (head or tail):
        return audio
    a = audio.copy()
    if head:
        a[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    if tail:
        a[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return a


def _trim_silence(audio: np.ndarray, thr: float) -> "tuple[np.ndarray, bool, bool]":
    """Returns (trimmed, cut_head, cut_tail) so the caller can fade only the
    edges it actually cut."""
    nz = np.where(np.abs(audio) > thr)[0]
    if len(nz) == 0:
        return audio, False, False
    return audio[nz[0]: nz[-1] + 1], bool(nz[0] > 0), bool(nz[-1] < len(audio) - 1)


def shape_fragment(audio: np.ndarray, sr: int, cfg: StreamingTTSConfig) -> np.ndarray:
    # Vietnamese is tonal and the rime attack carries tone: fade only where the
    # trim exposed a raw cut, never across an untouched speech onset.
    if not cfg.trim_silence:
        return audio
    audio, cut_head, cut_tail = _trim_silence(audio, cfg.silence_threshold)
    return _fade(audio, sr, cfg.fade_ms, cut_head, cut_tail)


# ---------------------------------------------------------------------------
# Streaming synthesizer
# ---------------------------------------------------------------------------
@dataclass
class FragmentAudio:
    index: int
    text: str
    audio: np.ndarray
    sr: int
    synth_ms: float


class StreamingTTS:
    """Fragment a translation and synthesize fragment-by-fragment.

    `synth_fn` may be a single backend, or a list [primary, fallback, ...]; on a
    synth exception the next backend is tried for that fragment (RealtimeTTS
    multi-engine fallback) so one bad string never drops the whole utterance.
    """

    def __init__(self, synth_fn, cfg: Optional[StreamingTTSConfig] = None):
        self._backends: list[SynthFn] = synth_fn if isinstance(synth_fn, list) else [synth_fn]
        self.cfg = cfg or StreamingTTSConfig()

    def _synth_one(self, text: str, target: str) -> "tuple[np.ndarray, int]":
        last_err: Optional[Exception] = None
        for fn in self._backends:
            try:
                audio, sr = fn(text, target)
                if audio is not None and len(audio) > 0:
                    return np.asarray(audio, dtype=np.float32), int(sr)
            except Exception as e:  # noqa: BLE001 — fall through to next backend
                last_err = e
        if last_err is not None:
            raise last_err
        return np.zeros(0, dtype=np.float32), 16_000

    def stream(self, text: str, target: str) -> Iterator[FragmentAudio]:
        """Synthesize fragments in order, yielding each as soon as it is ready."""
        import time as _t
        for i, frag in enumerate(fragment_text(text, self.cfg)):
            t0 = _t.perf_counter()
            audio, sr = self._synth_one(frag, target)
            audio = shape_fragment(audio, sr, self.cfg)
            yield FragmentAudio(i, frag, audio, sr, (_t.perf_counter() - t0) * 1000)

    def synthesize_full(self, text: str, target: str) -> "tuple[np.ndarray, int, dict]":
        """Non-streaming convenience: concatenate all fragments. Returns
        (audio, sr, stats) where stats.first_audio_ms is the key latency metric."""
        frags = list(self.stream(text, target))
        if not frags:
            return np.zeros(0, dtype=np.float32), 16_000, {"first_audio_ms": 0.0, "n": 0}
        sr = frags[0].sr
        audio = np.concatenate([f.audio for f in frags]) if frags else np.zeros(0, np.float32)
        stats = {
            "first_audio_ms": round(frags[0].synth_ms, 1),
            "total_synth_ms": round(sum(f.synth_ms for f in frags), 1),
            "n": len(frags),
            "fragments": [f.text for f in frags],
        }
        return audio, sr, stats

    # ---- Producer/consumer: background synth thread → bounded queue (drop-oldest) ----
    def stream_threaded(self, text: str, target: str) -> "queue.Queue":
        """Return a Queue that a playback thread can drain. A worker synthesizes
        fragments and pushes FragmentAudio; a final None sentinel marks the end.
        Bounded + drop-oldest so a slow consumer can't unbound memory."""
        q: "queue.Queue" = queue.Queue(maxsize=self.cfg.queue_max)

        def _worker():
            try:
                for frag in self.stream(text, target):
                    if q.full():
                        try: q.get_nowait()
                        except queue.Empty: pass
                    q.put(frag)
            finally:
                q.put(None)

        threading.Thread(target=_worker, daemon=True).start()
        return q


# ---------------------------------------------------------------------------
# Optional live playback (needs sounddevice; degrades gracefully if absent)
# ---------------------------------------------------------------------------
def play_stream(stts: StreamingTTS, text: str, target: str) -> float:
    """Play fragments as they synthesize. Returns measured first-audio latency (ms).
    Requires `sounddevice`; raises ImportError if unavailable."""
    import time as _t
    import sounddevice as sd

    q = stts.stream_threaded(text, target)
    t0 = _t.perf_counter()
    first_ms = -1.0
    while True:
        frag = q.get()
        if frag is None:
            break
        if first_ms < 0:
            first_ms = (_t.perf_counter() - t0) * 1000
        sd.play(frag.audio, frag.sr)
        sd.wait()
    return first_ms


if __name__ == "__main__":
    import argparse, sys
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    import soundfile as sf

    ap = argparse.ArgumentParser(description="Streaming TTS demo (fragment-by-fragment)")
    ap.add_argument("text")
    ap.add_argument("--target", default="vi")
    ap.add_argument("--backend", default="piper", choices=["piper", "mms"])
    ap.add_argument("--out", default="streaming_tts_out.wav")
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()

    if args.backend == "piper":
        from local_tts_piper import PiperTTS
        synth = PiperTTS().synthesize
    else:
        from pipeline_fast import FastCascadeS2ST, FastPipelineConfig
        synth = FastCascadeS2ST(FastPipelineConfig()).synthesize

    stts = StreamingTTS(synth)
    if args.play:
        first = play_stream(stts, args.text, args.target)
        print(f"first-audio latency: {first:.1f} ms")
    else:
        audio, sr, stats = stts.synthesize_full(args.text, args.target)
        sf.write(args.out, audio, sr)
        print(f"wrote {args.out}  ({len(audio)/sr:.2f}s @ {sr}Hz)")
        print(f"fragments ({stats['n']}): {stats['fragments']}")
        print(f"first-fragment synth: {stats['first_audio_ms']} ms  "
              f"total synth: {stats['total_synth_ms']} ms")
