"""Streaming ASR — LocalAgreement-2 incremental decoding over faster-whisper.

Ported from ufal/whisper_streaming (MIT) — HypothesisBuffer + OnlineASRProcessor
— and adapted to faster-whisper word timestamps.

Why: our old run_streaming() ran faster-whisper once over the WHOLE utterance and
emitted whichever segments it happened to finalize — partials could still change,
and compute grew with utterance length. LocalAgreement-2 fixes both:

  * A word is COMMITTED only when two consecutive re-transcriptions of the growing
    audio buffer agree on it (longest common prefix). Committed text never retracts
    → stable captions the user (and the downstream MT) can trust mid-utterance.
  * Once text is committed and the buffer grows past a threshold, the audio buffer
    is TRIMMED at the last committed word boundary and its offset advanced, so
    per-iteration decode cost stays bounded (critical for the <1.5 s NPU budget).
  * A suffix of committed text is fed back as `initial_prompt`, so trimming the
    audio does not cause a decoding discontinuity (accuracy survives the cut).

Self-contained: feed 16 kHz float32 mono chunks; get (committed, volatile) text.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

WordTS = "tuple[float, float, str]"  # (start, end, word)


class HypothesisBuffer:
    """LocalAgreement-n prefix confirmation (whisper_streaming, MIT). Commits the
    longest common prefix between the previous and current hypothesis."""

    def __init__(self):
        self.committed_in_buffer: list = []
        self.buffer: list = []
        self.new: list = []
        self.last_committed_time = 0.0
        self.last_committed_word: Optional[str] = None

    def insert(self, new: list, offset: float) -> None:
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_committed_time - 0.1]
        if not self.new:
            return
        a, b, t = self.new[0]
        if abs(a - self.last_committed_time) < 1 and self.committed_in_buffer:
            cn, nn = len(self.committed_in_buffer), len(self.new)
            # Drop an already-committed n-gram (n up to 5) re-emitted at the seam.
            for i in range(1, min(cn, nn, 5) + 1):
                c = " ".join(self.committed_in_buffer[-j][2] for j in range(i, 0, -1))
                tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                if c == tail:
                    for _ in range(i):
                        self.new.pop(0)
                    break

    def flush(self) -> list:
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            if not self.buffer:
                break
            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_committed_word = nt
                self.last_committed_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(commit)
        return commit

    def pop_committed(self, t: float) -> None:
        while self.committed_in_buffer and self.committed_in_buffer[0][1] <= t:
            self.committed_in_buffer.pop(0)

    def complete(self) -> list:
        return self.buffer


class OnlineASR:
    """Incremental faster-whisper decoder with LocalAgreement-2 commit + buffer trim.

    Parameters
    ----------
    model : faster_whisper.WhisperModel
    language : ASR language code (e.g. "vi").
    trim_seconds : trim the audio buffer once it exceeds this and text is committed.
    trim_policy : "word" cuts at the last committed word (original behaviour);
        "segment" cuts at the end of the second-to-last Whisper segment, so the
        decoder keeps a whole segment of left context after the cut.
    no_speech_ceiling : drop words coming from segments Whisper itself scored as
        more likely silence than speech than this.
    """

    def __init__(self, model, language: str = "vi", beam_size: int = 1,
                 trim_seconds: float = 12.0, prompt_chars: int = 200,
                 sample_rate: int = 16_000, logprob_floor: float = -0.8,
                 max_words_per_sec: float = 12.0, no_speech_ceiling: float = 0.9,
                 trim_policy: str = "word"):
        self.model = model
        self.language = language
        self.beam_size = beam_size
        self.trim_seconds = trim_seconds
        self.prompt_chars = prompt_chars
        self.sr = sample_rate
        # Hallucination guards: word_timestamps=True makes Whisper append a phantom
        # trailing segment on short clips. It is separable by (a) implausibly fast
        # speech rate (many words in a fraction of a second) — language/noise-robust —
        # and (b) a low avg_logprob floor. Both drop the phantom, keep real speech.
        # (c) no_speech_prob is Whisper's OWN silence posterior for the segment: on
        # factory noise it is the cheapest and most direct hallucination signal, and
        # unlike avg_logprob it stays meaningful when the invented text is fluent.
        self.logprob_floor = logprob_floor
        self.max_words_per_sec = max_words_per_sec
        self.no_speech_ceiling = no_speech_ceiling
        if trim_policy not in ("word", "segment"):
            raise ValueError(f"trim_policy must be 'word' or 'segment', got {trim_policy!r}")
        self.trim_policy = trim_policy
        self.init()

    def init(self, offset: float = 0.0) -> None:
        """Reset decoder state. ``offset`` re-bases the buffer clock so word
        timestamps stay absolute when a caller drops audio before restarting."""
        self.audio = np.zeros(0, dtype=np.float32)
        self.buffer_time_offset = offset
        self.hyp = HypothesisBuffer()
        self.hyp.last_committed_time = offset
        self.committed: list = []

    def insert_audio(self, chunk: np.ndarray) -> None:
        self.audio = np.concatenate([self.audio, chunk.astype(np.float32)])

    def _prompt(self) -> str:
        """A suffix of already-committed, already-scrolled-away text, carried into
        the next decode as context across the trim boundary."""
        k = max(0, len(self.committed) - 1)
        while k > 0 and self.committed[k - 1][1] > self.buffer_time_offset:
            k -= 1
        prev = [w[2] for w in self.committed[:k]]
        p, out = 0, []
        while prev and p < self.prompt_chars:
            x = prev.pop(-1)
            p += len(x) + 1
            out.append(x)
        return " ".join(out[::-1])

    def _transcribe(self, prompt: str) -> "tuple[list, list]":
        # Same anti-hallucination guards as the main pipeline: silero VAD trim (kills
        # the trailing-silence region where Whisper invents a phantom sentence) plus
        # compression-ratio gating.
        segments, _ = self.model.transcribe(
            self.audio, language=self.language, task="transcribe",
            beam_size=self.beam_size, temperature=0.0, word_timestamps=True,
            condition_on_previous_text=False, initial_prompt=prompt or None,
            no_repeat_ngram_size=3, repetition_penalty=1.2,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
            compression_ratio_threshold=1.8,
        )
        words: list = []
        segment_ends: list = []
        for seg in segments:
            sw = seg.words or []
            if not sw:
                continue
            # Drop phantom/hallucinated segments: Whisper's own no-speech posterior,
            # too-low confidence, or a physically impossible speech rate (Whisper
            # crams a long invented sentence into a sub-second trailing window).
            dur = max(seg.end - seg.start, 1e-3)
            if getattr(seg, "no_speech_prob", 0.0) > self.no_speech_ceiling:
                continue
            if seg.avg_logprob < self.logprob_floor:
                continue
            if len(sw) / dur > self.max_words_per_sec:
                continue
            # Only surviving segments are trim candidates — cutting at a rejected
            # segment's end would anchor the buffer to a hallucination.
            segment_ends.append(seg.end)
            for w in sw:
                words.append((w.start, w.end, w.word.strip()))
        return words, segment_ends

    def _chunk_at(self, cut: float) -> None:
        """Drop audio before absolute time ``cut`` and advance the buffer clock."""
        keep_from = int(max(0.0, cut - self.buffer_time_offset) * self.sr)
        if keep_from <= 0:
            return
        self.audio = self.audio[keep_from:]
        self.buffer_time_offset = cut
        self.hyp.pop_committed(cut)

    def _chunk_completed_segment(self, segment_ends: list) -> None:
        """Trim at the end of the second-to-last Whisper segment (whisper_streaming's
        ``chunk_completed_segment``). Cutting at the LAST segment end would leave the
        decoder with no complete unit of left context; cutting on any committed word
        (the ``"word"`` policy) cuts mid-sentence and costs accuracy at the seam. The
        candidate must also lie inside the committed region, else there is nothing
        stable to cut at and we keep the whole buffer."""
        if not self.committed:
            return
        ends = list(segment_ends)
        if len(ends) <= 1:
            return
        t = self.committed[-1][1]
        e = ends[-2] + self.buffer_time_offset
        while len(ends) > 2 and e > t:
            ends.pop(-1)
            e = ends[-2] + self.buffer_time_offset
        if e <= t:
            self._chunk_at(e)

    def process_iter(self) -> "tuple[str, str]":
        """Decode current buffer, commit the agreed prefix, trim if grown.
        Returns (newly_committed_text, volatile_tail_text)."""
        prompt = self._prompt()
        words, segment_ends = self._transcribe(prompt)
        self.hyp.insert(words, self.buffer_time_offset)
        committed = self.hyp.flush()
        self.committed.extend(committed)
        # Trim the audio buffer once it is long and we have committed something.
        if len(self.audio) / self.sr > self.trim_seconds:
            if self.trim_policy == "segment":
                self._chunk_completed_segment(segment_ends)
            elif committed:
                self._chunk_at(committed[-1][1])  # last committed word end (absolute)
        new_text = " ".join(w[2] for w in committed).strip()
        volatile = " ".join(w[2] for w in self.hyp.complete()).strip()
        return new_text, volatile

    def finish(self) -> str:
        """Flush the remaining hypothesis at end-of-utterance."""
        rest = self.hyp.complete()
        self.committed.extend(rest)
        return " ".join(w[2] for w in rest).strip()

    def full_text(self) -> str:
        return " ".join(w[2] for w in self.committed).strip()


class VACOnlineASR:
    """VAD-gated (Voice Activity Controlled) wrapper around :class:`OnlineASR`.

    Ported in spirit from whisper_streaming's ``VACOnlineASRProcessor`` (MIT) but
    built on OUR Silero VAD (``edge_frontend.get_vad`` → sherpa-onnx). ``OnlineASR``
    re-transcribes the whole growing buffer on every ``process_iter`` — wasteful when
    the incoming audio is silence. This layer runs the CHEAP VAD over freshly inserted
    audio and only forwards it into the (expensive) Whisper decode when enough NEW
    VOICED audio has accumulated. When it sees ``vad_silence_seconds`` of trailing
    silence it forces an end-of-utterance ``finish()`` and resets for the next one.

    Same public surface as ``OnlineASR`` (``insert_audio`` / ``process_iter`` /
    ``finish`` / ``full_text``) so it is drop-in for the streaming loop. It composes
    (does not subclass) ``OnlineASR`` and never touches its constructor signature, so
    existing ``OnlineASR`` callers are unaffected.

    Parameters
    ----------
    min_voiced_seconds : forward to Whisper only once this much new *voiced* audio has
        arrived (default 0.6 s). Larger → fewer, cheaper decodes, higher latency.
    vad_silence_seconds : trailing silence that triggers end-of-utterance (default 0.5 s).
    vad : an optional pre-built SileroVAD (else lazily obtained from ``get_vad()``).
    **online_kwargs : passed straight through to the inner ``OnlineASR``.
    """

    def __init__(self, model, language: str = "vi",
                 min_voiced_seconds: float = 0.6, vad_silence_seconds: float = 0.5,
                 keep_silence_seconds: float = 1.0, vad=None, **online_kwargs):
        self.online = OnlineASR(model, language=language, **online_kwargs)
        self.sr = self.online.sr
        self.min_voiced = min_voiced_seconds
        self.silence_needed = vad_silence_seconds
        self.keep_silence = keep_silence_seconds
        self._vad = vad
        self.pending = np.zeros(0, dtype=np.float32)
        self.finals: list = []   # committed text of completed utterances
        self._live = False       # inner currently holds an in-progress utterance
        self.decodes = 0         # count of underlying Whisper decodes (for benchmarking)
        self.stream_time = 0.0   # absolute time of pending[0] since stream start

    def _get_vad(self):
        if self._vad is None:
            from edge_frontend import get_vad
            self._vad = get_vad()
        return self._vad

    def insert_audio(self, chunk: np.ndarray) -> None:
        self.pending = np.concatenate([self.pending, chunk.astype(np.float32)])

    def _volatile(self) -> str:
        return " ".join(w[2] for w in self.online.hyp.complete()).strip()

    def _forward_pending(self, segs: list) -> None:
        """Hand ``pending`` to the inner decoder and clear it.

        On a speech-START event the leading silence before the first VAD segment is
        DISCARDED rather than decoded: it is pure decode cost and it is the classic
        Whisper hallucination trigger. The inner decoder is re-based to the onset's
        absolute time so committed word timestamps remain on the stream clock across
        the discarded gap (whisper_streaming does the same in its VAC ``start`` branch)."""
        audio = self.pending
        consumed = len(self.pending) / self.sr
        if not self._live and segs:
            skip = max(0, int(segs[0][0] * self.sr))
            if skip:
                audio = audio[skip:]
            self.online.init(offset=self.stream_time + skip / self.sr)
        self.online.insert_audio(audio)
        self.pending = np.zeros(0, dtype=np.float32)
        self.stream_time += consumed

    def _flush_utterance(self) -> str:
        """End the current utterance: flush the inner buffer, archive its text, reset."""
        self.online.finish()
        text = self.online.full_text()
        if text:
            self.finals.append(text)
        self.online.init()  # reset inner state, keep the loaded model
        self._live = False
        return text

    def process_iter(self) -> "tuple[str, str]":
        """VAD-gated step. Returns (newly_committed_or_final_text, volatile_tail).

        Only calls the underlying Whisper decode when enough new voiced audio is
        present; on trailing silence it finalizes the utterance instead."""
        vad = self._get_vad()
        total = len(self.pending) / self.sr
        segs = vad.segments(self.pending, self.sr) if len(self.pending) else []
        voiced = sum(e - s for s, e in segs)
        last_end = max((e for s, e in segs), default=0.0)
        trailing_sil = total - last_end  # if no segments, == total

        # (1) End of utterance: sustained trailing silence while an utterance is active.
        if (self._live or voiced > 0) and trailing_sil >= self.silence_needed:
            if voiced > 0:
                self._forward_pending(segs)
                self.online.process_iter()
                self.decodes += 1
                self._live = True
            else:
                self.stream_time += len(self.pending) / self.sr
                self.pending = np.zeros(0, dtype=np.float32)
            final = self._flush_utterance()
            return final, ""

        # (2) Enough new voiced audio: forward it and run one Whisper decode.
        if voiced >= self.min_voiced:
            self._forward_pending(segs)
            new, vol = self.online.process_iter()
            self.decodes += 1
            self._live = True
            return new, vol

        # (3) Not enough voiced audio yet → SKIP the expensive decode. Bound the
        # pending buffer while idle so long silences don't grow it without limit
        # (keep a tail in case speech starts inside it), like whisper_streaming.
        if not self._live and total > self.keep_silence:
            keep = int(self.keep_silence * self.sr)
            self.stream_time += (len(self.pending) - keep) / self.sr
            self.pending = self.pending[-keep:]
        return "", self._volatile()

    def finish(self) -> str:
        """Flush any buffered audio and the in-progress utterance at end-of-stream."""
        if len(self.pending):
            segs = self._get_vad().segments(self.pending, self.sr)
            if sum(e - s for s, e in segs) > 0:
                self._forward_pending(segs)
                self.online.process_iter()
                self.decodes += 1
                self._live = True
            else:
                self.stream_time += len(self.pending) / self.sr
                self.pending = np.zeros(0, dtype=np.float32)
        if self._live:
            return self._flush_utterance()
        return ""

    def full_text(self) -> str:
        parts = list(self.finals)
        cur = self.online.full_text()
        if cur:
            parts.append(cur)
        return " ".join(p for p in parts if p).strip()


def stream_wav(model, audio: np.ndarray, sr: int, language: str = "vi",
               chunk_seconds: float = 1.0):
    """Generator: feed a whole utterance in fixed chunks, yield (event, text) as
    LocalAgreement commits — simulates live streaming for testing/benchmarking.
    Yields ('partial', committed_so_far, volatile) then ('final', full_text, '')."""
    if sr != 16_000:
        n = int(round(len(audio) * 16_000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
    oa = OnlineASR(model, language=language)
    step = int(16_000 * chunk_seconds)
    for i in range(0, len(audio), step):
        oa.insert_audio(audio[i:i + step])
        new, vol = oa.process_iter()
        yield ("partial", oa.full_text(), vol)
    oa.finish()
    yield ("final", oa.full_text(), "")


def _resample_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16_000:
        return audio.astype(np.float32)
    n = int(round(len(audio) * 16_000 / sr))
    return np.interp(np.linspace(0, len(audio), n, endpoint=False),
                     np.arange(len(audio)), audio).astype(np.float32)


def stream_wav_adaptive(model, audio: np.ndarray, sr: int, language: str = "vi",
                        min_chunk: float = 0.5, max_chunk: float = 5.0):
    """Like :func:`stream_wav` but with SELF-ADAPTIVE chunk pacing (whisper_streaming's
    online loop). Instead of a fixed step, the next chunk absorbs all audio that would
    have "arrived" while the previous ``process_iter`` was decoding: the step is set to
    the last decode's wall-time, clamped to ``[min_chunk, max_chunk]``. Under CPU load
    the effective chunk grows, so the decoder never falls behind real time — at the cost
    of coarser partials. On a fast machine it stays near ``min_chunk``.

    ``stream_wav``'s signature/behaviour is untouched; this is a separate entry point."""
    audio = _resample_16k(audio, sr)
    oa = OnlineASR(model, language=language)
    i, n = 0, len(audio)
    step_sec = min_chunk
    while i < n:
        step = max(1, int(16_000 * step_sec))
        oa.insert_audio(audio[i:i + step])
        i += step
        t0 = time.perf_counter()
        _new, vol = oa.process_iter()
        dt = time.perf_counter() - t0
        step_sec = min(max_chunk, max(min_chunk, dt))
        yield ("partial", oa.full_text(), vol)
    oa.finish()
    yield ("final", oa.full_text(), "")


if __name__ == "__main__":
    import argparse, sys, time
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    import soundfile as sf
    from faster_whisper import WhisperModel
    from pipeline_fast import FastPipelineConfig

    ap = argparse.ArgumentParser(description="Streaming ASR (LocalAgreement-2) demo")
    ap.add_argument("wav")
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--chunk", type=float, default=1.0)
    ap.add_argument("--vac", action="store_true",
                    help="also run the VAD-gated VACOnlineASR path and compare decode counts")
    args = ap.parse_args()

    cfg = FastPipelineConfig()
    model = WhisperModel(cfg.asr_ct2_dir, device="cpu", compute_type=cfg.asr_compute_type)
    audio, sr = sf.read(args.wav, dtype="float32", always_2d=False)
    if audio.ndim > 1: audio = audio[:, 0]

    t0 = time.perf_counter()
    prev = ""
    fixed_decodes = 0
    for ev, committed, vol in stream_wav(model, audio, sr, args.lang, args.chunk):
        dt = (time.perf_counter() - t0) * 1000
        if ev == "partial":
            fixed_decodes += 1
            add = committed[len(prev):].strip()
            print(f"[{dt:6.0f}ms] committed+= '{add}'   volatile='{vol}'")
            prev = committed
        else:
            print(f"[{dt:6.0f}ms] FINAL: {committed}")

    if args.vac:
        print("\n=== VAC (VAD-gated) path ===")
        a16 = _resample_16k(audio, sr)
        vac = VACOnlineASR(model, language=args.lang)
        feed = int(16_000 * 0.2)  # feed fine 0.2s chunks so the VAD can gate/finalize
        t1 = time.perf_counter()
        for i in range(0, len(a16), feed):
            vac.insert_audio(a16[i:i + feed])
            new, vol = vac.process_iter()
            if new:
                dt = (time.perf_counter() - t1) * 1000
                print(f"[{dt:6.0f}ms] committed+= '{new}'   volatile='{vol}'")
        tail = vac.finish()
        dt = (time.perf_counter() - t1) * 1000
        print(f"[{dt:6.0f}ms] FINAL: {vac.full_text()}")
        n_feeds = (len(a16) + feed - 1) // feed
        print(f"\nDecode-count comparison on this clip:")
        print(f"  fixed stream_wav (chunk={args.chunk}s): {fixed_decodes} Whisper decodes")
        print(f"  VAC ({n_feeds} feeds of 0.2s, VAD-gated):  {vac.decodes} Whisper decodes")
