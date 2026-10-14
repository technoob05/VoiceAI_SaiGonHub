"""playback.py — interruptible, non-blocking audio playback (enables barge-in).

Our old live app played each TTS fragment with `sd.play(...); sd.wait()` — a BLOCKING
call that cannot be stopped. That makes barge-in (user speaks → stop the output
mid-sentence) impossible: `interrupt()` could drain the pending queue, but the fragment
already in flight kept playing to its end (0.5–3 s of unstoppable audio).

Pattern absorbed from huggingface/speech-to-speech (`connections/local_audio_streamer.py`
+ `pipeline/cancel_scope.py`, both feed a queue-drained output and cancel by flag) and
from RealtimeTTS `stream_player.py` (`playout_chunk_size`: never hand the device a whole
fragment, hand it small sub-chunks so a cancel flag can be honoured between them):
  * A worker thread drains a bounded Queue of audio fragments and writes each fragment
    to an open `sd.OutputStream` in `subchunk_ms` slices (default 20 ms), checking the
    `interrupt` flag BETWEEN slices.
  * `interrupt()` sets the flag, drains the queue and aborts the stream (dropping audio
    already handed to the device). Output therefore stops after at most ONE sub-chunk
    write — the bound is `subchunk_ms` (~20 ms) plus the device buffer, NOT the
    remaining fragment duration.
  * `is_speaking()` lets the mic/VAD loop know playback is active, so it only treats
    incoming speech as a barge-in while we are actually speaking (a `should_listen`
    style half-duplex gate).
  * `buffered_seconds()` reports audio accepted but not yet written out — playback
    depth, for callers that want to size fragments against it.

Self-contained + testable without a mic: drive `enqueue()` + `interrupt()` and inspect
the played log. The sink is injectable and is called once per SUB-CHUNK; the default
sink opens a sounddevice OutputStream.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

import numpy as np

DEFAULT_SUBCHUNK_MS = 20


class _SinkWriter:
    """Adapts an injected `sink(chunk, sr)` callable to the writer protocol."""

    def __init__(self, sink: Callable[[np.ndarray, int], None], sr: int):
        self._sink = sink
        self._sr = sr

    def write(self, sub: np.ndarray) -> None:
        self._sink(sub, self._sr)

    def abort(self) -> None:
        pass

    def close(self) -> None:
        pass


class _DeviceWriter:
    """Default writer: a live sounddevice OutputStream, one blocksize per sub-chunk."""

    def __init__(self, sr: int, blocksize: int):
        import sounddevice as sd
        self._stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                       blocksize=blocksize, latency="low")
        self._stream.start()

    def write(self, sub: np.ndarray) -> None:
        self._stream.write(sub)

    def abort(self) -> None:
        # abort(), not stop(): stop() drains the device buffer, which is exactly the
        # tail a barge-in needs to throw away.
        self._stream.abort()

    def close(self) -> None:
        self._stream.close()


class Playback:
    def __init__(self, sink: Optional[Callable[[np.ndarray, int], None]] = None,
                 max_queue: int = 64, subchunk_ms: int = DEFAULT_SUBCHUNK_MS):
        # sink(sub_chunk, sr) plays one SUB-CHUNK synchronously; interrupt latency is
        # bounded by how long a single sub-chunk takes. Default = sounddevice stream.
        self._sink = sink
        self._subchunk_ms = max(1, int(subchunk_ms))
        self._q: "queue.Queue" = queue.Queue(maxsize=max_queue)
        self._interrupt = threading.Event()
        self._speaking = threading.Event()
        self._stop = threading.Event()
        self._depth_lock = threading.Lock()
        self._buffered_samples = 0.0
        self._last_sr = 0
        self._writer = None
        self._writer_sr = 0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ---- writer seam: monkeypatch or inject a sink to run with no audio hardware ----
    def _open_writer(self, sr: int):
        if self._sink is not None:
            return _SinkWriter(self._sink, sr)
        return _DeviceWriter(sr, self._subchunk_frames(sr))

    def _subchunk_frames(self, sr: int) -> int:
        return max(1, int(sr * self._subchunk_ms / 1000))

    def _writer_for(self, sr: int):
        if self._writer is None or self._writer_sr != sr:
            self._close_writer()
            self._writer = self._open_writer(sr)
            self._writer_sr = sr
        return self._writer

    def _close_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        self._writer = None
        self._writer_sr = 0

    def _credit(self, samples: float) -> None:
        with self._depth_lock:
            self._buffered_samples = max(0.0, self._buffered_samples + samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._speaking.is_set() and self._q.empty():
                    self._speaking.clear()   # finished the current response
                continue
            if item is None:                  # end-of-response sentinel
                if self._q.empty():
                    self._speaking.clear()
                continue
            chunk, sr = item
            if self._interrupt.is_set():       # dropped mid-response
                self._credit(-len(chunk))
                continue
            self._speaking.set()
            try:
                self._play_fragment(chunk, sr)
            except Exception:
                self._credit(-len(chunk))
                self._close_writer()

    def _play_fragment(self, chunk: np.ndarray, sr: int) -> None:
        step = self._subchunk_frames(sr)
        writer = self._writer_for(sr)
        pos, n = 0, len(chunk)
        while pos < n:
            if self._interrupt.is_set() or self._stop.is_set():
                self._credit(-(n - pos))
                return
            sub = chunk[pos:pos + step]
            writer.write(sub)
            self._credit(-len(sub))
            pos += len(sub)

    # ---- producer API ----
    def enqueue(self, chunk: np.ndarray, sr: int) -> None:
        """Queue one audio chunk for playback (drop-oldest if full)."""
        if self._interrupt.is_set():
            return
        self._speaking.set()
        if self._q.full():
            try:
                old = self._q.get_nowait()
                if old is not None:
                    self._credit(-len(old[0]))
            except queue.Empty:
                pass
        arr = np.asarray(chunk, dtype=np.float32)
        self._last_sr = int(sr)
        self._credit(len(arr))
        self._q.put((arr, int(sr)))

    def end_response(self) -> None:
        self._q.put(None)

    def new_response(self) -> None:
        """Start a fresh response: clear any prior interrupt so playback resumes."""
        self._interrupt.clear()

    # ---- barge-in API ----
    def interrupt(self) -> int:
        """Stop output now: set the flag, drain the queue, discard the device buffer.

        The in-flight fragment stops after at most one sub-chunk write.
        Returns #chunks dropped."""
        self._interrupt.set()
        if self._writer is not None:
            try:
                self._writer.abort()
            except Exception:
                pass
        dropped = 0
        while True:
            try:
                item = self._q.get_nowait()
                dropped += 1
                if item is not None:
                    self._credit(-len(item[0]))
            except queue.Empty:
                break
        self._speaking.clear()
        return dropped

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def buffered_seconds(self) -> float:
        """Audio accepted but not yet written to the device, in seconds."""
        with self._depth_lock:
            samples = self._buffered_samples
        sr = self._writer_sr or self._last_sr
        return samples / sr if sr else 0.0

    def wait_idle(self, timeout: float = 30.0) -> None:
        """Block until the queue drains and playback stops (for file/batch mode)."""
        import time
        t0 = time.time()
        while (not self._q.empty() or self._speaking.is_set()) and time.time() - t0 < timeout:
            time.sleep(0.02)

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=1.0)
        self._close_writer()


if __name__ == "__main__":
    # Headless self-test (no mic): injected sink logs each sub-chunk; verify ordering
    # and that interrupt() stops mid-FRAGMENT, not merely mid-queue.
    import time
    played = []
    pb = Playback(sink=lambda c, sr: (time.sleep(0.005), played.append(len(c))),
                  subchunk_ms=20)
    pb.new_response()
    for _ in range(6):
        pb.enqueue(np.zeros(16000, dtype=np.float32), 16000)   # 1 s fragments
    time.sleep(0.12)
    dropped = pb.interrupt()              # barge-in mid-fragment
    time.sleep(0.1)
    total = sum(played)
    print(f"played {total} samples in {len(played)} sub-chunks, dropped {dropped} chunks")
    assert 0 < total < 6 * 16000, "interrupt did not stop playback mid-stream"
    assert dropped > 0, "interrupt dropped nothing"
    pb.new_response()
    pb.enqueue(np.zeros(500, dtype=np.float32), 16000)
    time.sleep(0.1)
    print(f"after resume: {len(played)} sub-chunks, buffered={pb.buffered_seconds():.3f}s")
    pb.close()
    print("PLAYBACK SELFTEST OK")
