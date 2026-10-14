"""Headless tests for playback.Playback — no audio device required.

The sink seam is exercised directly: every test injects a fake sink that records the
sub-chunks it was handed, so the interrupt bound is measured in sub-chunk writes rather
than wall-clock against real hardware.
"""

import threading
import time

import numpy as np
import pytest

from playback import Playback

SR = 16000


class FakeSink:
    """Records every sub-chunk written; optionally blocks so a barge-in can race it."""

    def __init__(self, per_write_delay: float = 0.0):
        self.writes = []
        self.delay = per_write_delay
        self._lock = threading.Lock()

    def __call__(self, sub, sr):
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.writes.append(np.asarray(sub).copy())

    @property
    def samples(self):
        with self._lock:
            return int(sum(len(w) for w in self.writes))

    def concat(self):
        with self._lock:
            return np.concatenate(self.writes) if self.writes else np.zeros(0, np.float32)

    def n_writes(self):
        with self._lock:
            return len(self.writes)


@pytest.fixture
def pb_factory():
    made = []

    def _make(sink, **kw):
        pb = Playback(sink=sink, **kw)
        made.append(pb)
        return pb

    yield _make
    for pb in made:
        pb.close()


def _wait(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_normal_playback_writes_every_sample_in_order(pb_factory):
    sink = FakeSink()
    pb = pb_factory(sink, subchunk_ms=20)
    pb.new_response()
    ramp = np.arange(5000, dtype=np.float32)
    tail = np.arange(5000, 7777, dtype=np.float32)
    pb.enqueue(ramp, SR)
    pb.enqueue(tail, SR)
    pb.end_response()
    pb.wait_idle(timeout=5)
    assert _wait(lambda: sink.samples == len(ramp) + len(tail))
    np.testing.assert_array_equal(sink.concat(), np.concatenate([ramp, tail]))


def test_subchunk_size_is_bounded_by_subchunk_ms(pb_factory):
    sink = FakeSink()
    pb = pb_factory(sink, subchunk_ms=20)
    pb.new_response()
    pb.enqueue(np.zeros(SR, dtype=np.float32), SR)   # 1 s
    assert _wait(lambda: sink.samples == SR)
    expected = int(SR * 20 / 1000)
    assert max(len(w) for w in sink.writes) == expected
    assert sink.n_writes() == SR // expected


def test_interrupt_stops_within_about_one_subchunk(pb_factory):
    # 20 ms sub-chunks, each fake write costs 10 ms => the worker cannot get more than
    # one extra sub-chunk out after the flag is set.
    sink = FakeSink(per_write_delay=0.010)
    pb = pb_factory(sink, subchunk_ms=20)
    pb.new_response()
    pb.enqueue(np.ones(SR * 3, dtype=np.float32), SR)   # 3 s fragment
    assert _wait(lambda: sink.n_writes() >= 3)
    at_interrupt = sink.n_writes()
    pb.interrupt()
    time.sleep(0.2)
    extra = sink.n_writes() - at_interrupt
    assert extra <= 1, f"interrupt leaked {extra} sub-chunks"
    assert sink.samples < SR * 3, "whole fragment played despite interrupt"


def test_interrupt_drains_the_queue_and_stops_speaking(pb_factory):
    sink = FakeSink(per_write_delay=0.010)
    pb = pb_factory(sink, subchunk_ms=20)
    pb.new_response()
    for _ in range(5):
        pb.enqueue(np.zeros(SR, dtype=np.float32), SR)
    assert _wait(lambda: sink.n_writes() >= 2)
    dropped = pb.interrupt()
    assert dropped > 0
    assert pb._q.empty()
    assert not pb.is_speaking()
    played_after = sink.samples
    time.sleep(0.2)
    assert sink.samples - played_after <= int(SR * 20 / 1000)


def test_enqueue_after_interrupt_is_ignored_until_new_response(pb_factory):
    sink = FakeSink()
    pb = pb_factory(sink, subchunk_ms=20)
    pb.new_response()
    pb.interrupt()
    pb.enqueue(np.ones(1000, dtype=np.float32), SR)
    time.sleep(0.15)
    assert sink.samples == 0
    pb.new_response()
    pb.enqueue(np.ones(1000, dtype=np.float32), SR)
    assert _wait(lambda: sink.samples == 1000)


def test_bounded_queue_drops_oldest(pb_factory):
    sink = FakeSink(per_write_delay=0.050)   # stall the worker so the queue fills
    pb = pb_factory(sink, max_queue=3, subchunk_ms=20)
    pb.new_response()
    for i in range(10):
        pb.enqueue(np.full(100, float(i), dtype=np.float32), SR)
    assert pb._q.qsize() <= 3
    remaining = [item[0][0] for item in list(pb._q.queue) if item is not None]
    assert remaining == sorted(remaining)
    assert remaining[-1] == 9.0          # newest survived, oldest were dropped


def test_buffered_seconds_tracks_pending_audio(pb_factory):
    sink = FakeSink(per_write_delay=0.050)
    pb = pb_factory(sink, subchunk_ms=20)
    pb.new_response()
    assert pb.buffered_seconds() == 0.0
    for _ in range(4):
        pb.enqueue(np.zeros(SR, dtype=np.float32), SR)   # 4 x 1 s
    assert pb.buffered_seconds() == pytest.approx(4.0, abs=0.1)
    pb.interrupt()
    # The queued fragments are credited back synchronously; the in-flight one only once
    # the worker returns from its current sub-chunk write.
    assert pb.buffered_seconds() <= 1.0
    assert _wait(lambda: pb.buffered_seconds() == pytest.approx(0.0, abs=0.05))


def test_buffered_seconds_returns_to_zero_after_full_playout(pb_factory):
    sink = FakeSink()
    pb = pb_factory(sink, subchunk_ms=20)
    pb.new_response()
    pb.enqueue(np.zeros(SR // 2, dtype=np.float32), SR)
    pb.end_response()
    pb.wait_idle(timeout=5)
    assert _wait(lambda: pb.buffered_seconds() == pytest.approx(0.0, abs=1e-6))


def test_default_writer_uses_output_stream_not_blocking_play(monkeypatch, pb_factory):
    """No sink injected -> the seam must build an OutputStream, never sd.play()."""
    import playback as pbmod

    opened = {}

    class FakeStream:
        def __init__(self, sr, blocksize):
            opened["sr"] = sr
            opened["blocksize"] = blocksize
            self.written = []
            self.aborted = False

        def write(self, sub):
            time.sleep(0.010)
            self.written.append(len(sub))

        def abort(self):
            self.aborted = True

        def close(self):
            pass

    monkeypatch.setattr(pbmod.Playback, "_open_writer",
                        lambda self, sr: FakeStream(sr, self._subchunk_frames(sr)))
    pb = pb_factory(None, subchunk_ms=20)
    pb.new_response()
    pb.enqueue(np.zeros(SR * 3, dtype=np.float32), SR)
    assert _wait(lambda: pb._writer is not None and len(pb._writer.written) >= 3)
    stream = pb._writer
    at_interrupt = len(stream.written)
    pb.interrupt()
    time.sleep(0.2)
    assert opened["blocksize"] == int(SR * 20 / 1000)
    assert stream.aborted, "device buffer was not discarded on barge-in"
    assert len(stream.written) - at_interrupt <= 1
