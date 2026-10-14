"""pipeline_guard.py — make the ONE resident pipeline safe to call from the
HTTP threads, the keepalive thread and the mic loop at the same time.

The problem, stated precisely
-----------------------------
`prototype/pipeline_fast.FastCascadeS2ST` contains no locks. It is written for
one caller. The supervisor gives it four:

  * every HTTP worker thread (`ThreadingHTTPServer`, one thread per request)
  * the keepalive probe thread
  * the mic loop thread (`--mic`)
  * threads the mic loop itself spawns — `live_translate._translate_pipelined`
    runs MT on a "mt-pipeline" worker while `streaming_tts.stream_threaded`
    synthesizes on another.

What breaks when two of them overlap is not theoretical:

  * `translate()` does `self._mt_tok.src_lang = src_code` on the SHARED
    tokenizer and then encodes. Two turns with different source languages
    interleave into a translation from the wrong language.
  * `synthesize()` mutates `_tts_audio_cache` and `_tts_audio_cache_order` as a
    hand-rolled LRU. Concurrent `list.remove` / `pop(0)` on that pair raises
    ValueError or evicts a key that is still in the dict.
  * `faster_whisper.WhisperModel` is documented as NOT thread-safe; its own
    README tells you to use one model per thread. Two concurrent decodes on one
    model is a native-code crash, not an exception.
  * `ctranslate2.Translator` IS internally thread-safe, but only the translator —
    the tokenizer around it is not, and that is where our state lives.

So: a judge curling `/translate` while the mic is live is a real corruption
path, and the runbook's claim that there is "one inference at a time by design"
was only true for the HTTP paths.

The design, and what it costs
-----------------------------
PER-STAGE locks, not one global lock:

    asr  ──  corrector  ──  mt  ──  tts        (that is also the lock ORDER)

A single global lock would be simpler, and would also destroy the thing
`prototype/EDGE_UPGRADES.md` measured: streaming TTS speaks sentence *i* while
MT is still decoding sentence *i+1* (first audio 98 ms vs 876 ms, 8.9x). Those
are two different models, so they may run at once — they just may not run twice
each at once. Per-stage locks keep that overlap and still forbid the unsafe
case.

What serialization costs, concretely:

  * Two callers of the SAME stage queue. On the reference box a light probe is
    MT ~0.5-3.6 s + TTS ~0.6-2.6 s, so a `/translate` arriving mid-probe waits
    up to about that long. Under the demo pattern (one speaker, one curl) the
    measured wait is zero; the guard counts every wait so `/healthz` can prove
    it rather than assume it.
  * Whole-cascade entry points (`run_streaming`, `run`) take ALL four locks for
    their duration, because their internals call `self._asr` / `self._mt`
    directly and a proxy cannot see inside them. While such a call is in flight
    the device is genuinely single-threaded. That is the batch mic path; the
    streaming mic path (the default) does not use it.
  * Nothing waits forever. Every acquire has a timeout and raises `Busy`, which
    the supervisor turns into HTTP 503 with `busy: true`. A deadlocked demo is
    worse than a refused request.

Coverage note (honest): the guard protects calls made THROUGH it. Code that
grabs `pipe._mt` or `pipe._piper` and calls it directly bypasses the guard.
There is exactly one such caller in the tree — `live_translate._make_online_asr`
reads `pipe._asr` and hands it to `streaming_asr.OnlineASR` — and the guard
handles it specially by returning a locked view of the model rather than the
model itself (see `_GuardedAsrModel`).
"""

from __future__ import annotations

import contextlib
import threading
import time

STAGES = ("asr", "corrector", "mt", "tts")
# Fixed global order. Every multi-lock acquisition follows it, so two callers
# can never hold locks in an order that deadlocks.
_ORDER = {name: i for i, name in enumerate(STAGES)}

DEFAULT_TIMEOUT_S = 90.0


class Busy(RuntimeError):
    """A stage was held by another caller for longer than the timeout."""

    def __init__(self, stage: str, waited_s: float, holder: str | None = None):
        self.stage = stage
        self.waited_s = waited_s
        self.holder = holder
        super().__init__(
            f"stage {stage!r} busy: waited {waited_s:.1f}s"
            + (f", held by thread {holder!r}" if holder else ""))


class GuardedPipeline:
    """Serializing proxy over a FastCascadeS2ST-compatible object."""

    # method name -> stages it must hold, in _ORDER order
    GUARDED = {
        "transcribe_raw": ("asr",),
        "transcribe": ("asr", "corrector"),
        "transcribe_detailed": ("asr", "corrector"),
        "finalize_transcript": ("corrector",),
        "translate": ("mt",),
        "translate_sentences_batch": ("mt",),
        "synthesize": ("tts",),
        "_ensure_asr": ("asr",),
        "_ensure_corrector": ("corrector",),
        "_ensure_mt": ("mt",),
        "_ensure_piper": ("tts",),
        "_ensure_tts": ("tts",),
    }
    # Whole-cascade entry points: internals bypass the proxy, so hold everything.
    GUARDED_ALL = ("run", "run_streaming")
    GENERATORS = ("run_streaming",)

    def __init__(self, pipe, timeout_s: float = DEFAULT_TIMEOUT_S, logger=None):
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "_timeout_s", float(timeout_s))
        object.__setattr__(self, "_logger", logger)
        object.__setattr__(self, "_locks", {s: threading.RLock() for s in STAGES})
        object.__setattr__(self, "_holder", {s: None for s in STAGES})
        object.__setattr__(self, "_stats_lock", threading.Lock())
        object.__setattr__(self, "_stats", {
            s: {"calls": 0, "waits": 0, "wait_ms_total": 0.0,
                "max_wait_ms": 0.0, "timeouts": 0} for s in STAGES})
        object.__setattr__(self, "_asr_view", None)
        object.__setattr__(self, "_tls", threading.local())

    @contextlib.contextmanager
    def deadline(self, timeout_s: float):
        """Override the wait budget for THIS thread only, for the duration of the
        block. The keepalive probe uses a short budget so it gives way to live
        speech instead of queueing behind it; a judge's /translate uses the long
        one because they are waiting at the device."""
        tls = object.__getattribute__(self, "_tls")
        prev = getattr(tls, "timeout_s", None)
        tls.timeout_s = float(timeout_s)
        try:
            yield self
        finally:
            tls.timeout_s = prev

    # -- introspection ------------------------------------------------------
    @property
    def raw(self):
        """The unguarded pipeline. Only for code that knows what it is doing
        (startup, before any other thread exists)."""
        return object.__getattribute__(self, "_pipe")

    def stats(self) -> dict:
        with object.__getattribute__(self, "_stats_lock"):
            return {s: dict(v) for s, v in
                    object.__getattribute__(self, "_stats").items()}

    def contended(self) -> bool:
        return any(v["waits"] for v in self.stats().values())

    def holders(self) -> dict:
        return dict(object.__getattribute__(self, "_holder"))

    # -- locking ------------------------------------------------------------
    def _acquire(self, stages, timeout_s: float | None = None):
        """Acquire `stages` in global order. All-or-nothing: on timeout the
        already-held locks are released before Busy is raised."""
        if timeout_s is None:
            tls = object.__getattribute__(self, "_tls")
            timeout_s = getattr(tls, "timeout_s", None)
            if timeout_s is None:
                timeout_s = object.__getattribute__(self, "_timeout_s")
        locks = object.__getattribute__(self, "_locks")
        holder = object.__getattribute__(self, "_holder")
        me = threading.current_thread().name
        ordered = sorted(stages, key=lambda s: _ORDER[s])
        held: list[str] = []
        deadline = time.monotonic() + timeout_s
        for s in ordered:
            t0 = time.monotonic()
            remaining = max(deadline - t0, 0.0)
            got = locks[s].acquire(timeout=remaining) if remaining > 0 \
                else locks[s].acquire(blocking=False)
            waited = time.monotonic() - t0
            if not got:
                for h in reversed(held):
                    holder[h] = None
                    locks[h].release()
                self._note(s, waited, timeout=True)
                log = object.__getattribute__(self, "_logger")
                if log is not None:
                    import devlog
                    log.error(devlog.E_GUARD_TIMEOUT, stage=s,
                              waited_s=round(waited, 2), holder=holder.get(s))
                raise Busy(s, waited, holder.get(s))
            held.append(s)
            holder[s] = me
            self._note(s, waited, timeout=False)
        return held

    def _release(self, held) -> None:
        locks = object.__getattribute__(self, "_locks")
        holder = object.__getattribute__(self, "_holder")
        for s in reversed(held):
            holder[s] = None
            locks[s].release()

    def _note(self, stage: str, waited_s: float, timeout: bool) -> None:
        ms = waited_s * 1000.0
        with object.__getattribute__(self, "_stats_lock"):
            st = object.__getattribute__(self, "_stats")[stage]
            if timeout:
                st["timeouts"] += 1
                return
            st["calls"] += 1
            # 5 ms is well above uncontended RLock acquisition and well below any
            # real inference, so it separates "queued" from "took the lock".
            if ms > 5.0:
                st["waits"] += 1
                st["wait_ms_total"] = round(st["wait_ms_total"] + ms, 1)
                st["max_wait_ms"] = round(max(st["max_wait_ms"], ms), 1)
        if ms > 5.0:
            log = object.__getattribute__(self, "_logger")
            if log is not None:
                import devlog
                log.info(devlog.E_GUARD_WAIT, stage=stage, waited_ms=round(ms, 1))

    # -- proxying -----------------------------------------------------------
    def __getattr__(self, name):
        pipe = object.__getattribute__(self, "_pipe")
        if name == "_asr":
            return self._guarded_asr()
        attr = getattr(pipe, name)
        if not callable(attr):
            return attr
        stages = self.GUARDED.get(name)
        if stages is None and name in self.GUARDED_ALL:
            stages = STAGES
        if stages is None:
            return attr
        if name in self.GENERATORS:
            return self._wrap_generator(attr, stages)
        return self._wrap_call(attr, stages)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_pipe"), name, value)

    def _wrap_call(self, fn, stages):
        def call(*a, **kw):
            held = self._acquire(stages)
            try:
                return fn(*a, **kw)
            finally:
                self._release(held)
        call.__name__ = getattr(fn, "__name__", "guarded")
        return call

    def _wrap_generator(self, fn, stages):
        """Hold the locks for the LIFETIME of the generator, not just the call.
        `run_streaming` yields progressively while the models keep working; a
        lock released at first yield would guard nothing."""
        def call(*a, **kw):
            held = self._acquire(stages)
            try:
                yield from fn(*a, **kw)
            finally:
                self._release(held)
        call.__name__ = getattr(fn, "__name__", "guarded_gen")
        return call

    def _guarded_asr(self):
        view = object.__getattribute__(self, "_asr_view")
        pipe = object.__getattribute__(self, "_pipe")
        model = getattr(pipe, "_asr", None)
        if model is None:
            return None
        if view is None or view.model is not model:
            view = _GuardedAsrModel(model, self)
            object.__setattr__(self, "_asr_view", view)
        return view


class _GuardedAsrModel:
    """A faster-whisper WhisperModel that may only be decoded by one thread.

    `WhisperModel.transcribe()` returns a LAZY generator: the decode happens
    while the caller iterates, not during the call. Locking only the call would
    therefore protect nothing. This holds the ASR lock from the call until the
    generator is exhausted or closed.

    Every consumer in this tree (`streaming_asr.OnlineASR._transcribe`,
    `pipeline_fast._asr_prewarm`) drains the generator inside one function, so
    the lock is held for one decode and no longer.
    """

    def __init__(self, model, guard: GuardedPipeline):
        self.model = model
        self._guard = guard

    def __getattr__(self, name):
        return getattr(self.model, name)

    def transcribe(self, *a, **kw):
        held = self._guard._acquire(("asr",))
        released = False
        try:
            segments, info = self.model.transcribe(*a, **kw)
        except BaseException:
            self._guard._release(held)
            raise

        def drain():
            nonlocal released
            try:
                for seg in segments:
                    yield seg
            finally:
                if not released:
                    released = True
                    self._guard._release(held)

        return drain(), info
