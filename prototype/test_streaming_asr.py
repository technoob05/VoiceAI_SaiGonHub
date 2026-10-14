"""Unit tests for streaming_asr.py — no model is loaded; the ASR and the VAD are
scripted stubs so every assertion is about OUR bookkeeping, not Whisper's output."""

from __future__ import annotations

import numpy as np
import pytest

from streaming_asr import HypothesisBuffer, OnlineASR, VACOnlineASR

SR = 16_000


class FakeWord:
    def __init__(self, start: float, end: float, word: str):
        self.start = start
        self.end = end
        self.word = word


class FakeSegment:
    def __init__(self, start: float, end: float, words: list,
                 avg_logprob: float = -0.2, no_speech_prob: float = 0.0):
        self.start = start
        self.end = end
        self.words = [FakeWord(*w) for w in words]
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class ScriptedASR:
    """Stands in for faster_whisper.WhisperModel. Returns the n-th scripted response
    on the n-th call, repeating the last one forever after."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"n_samples": len(audio), **kwargs})
        i = min(len(self.calls) - 1, len(self.responses) - 1)
        return iter(self.responses[i]), None


class ScriptedVAD:
    def __init__(self, script: list):
        self.script = list(script)
        self.calls = 0

    def segments(self, audio, sr):
        i = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[i]


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


# --- fix 1: no_speech_prob gate -------------------------------------------------

def test_high_no_speech_prob_segment_is_dropped():
    model = ScriptedASR([[
        FakeSegment(0.0, 2.0, [(0.5, 1.0, "xin")], no_speech_prob=0.02),
        FakeSegment(2.0, 4.0, [(2.5, 3.0, "HALLUCINATION")], no_speech_prob=0.97),
    ]])
    oa = OnlineASR(model)
    words, ends = oa._transcribe("")
    assert [w[2] for w in words] == ["xin"]
    assert ends == [2.0]


def test_moderate_no_speech_prob_segment_passes():
    model = ScriptedASR([[
        FakeSegment(0.0, 2.0, [(0.5, 1.0, "xin")], no_speech_prob=0.5),
    ]])
    words, _ = OnlineASR(model)._transcribe("")
    assert [w[2] for w in words] == ["xin"]


def test_no_speech_ceiling_is_configurable():
    resp = [FakeSegment(0.0, 2.0, [(0.5, 1.0, "xin")], no_speech_prob=0.5)]
    assert OnlineASR(ScriptedASR([resp]), no_speech_ceiling=0.3)._transcribe("")[0] == []
    assert OnlineASR(ScriptedASR([resp]), no_speech_ceiling=0.9)._transcribe("")[0] != []


def test_dropped_segment_is_not_a_trim_candidate():
    model = ScriptedASR([[
        FakeSegment(0.0, 2.0, [(0.5, 1.0, "a")], no_speech_prob=0.02),
        FakeSegment(2.0, 9.0, [(3.0, 3.5, "noise")], no_speech_prob=0.99),
    ]])
    _, ends = OnlineASR(model)._transcribe("")
    assert 9.0 not in ends


# --- fix 2: segment-boundary trim ----------------------------------------------

def _three_segment_response(fourth_word: str):
    return [
        FakeSegment(0.0, 2.5, [(0.5, 1.0, "a"), (1.5, 2.0, "b")]),
        FakeSegment(2.5, 5.0, [(3.0, 3.5, "c"), (4.0, 4.5, fourth_word)]),
        FakeSegment(5.0, 6.5, [(5.5, 6.0, "e")]),
    ]


def _run_two_iters(trim_policy: str, fourth_word_second_pass: str = "d"):
    model = ScriptedASR([_three_segment_response("d"),
                         _three_segment_response(fourth_word_second_pass)])
    oa = OnlineASR(model, trim_seconds=2.0, trim_policy=trim_policy)
    oa.insert_audio(silence(8.0))
    oa.process_iter()
    oa.process_iter()
    return oa


def test_segment_trim_cuts_at_second_to_last_segment_end():
    oa = _run_two_iters("segment")
    assert [w[2] for w in oa.committed] == ["a", "b", "c", "d", "e"]
    # ends = [2.5, 5.0, 6.5]; second-to-last = 5.0, inside the committed region (6.0).
    assert oa.buffer_time_offset == pytest.approx(5.0)
    assert len(oa.audio) == int(8.0 * SR) - int(5.0 * SR)


def test_segment_trim_walks_back_when_candidate_is_uncommitted():
    # Only "a b c" agree across the two passes, so committed ends at 3.5; the 5.0
    # candidate lies past it and the policy must fall back to the 2.5 boundary.
    oa = _run_two_iters("segment", fourth_word_second_pass="X")
    assert [w[2] for w in oa.committed] == ["a", "b", "c"]
    assert oa.buffer_time_offset == pytest.approx(2.5)
    assert len(oa.audio) == int(8.0 * SR) - int(2.5 * SR)


def test_word_trim_policy_is_unchanged_and_default():
    oa = _run_two_iters("word")
    assert oa.buffer_time_offset == pytest.approx(6.0)  # last committed word end
    assert len(oa.audio) == int(8.0 * SR) - int(6.0 * SR)
    assert OnlineASR(ScriptedASR([[]])).trim_policy == "word"


def test_timestamps_after_segment_trim_are_absolute():
    oa = _run_two_iters("segment")
    assert oa.buffer_time_offset == pytest.approx(5.0)
    n_before = len(oa.audio)
    oa.insert_audio(silence(1.0))
    # A word the decoder reports 1.2 s into the trimmed buffer is at 6.2 s absolute.
    oa.model.responses = [[FakeSegment(0.0, 2.0, [(1.2, 1.5, "f")])]]
    oa.model.calls = []
    oa.process_iter()
    oa.process_iter()
    a, b, t = oa.committed[-1]
    assert (round(a, 6), round(b, 6), t) == (6.2, 6.5, "f")
    assert len(oa.audio) == n_before + int(1.0 * SR)


def test_invalid_trim_policy_rejected():
    with pytest.raises(ValueError):
        OnlineASR(ScriptedASR([[]]), trim_policy="sentence")


# --- LocalAgreement-2 regression -----------------------------------------------

def test_local_agreement_commits_only_the_agreed_prefix():
    model = ScriptedASR([
        [FakeSegment(0.0, 3.0, [(0.5, 1.0, "toi"), (1.2, 1.6, "di"), (2.0, 2.4, "hoc")])],
        [FakeSegment(0.0, 3.0, [(0.5, 1.0, "toi"), (1.2, 1.6, "di"), (2.0, 2.4, "lam")])],
    ])
    oa = OnlineASR(model)
    oa.insert_audio(silence(3.0))
    first_new, first_vol = oa.process_iter()
    assert first_new == ""                     # nothing to agree with yet
    assert first_vol == "toi di hoc"
    second_new, second_vol = oa.process_iter()
    assert second_new == "toi di"              # longest common prefix only
    assert second_vol == "lam"
    assert oa.full_text() == "toi di"
    assert oa.finish() == "lam"
    assert oa.full_text() == "toi di lam"


def test_hypothesis_buffer_never_retracts():
    hyp = HypothesisBuffer()
    hyp.insert([(0.0, 0.5, "a"), (0.6, 1.0, "b")], 0.0)
    assert hyp.flush() == []
    hyp.insert([(0.0, 0.5, "a"), (0.6, 1.0, "b"), (1.2, 1.5, "c")], 0.0)
    assert [w[2] for w in hyp.flush()] == ["a", "b"]
    assert hyp.last_committed_time == pytest.approx(1.0)


# --- fix 3: VAC leading-silence slice -------------------------------------------

def _vac(model, vad_script, **kw):
    return VACOnlineASR(model, vad=ScriptedVAD(vad_script),
                        min_voiced_seconds=0.5, vad_silence_seconds=0.5, **kw)


def test_vac_drops_leading_silence_and_rebases_the_clock():
    model = ScriptedASR([[]])
    vac = _vac(model, [[(2.0, 3.0)]])
    vac.insert_audio(silence(3.0))
    vac.process_iter()
    assert len(vac.online.audio) == int(1.0 * SR)          # only the voiced tail
    assert vac.online.buffer_time_offset == pytest.approx(2.0)
    assert vac.stream_time == pytest.approx(3.0)
    assert len(vac.pending) == 0
    assert model.calls[0]["n_samples"] == int(1.0 * SR)


def test_vac_committed_timestamps_stay_absolute_across_the_slice():
    resp = [FakeSegment(0.0, 1.0, [(0.2, 0.6, "xin"), (0.7, 0.9, "chao")])]
    model = ScriptedASR([resp])
    vac = _vac(model, [[(2.0, 3.0)], [(0.0, 1.0)]])
    vac.insert_audio(silence(3.0))
    vac.process_iter()
    vac.insert_audio(silence(1.0))
    new, _vol = vac.process_iter()
    assert new == "xin chao"
    assert [(round(a, 6), round(b, 6), t) for a, b, t in vac.online.committed] == \
        [(2.2, 2.6, "xin"), (2.7, 2.9, "chao")]
    assert vac.decodes == 2
    assert vac.stream_time == pytest.approx(4.0)


def test_vac_without_leading_silence_keeps_offset_zero():
    vac = _vac(ScriptedASR([[]]), [[(0.0, 1.0)]])
    vac.insert_audio(silence(1.0))
    vac.process_iter()
    assert vac.online.buffer_time_offset == pytest.approx(0.0)
    assert len(vac.online.audio) == int(1.0 * SR)


def test_vac_idle_silence_trim_advances_the_stream_clock():
    vac = _vac(ScriptedASR([[]]), [[]], keep_silence_seconds=1.0)
    vac.insert_audio(silence(3.0))
    new, vol = vac.process_iter()
    assert (new, vol) == ("", "")
    assert len(vac.pending) == int(1.0 * SR)
    assert vac.stream_time == pytest.approx(2.0)


def test_vac_second_utterance_rebases_from_the_stream_clock():
    resp = [FakeSegment(0.0, 1.0, [(0.2, 0.6, "hai")])]
    model = ScriptedASR([resp])
    # utterance 1 ends on trailing silence, then a fresh onset 1.5 s into new audio.
    vac = _vac(model, [[(0.0, 1.0)], [(1.5, 3.0)]])
    vac.insert_audio(silence(2.0))
    vac.process_iter()                       # voiced 1.0, trailing 1.0 >= 0.5 → final
    assert vac.stream_time == pytest.approx(2.0)
    assert vac._live is False
    vac.insert_audio(silence(3.0))
    vac.process_iter()
    assert vac.online.buffer_time_offset == pytest.approx(3.5)
    assert vac.stream_time == pytest.approx(5.0)
    assert len(vac.online.audio) == int(1.5 * SR)


def test_vac_slices_onset_on_a_self_contained_utterance():
    model = ScriptedASR([[]])
    # speech 1.0-2.0 then 1.0 s of trailing silence → end-of-utterance in one step.
    vac = _vac(model, [[(1.0, 2.0)]])
    vac.insert_audio(silence(3.0))
    vac.process_iter()
    assert model.calls[0]["n_samples"] == int(2.0 * SR)   # leading silence discarded
    assert vac.stream_time == pytest.approx(3.0)
