"""Unit tests for the pure fragmentation + audio-shaping helpers of streaming_tts.

No TTS backend is loaded: fragment_text/split_sentences are pure string functions
and shape_fragment only needs a numpy array.
"""

from __future__ import annotations

import numpy as np
import pytest

from streaming_tts import (
    StreamingTTSConfig,
    _fade,
    _trim_silence,
    fragment_text,
    shape_fragment,
    split_sentences,
)


def cfg(**kw) -> StreamingTTSConfig:
    return StreamingTTSConfig(**kw)


# --- fix 2: decimal numbers must not split a sentence ----------------------
def test_decimal_not_split():
    assert split_sentences("Nhiet do la 1.5 do C. Xong.") == [
        "Nhiet do la 1.5 do C.",
        "Xong.",
    ]


def test_multiple_decimals_and_versions():
    assert split_sentences("Ap suat 2.75 bar, sai so 0.05. Kiem tra lai.") == [
        "Ap suat 2.75 bar, sai so 0.05.",
        "Kiem tra lai.",
    ]


def test_ascii_no_space_split_still_works():
    assert split_sentences("Xong.Tiep tuc.") == ["Xong.", "Tiep tuc."]


def test_cjk_terminators_still_split():
    assert split_sentences("你好。世界！好的？") == ["你好。", "世界！", "好的？"]


def test_vietnamese_sentences_split():
    assert split_sentences("Xin chào mọi người. Hôm nay trời rất đẹp.") == [
        "Xin chào mọi người.",
        "Hôm nay trời rất đẹp.",
    ]


# --- fix 1: fast first fragment -------------------------------------------
LONG_FIRST = ("Nhiet do trong nha may hien tai la bon muoi lam do C, "
              "vuot nguong an toan, hay dung day chuyen ngay. Cam on.")


def test_first_fragment_shorter_than_first_sentence():
    frags = fragment_text(LONG_FIRST, cfg())
    first_sentence = split_sentences(LONG_FIRST)[0]
    assert len(frags[0]) < len(first_sentence)
    assert frags[0].endswith(",")
    assert len(frags[0]) >= StreamingTTSConfig().min_first_fragment_chars


def test_fast_cut_takes_first_clause_at_or_after_threshold():
    frags = fragment_text("Mot hai ba bon nam sau, bay tam chin muoi, ket thuc.",
                          cfg(min_first_fragment_chars=12))
    assert frags[0] == "Mot hai ba bon nam sau,"


def test_short_first_sentence_left_alone():
    text = "Dung lai. " + "Kiem tra day chuyen so hai ngay bay gio, bao cao lai."
    frags = fragment_text(text, cfg(min_first_fragment_chars=50))
    assert frags[0].startswith("Dung lai.")


def test_no_clause_delimiter_no_cut():
    frags = fragment_text("Xin chao moi nguoi hom nay. Tam biet nhe cac ban.", cfg())
    assert frags[0] == "Xin chao moi nguoi hom nay."


def test_no_words_lost_after_fast_cut():
    frags = fragment_text(LONG_FIRST, cfg())
    joined = " ".join(frags).replace("  ", " ")
    for word in LONG_FIRST.replace(",", "").replace(".", "").split():
        assert word in joined


# --- fix 3: short-fragment merge ------------------------------------------
def test_short_sentence_merged_into_next():
    frags = fragment_text("Duoc. Chung ta hay bat dau cuoc hop ngay bay gio.",
                          cfg(min_first_fragment_chars=100, min_fragment_chars=25))
    assert frags == ["Duoc. Chung ta hay bat dau cuoc hop ngay bay gio."]


def test_merge_does_not_defeat_fast_first_fragment():
    frags = fragment_text(LONG_FIRST, cfg())
    assert len(frags) >= 2
    # fragment 0 is under min_fragment_chars-style merging protection
    assert frags[0].endswith(",")
    assert not frags[1].startswith(frags[0])


def test_merge_refused_when_it_would_exceed_max_fragment_chars():
    text = "Ok. " + " ".join(["dai"] * 60) + "."
    frags = fragment_text(text, cfg(min_first_fragment_chars=100, max_fragment_chars=40))
    assert frags[0] == "Ok."


def test_merge_disabled_when_threshold_zero():
    frags = fragment_text("Duoc. Chung ta bat dau ngay.",
                          cfg(min_first_fragment_chars=0, min_fragment_chars=0))
    assert frags == ["Duoc.", "Chung ta bat dau ngay."]


# --- fix 4: fade only on trimmed edges ------------------------------------
SR = 16_000


def speech(n: int = 4_000) -> np.ndarray:
    return np.full(n, 0.5, dtype=np.float32)


def test_trim_silence_reports_edges():
    a = np.concatenate([np.zeros(100, np.float32), speech(), np.zeros(100, np.float32)])
    trimmed, head, tail = _trim_silence(a, 0.01)
    assert head and tail and len(trimmed) == 4_000

    trimmed, head, tail = _trim_silence(speech(), 0.01)
    assert not head and not tail


def test_no_fade_when_nothing_trimmed():
    a = speech()
    out = shape_fragment(a, SR, cfg())
    assert np.array_equal(out, a)


def test_fade_only_on_trimmed_edge():
    a = np.concatenate([np.zeros(100, np.float32), speech()])
    out = shape_fragment(a, SR, cfg())
    n = int(SR * StreamingTTSConfig().fade_ms / 1000)
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < out[n // 2] < 0.5
    assert out[-1] == pytest.approx(0.5, abs=1e-6)


def test_fade_both_edges_when_both_trimmed():
    a = np.concatenate([np.zeros(100, np.float32), speech(), np.zeros(100, np.float32)])
    out = shape_fragment(a, SR, cfg())
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[-1] == pytest.approx(0.0, abs=1e-6)


def test_fade_helper_respects_flags():
    a = speech()
    head_only = _fade(a, SR, 12, head=True, tail=False)
    assert head_only[0] == pytest.approx(0.0, abs=1e-6)
    assert head_only[-1] == pytest.approx(0.5, abs=1e-6)


def test_shape_fragment_noop_when_trim_disabled():
    a = np.concatenate([np.zeros(100, np.float32), speech()])
    out = shape_fragment(a, SR, cfg(trim_silence=False))
    assert np.array_equal(out, a)
