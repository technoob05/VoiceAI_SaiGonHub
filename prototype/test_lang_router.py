"""Tests for lang_router: text language ID over ASR transcripts + direction routing."""

from __future__ import annotations

import pytest

from lang_router import (
    COMPETITION_PAIRS,
    NLLB_CODE,
    DetectionResult,
    DetectorConfig,
    DirectionRouter,
    detect_language,
    detect_language_in_pair,
    normalize_pair,
    score_languages,
)


VI_WITH_DIACRITICS = [
    "tôi cần đặt một phòng khách sạn cho ngày mai",
    "xin chào, anh có thể giúp tôi kiểm tra máy này không",
    "chỗ này bị hỏng rồi, mình phải gọi kỹ thuật lên sửa",
]

VI_WITHOUT_DIACRITICS = [
    "toi muon dat mot phong khach san cho ngay mai",
    "toi khong biet cai nay lam sao cho dung",
    "anh cai dat phan mem giup em nhe",
]

EN_SENTENCES = [
    "please translate this into vietnamese for me",
    "can you show me where the safety helmet is stored",
    "the machine is not working and we need help",
]

ZH_SENTENCES = [
    "我 想 预订 一 个 房间 明天 入住",
    "这 台 机器 坏 了 请 找 技术 人员",
]

KO_SENTENCES = [
    "안녕하세요 저는 한국에서 왔습니다 도와주세요",
    "이 기계 가 고장 났 습니다 기술자 를 불러 주세요",
]


# --- open 4-way detection -------------------------------------------------


@pytest.mark.parametrize("text", VI_WITH_DIACRITICS)
def test_vietnamese_with_diacritics(text):
    result = detect_language(text)
    assert result.lang == "vi"
    assert not result.low_confidence
    assert result.reason == "detected"
    assert result.confidence > 0.9


@pytest.mark.parametrize("text", VI_WITHOUT_DIACRITICS)
def test_vietnamese_without_diacritics(text):
    """ASR sometimes emits unaccented Vietnamese; function words + syllable shape
    have to carry the decision on their own."""
    result = detect_language(text)
    assert result.lang == "vi"
    assert not result.low_confidence
    assert result.scores["vi"] > result.scores["en"]


@pytest.mark.parametrize("text", EN_SENTENCES)
def test_english(text):
    result = detect_language(text)
    assert result.lang == "en"
    assert not result.low_confidence


@pytest.mark.parametrize("text", ZH_SENTENCES)
def test_chinese(text):
    result = detect_language(text)
    assert result.lang == "zh"
    assert not result.low_confidence


@pytest.mark.parametrize("text", KO_SENTENCES)
def test_korean(text):
    result = detect_language(text)
    assert result.lang == "ko"
    assert not result.low_confidence


def test_hangul_jamo_still_reads_korean():
    result = detect_language("ㄱㅏㄴㅏㄷㅏ ㅎㅏㅅㅔㅇㅛ")
    assert result.lang == "ko"


# --- code-switching -------------------------------------------------------


@pytest.mark.parametrize("text", [
    "tôi cần cài đặt phần mềm Bluetooth trên máy tính bảng",
    "anh ơi cho em hỏi cái sensor này bị lỗi firmware phải không",
    "máy in bị kẹt giấy, chị reset lại giúp em",
])
def test_vietnamese_with_embedded_english_term(text):
    """One English technical term must not flip a Vietnamese sentence."""
    result = detect_language(text)
    assert result.lang == "vi"
    assert not result.low_confidence
    assert result.scores["en"] > 0.0, "the English token should still score, just be outweighed"


def test_chinese_with_latin_loanword():
    result = detect_language("请 用 Python 写 一个 程序 处理 这个 文件")
    assert result.lang == "zh"
    assert not result.low_confidence
    assert result.scores["zh"] > result.scores["en"]


def test_chinese_with_digits_and_latin():
    result = detect_language("订单 号码 是 A1234 请 在 USB 接口 检查")
    assert result.lang == "zh"


# --- pair-restricted path -------------------------------------------------


@pytest.mark.parametrize("pair", COMPETITION_PAIRS)
def test_vietnamese_routes_as_vi_in_every_competition_pair(pair):
    result = detect_language_in_pair("tôi rất vui được gặp anh hôm nay", pair)
    assert result.lang == "vi"
    assert not result.low_confidence
    assert result.candidates == tuple(pair)


def test_pair_restricted_picks_the_non_vietnamese_side():
    assert detect_language_in_pair("안녕하세요 만나서 반갑습니다", ("vi", "ko")).lang == "ko"
    assert detect_language_in_pair("很 高兴 认识 你 今天 天气 很 好", ("vi", "zh")).lang == "zh"
    assert detect_language_in_pair("nice to meet you today", ("vi", "en")).lang == "en"


def test_pair_restricted_on_third_language_flags_out_of_pair():
    """Korean speech on a vi<->en device: say so, do not coin-flip."""
    result = detect_language_in_pair("안녕하세요 저는 한국에서 왔습니다", ("vi", "en"))
    assert result.low_confidence
    assert result.reason == "out_of_pair:ko"
    assert result.lang in ("vi", "en")
    assert result.confidence == 0.0
    assert result.scores["ko"] > 0.0


def test_pair_restricted_is_more_decisive_than_open_detection():
    text = "toi di lam ve"
    assert detect_language_in_pair(text, ("vi", "ko")).confidence >= detect_language(text).confidence


def test_confidence_always_describes_the_returned_language():
    result = detect_language_in_pair("Samsung Galaxy", ("vi", "en"))
    assert result.low_confidence
    scores = result.scores
    total = scores[result.candidates[0]] + scores[result.candidates[1]]
    expected = scores[result.lang] / total if total else 0.0
    assert result.confidence == pytest.approx(expected)


def test_normalize_pair_rejects_bad_pairs():
    assert normalize_pair(["VI", " en "]) == ("vi", "en")
    with pytest.raises(ValueError):
        normalize_pair(("vi", "vi"))
    with pytest.raises(ValueError):
        normalize_pair(("vi", "th"))
    with pytest.raises(ValueError):
        normalize_pair(("vi", "en", "ko"))


# --- low-confidence path --------------------------------------------------


@pytest.mark.parametrize("text,reason", [
    ("", "empty_input"),
    ("   ", "empty_input"),
    (None, "empty_input"),
    ("ok", "too_short"),
    ("hi", "too_short"),
    ("123 456", "too_short"),
    ("...", "too_short"),
])
def test_short_or_empty_input_is_low_confidence(text, reason):
    result = detect_language_in_pair(text, ("vi", "en"))
    assert result.low_confidence
    assert result.reason == reason
    assert result.confidence == 0.0


def test_no_evidence_on_content_free_latin_token():
    result = detect_language_in_pair("Samsung", ("vi", "en"))
    assert result.low_confidence
    assert result.reason in ("no_evidence", "weak_margin")


def test_tightening_the_config_makes_a_borderline_case_low_confidence():
    text = "turn it on"
    assert not detect_language_in_pair(text, ("vi", "en")).low_confidence
    strict = DetectorConfig(min_evidence=50.0)
    assert detect_language_in_pair(text, ("vi", "en"), strict).low_confidence


def test_score_languages_is_zero_for_empty_text():
    assert score_languages("") == {"vi": 0.0, "en": 0.0, "zh": 0.0, "ko": 0.0}


# --- DirectionRouter ------------------------------------------------------


def test_router_routes_vietnamese_source():
    router = DirectionRouter(("vi", "en"))
    decision = router.route("cho tôi hỏi phòng họp ở tầng mấy")
    assert (decision.source, decision.target) == ("vi", "en")
    assert decision.nllb_source == NLLB_CODE["vi"]
    assert decision.nllb_target == NLLB_CODE["en"]
    assert decision.tts_lang == "en"
    assert not decision.low_confidence
    assert not decision.manual
    assert isinstance(decision.detection, DetectionResult)


def test_router_routes_the_reverse_direction():
    router = DirectionRouter(("vi", "ko"))
    decision = router.route("안녕하세요 화장실 이 어디 에 있 습니까")
    assert (decision.source, decision.target) == ("ko", "vi")
    assert decision.nllb_source == "kor_Hang"
    assert decision.nllb_target == "vie_Latn"
    assert decision.tts_lang == "vi"


def test_manual_override_beats_detection():
    """A human pressed the English button while speaking Vietnamese-looking text.
    Detection must not win."""
    router = DirectionRouter(("vi", "en"))
    text = "tôi cần đặt một phòng khách sạn"
    assert router.route(text).source == "vi"

    router.set_manual_source("en")
    decision = router.route(text)
    assert decision.source == "en"
    assert decision.target == "vi"
    assert decision.tts_lang == "vi"
    assert decision.manual
    assert not decision.low_confidence
    assert decision.reason == "manual_override:en"
    assert decision.detection is None


def test_per_call_override_beats_the_sticky_one():
    router = DirectionRouter(("vi", "zh"), manual_source="vi")
    assert router.route("随便 什么 文字").source == "vi"
    assert router.route("随便 什么 文字", manual_source="zh").source == "zh"
    assert router.manual_source == "vi", "a per-call pin must not mutate the sticky pin"


def test_clearing_the_override_restores_detection():
    router = DirectionRouter(("vi", "en"), manual_source="en")
    assert router.route("tôi muốn uống nước").source == "en"
    router.clear_manual_source()
    assert router.route("tôi muốn uống nước").source == "vi"


def test_router_rejects_an_override_outside_the_pair():
    router = DirectionRouter(("vi", "en"))
    with pytest.raises(ValueError):
        router.set_manual_source("ko")
    with pytest.raises(ValueError):
        router.route("hello there", manual_source="zh")


def test_low_confidence_falls_back_to_the_default_source_first():
    router = DirectionRouter(("vi", "en"), default_source="en")
    decision = router.route("ok")
    assert decision.low_confidence
    assert decision.source == "en"
    assert decision.reason == "too_short->default_source"


def test_low_confidence_falls_back_to_the_last_confident_direction():
    router = DirectionRouter(("vi", "en"))
    first = router.route("can you check the pressure gauge for me please")
    assert first.source == "en" and not first.low_confidence

    decision = router.route("ok")
    assert decision.low_confidence
    assert decision.source == "en", "a short grunt should continue the running direction"
    assert decision.reason.endswith("->last_direction")


def test_low_confidence_is_visible_to_the_caller_on_a_third_language():
    router = DirectionRouter(("vi", "en"))
    decision = router.route("안녕하세요 저는 한국에서 왔습니다")
    assert decision.low_confidence
    assert decision.detection.reason == "out_of_pair:ko"


def test_route_decision_is_serialisable():
    decision = DirectionRouter(("vi", "en")).route("tôi khỏe, cảm ơn anh")
    payload = decision.to_dict()
    assert payload["source"] == "vi"
    assert payload["nllb_target"] == "eng_Latn"
    assert payload["detection"]["lang"] == "vi"


def test_all_competition_pairs_produce_valid_nllb_codes():
    for pair in COMPETITION_PAIRS:
        router = DirectionRouter(pair)
        for pinned in pair:
            decision = router.route("bất kỳ văn bản nào", manual_source=pinned)
            assert decision.nllb_source == NLLB_CODE[pinned]
            assert decision.nllb_target == NLLB_CODE[router.other(pinned)]
            assert decision.tts_lang in pair
