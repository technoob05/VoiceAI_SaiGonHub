"""Dependency-free text language ID and direction routing for the bidirectional device.

The competition requires two-way conversation over three pairs (vi<->en, vi<->zh,
vi<->ko), but the device always knows which pair it is configured for. That turns
the hard open-set language-ID problem into a cheap two-way decision, which is what
this module solves: score a transcript against exactly the two configured
languages, pick a direction, and say how much evidence backed the pick.

The four languages are near-separable by script. Hangul syllables and jamo mark
Korean, CJK ideographs mark Chinese, and the only genuinely contested pair is
Vietnamese versus English on shared Latin letters. That one is decided by
Vietnamese-specific graphemes (ăâđêôơư plus the five tone marks), by Vietnamese
function words in their diacritic-free ASR spelling, and by Vietnamese syllable
shape (short, monosyllabic, a closed set of codas, no w/f/j/z). Everything is one
pass over the characters plus one pass over the Latin tokens, so cost is linear in
transcript length and negligible next to ASR.

Decisions are additive weight of evidence, never a single trigger token, so a
Vietnamese sentence carrying an English technical term still routes as Vietnamese.
When the evidence is thin the result is flagged `low_confidence` with an explicit
`reason`, in the same spirit as transcript_corrector's CorrectionResult: a weak
decision is a normal outcome the caller handles, not a failure. DirectionRouter
additionally supports a manual pin, so the physical push-to-talk buttons always
beat detection.

WHAT THIS CANNOT DO -- read before trusting it
  * This is TEXT language ID over an ASR transcript. It inherits whatever language
    the ASR already committed to. A Vietnamese-only ASR fed Korean speech emits
    Vietnamese-looking tokens, and this module will faithfully report "vi" with
    high confidence. It cannot know the acoustics were Korean. Only an acoustic
    LID stage can.
  * Very short utterances ("ok", "hello", proper nouns, digits) carry too little
    evidence; they come back low_confidence rather than guessed.
  * Diacritic-free Vietnamese is decided by function words and syllable shape. A
    short diacritic-free Vietnamese sentence with no function words, or one built
    from English-lookalike syllables, is genuinely ambiguous.
  * Non-English Latin diacritics (cafe -> café, résumé) read as weak Vietnamese
    evidence; heavy code-switching is decided by majority, so the minority
    language in a mixed utterance is by construction routed wrong.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

SUPPORTED = ("vi", "en", "zh", "ko")

# Mirrors pipeline_fast.NLLB_CODE; duplicated so this module stays import-free.
NLLB_CODE = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
}

COMPETITION_PAIRS = (("vi", "en"), ("vi", "zh"), ("vi", "ko"))

_LATIN_TOKEN_RE = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]+")

# Combining marks Vietnamese uses pervasively: the five tones plus the three
# letter-shaping marks (breve, circumflex, horn).
_VI_TONE_MARKS = frozenset("̣̀́̃̉")
_VI_LETTER_MARKS = frozenset("̛̆̂")

_VI_SYLLABLE_RE = re.compile(
    r"^(ngh|ng|nh|ch|gh|gi|kh|ph|th|tr|qu|b|c|d|g|h|k|l|m|n|p|r|s|t|v|x)?"
    r"[aeiouy]{1,3}"
    r"(ng|nh|ch|c|m|n|p|t)?$"
)

# Diacritic-free spellings of common Vietnamese function words, filtered to drop
# anything that is also a common English word (an, can, do, day, no, on, van, in,
# it, so, ban, con, tin, that, ...). Precision matters more than recall here:
# a false Vietnamese hit on English text is worse than a missed cue, because
# diacritics already carry the ordinary Vietnamese case.
_VI_WORDS = frozenset("""
khong duoc nguoi nhung mot cac cho voi nay toi cua va roi thi vi nen cung dang
chung minh nhieu biet lam noi nghe muon phai sao dau moi luon chua boi tren
duoi trong ngoai truoc khi neu tuy ong chao cam giup xin vang dung vao len
xuong gio hom mai thang ngay tieng viet nha nuoc uong cuoc chuyen viec tot xau
nhanh cham kia ay ho la ma da se qua vay nao nhe cai chiec giay phut lai ve ra
bao gap doi doc hoc thay tro cong ty may tinh dien thoai xe duong bat tat mo
dong nong lanh sang dem trua chieu giup dat phong khach san
""".split())

_EN_WORDS = frozenset("""
the and is are was were of to that this with for have has had will would on at
be been being not but from they we she his her our your my what how why when
where which could should does did there here about into over after before
please thank thanks hello need want make know think time work people very more
than then also just only some any all one two three because their them english
vietnamese korean chinese
""".split())

_EN_ONLY_LETTERS = frozenset("wfjz")

_W_VI_DIACRITIC = 2.0
_W_VI_EXTRA_MARK = 0.25
_W_VI_WORD = 1.0
_W_EN_WORD = 1.0
_W_EN_LETTER = 0.6
_W_LONG_TOKEN = 0.5
_W_SYLLABLE_OK = 0.25
_W_SYLLABLE_BAD = 0.35
_W_HANGUL_CHAR = 3.0
_W_HAN_CHAR = 3.0


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _ascii_fold(token: str) -> str:
    decomposed = unicodedata.normalize("NFD", token.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _is_hangul(code: int) -> bool:
    return (
        0xAC00 <= code <= 0xD7A3
        or 0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
        or 0xA960 <= code <= 0xA97F
    )


def _is_han(code: int) -> bool:
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
    )


def _vi_marks(token: str) -> int:
    """Vietnamese-specific graphemes in a token: tones, shaped letters, and đ."""
    hits = sum(1 for ch in token if ch in "đĐ")
    for ch in unicodedata.normalize("NFD", token):
        if ch in _VI_TONE_MARKS or ch in _VI_LETTER_MARKS:
            hits += 1
    return hits


def _vn_syllable_ok(ascii_token: str) -> bool:
    return len(ascii_token) <= 7 and bool(_VI_SYLLABLE_RE.match(ascii_token))


@dataclass
class DetectorConfig:
    """Gates on when a decision counts as confident.

    The thresholds sit on accumulated evidence mass rather than on a
    probability, so a long confident sentence and a two-word guess are not
    conflated just because both normalise to the same ratio.
    """

    min_chars: int = 3
    min_evidence: float = 1.0
    min_margin: float = 0.5
    min_confidence: float = 0.60


@dataclass
class DetectionResult:
    text: str
    lang: str
    confidence: float
    reason: str
    low_confidence: bool
    scores: dict = field(default_factory=dict)
    candidates: tuple = ()

    def to_dict(self) -> dict:
        return asdict(self)


def score_languages(text: str) -> dict:
    """Raw additive evidence for each of the four languages. O(len(text))."""
    raw = normalize_text(text)
    scores = {lang: 0.0 for lang in SUPPORTED}
    if not raw:
        return scores

    for ch in raw:
        code = ord(ch)
        if _is_hangul(code):
            scores["ko"] += _W_HANGUL_CHAR
        elif _is_han(code):
            scores["zh"] += _W_HAN_CHAR

    for token in _LATIN_TOKEN_RE.findall(raw):
        folded = _ascii_fold(token)
        if not folded:
            continue
        marks = _vi_marks(token)
        if marks:
            scores["vi"] += _W_VI_DIACRITIC + _W_VI_EXTRA_MARK * (marks - 1)
        elif folded in _VI_WORDS:
            scores["vi"] += _W_VI_WORD
        elif folded in _EN_WORDS:
            scores["en"] += _W_EN_WORD

        if _EN_ONLY_LETTERS.intersection(folded):
            scores["en"] += _W_EN_LETTER
        if len(folded) > 7:
            scores["en"] += _W_LONG_TOKEN
        if _vn_syllable_ok(folded):
            scores["vi"] += _W_SYLLABLE_OK
        else:
            scores["en"] += _W_SYLLABLE_BAD

    return scores


def normalize_pair(pair: Iterable[str]) -> tuple:
    langs = tuple(str(lang).strip().lower() for lang in pair)
    if len(langs) != 2:
        raise ValueError(f"a language pair needs exactly 2 languages, got {langs!r}")
    if langs[0] == langs[1]:
        raise ValueError(f"a language pair needs 2 distinct languages, got {langs!r}")
    for lang in langs:
        if lang not in SUPPORTED:
            raise ValueError(f"unsupported language {lang!r}; supported: {SUPPORTED}")
    return langs


def _decide(
    raw: str,
    scores: dict,
    candidates: tuple,
    config: DetectorConfig,
    fallback: str,
) -> DetectionResult:
    ranked = sorted(candidates, key=lambda lang: (-scores[lang], lang))
    best = ranked[0]
    top = scores[best]
    second = scores[ranked[1]] if len(ranked) > 1 else 0.0
    total = sum(scores[lang] for lang in candidates)

    # Confidence always describes the language actually returned, including the
    # fallback ones -- a caller must never read 1.0 next to a language the
    # evidence did not pick.
    def confidence_for(lang: str) -> float:
        return (scores[lang] / total) if total > 0 else 0.0

    confidence = confidence_for(best)

    outside = [lang for lang in SUPPORTED if lang not in candidates and scores[lang] > top]
    if outside:
        dominant = max(outside, key=lambda lang: scores[lang])
        return DetectionResult(
            raw, fallback, confidence_for(fallback), f"out_of_pair:{dominant}", True, scores, candidates
        )
    if total < config.min_evidence:
        return DetectionResult(
            raw, fallback, confidence_for(fallback), "no_evidence", True, scores, candidates
        )
    if top - second < config.min_margin:
        return DetectionResult(raw, best, confidence, "weak_margin", True, scores, candidates)
    if confidence < config.min_confidence:
        return DetectionResult(raw, best, confidence, "weak_confidence", True, scores, candidates)
    return DetectionResult(raw, best, confidence, "detected", False, scores, candidates)


def detect_language_in_pair(
    text: str,
    pair: Iterable[str],
    config: DetectorConfig | None = None,
) -> DetectionResult:
    """Primary entry point: score a transcript against only the configured pair.

    Evidence for the other two languages is still computed, and when it dominates
    the result is flagged `out_of_pair:<lang>` -- the caller learns the device is
    configured for the wrong pair instead of silently getting a coin flip.
    """
    config = config or DetectorConfig()
    candidates = tuple(str(x).strip().lower() for x in pair)
    if candidates != SUPPORTED:
        candidates = normalize_pair(candidates)
    fallback = candidates[0]
    raw = normalize_text(text)
    empty_scores = {lang: 0.0 for lang in SUPPORTED}

    if not raw:
        return DetectionResult(raw, fallback, 0.0, "empty_input", True, empty_scores, candidates)
    if sum(1 for ch in raw if ch.isalpha()) < config.min_chars:
        return DetectionResult(
            raw, fallback, 0.0, "too_short", True, score_languages(raw), candidates
        )

    return _decide(raw, score_languages(raw), candidates, config, fallback)


def detect_language(text: str, config: DetectorConfig | None = None) -> DetectionResult:
    """Open 4-way identification over vi/en/zh/ko.

    Prefer `detect_language_in_pair` whenever the configured pair is known: a
    2-way decision inside a known pair is strictly more reliable, because it
    never has to rule out a language the device is not configured for.
    """
    return detect_language_in_pair(text, SUPPORTED, config)


@dataclass
class RouteDecision:
    text: str
    source: str
    target: str
    nllb_source: str
    nllb_target: str
    tts_lang: str
    confidence: float
    reason: str
    low_confidence: bool
    manual: bool
    detection: Optional[DetectionResult] = None

    def to_dict(self) -> dict:
        return asdict(self)


class DirectionRouter:
    """Turns a transcript plus a configured pair into a translation direction.

    Precedence is fixed and deliberate: a manual pin (the physical push-to-talk
    buttons) wins, then detection, then the last confident direction, then the
    pair's default source. Detection can never override a human's explicit choice.
    """

    def __init__(
        self,
        pair: Iterable[str] = ("vi", "en"),
        config: DetectorConfig | None = None,
        manual_source: Optional[str] = None,
        default_source: Optional[str] = None,
    ):
        self.pair = normalize_pair(pair)
        self.config = config or DetectorConfig()
        self.default_source = self._check(default_source) if default_source else self.pair[0]
        self.manual_source = self._check(manual_source) if manual_source else None
        self.last_source: Optional[str] = None

    def _check(self, lang: Optional[str]) -> str:
        lang = (lang or "").strip().lower()
        if lang not in self.pair:
            raise ValueError(f"{lang!r} is not in the configured pair {self.pair}")
        return lang

    def set_manual_source(self, lang: Optional[str]) -> None:
        self.manual_source = None if lang is None else self._check(lang)

    def clear_manual_source(self) -> None:
        self.manual_source = None

    def other(self, lang: str) -> str:
        return self.pair[1] if lang == self.pair[0] else self.pair[0]

    def _decision(self, text, source, confidence, reason, low, manual, detection) -> RouteDecision:
        target = self.other(source)
        return RouteDecision(
            text=text,
            source=source,
            target=target,
            nllb_source=NLLB_CODE[source],
            nllb_target=NLLB_CODE[target],
            tts_lang=target,
            confidence=confidence,
            reason=reason,
            low_confidence=low,
            manual=manual,
            detection=detection,
        )

    def route(self, text: str, manual_source: Optional[str] = None) -> RouteDecision:
        pinned = self._check(manual_source) if manual_source else self.manual_source
        raw = normalize_text(text)
        if pinned:
            return self._decision(raw, pinned, 1.0, f"manual_override:{pinned}", False, True, None)

        detection = detect_language_in_pair(text, self.pair, self.config)
        if not detection.low_confidence:
            self.last_source = detection.lang
            return self._decision(
                raw, detection.lang, detection.confidence, detection.reason, False, False, detection
            )

        source = self.last_source or self.default_source
        origin = "last_direction" if self.last_source else "default_source"
        return self._decision(
            raw, source, detection.confidence, f"{detection.reason}->{origin}", True, False, detection
        )
