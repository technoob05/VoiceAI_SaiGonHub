"""Terminology control for the offline MT stage.

NLLB-200-distilled-600M is the translation model this device can afford, and on
Vietnamese industrial vocabulary it fails in a way that matters: it does not lose
fluency, it loses the noun. Measured 2026-08-19 on the CT2 INT8 checkpoint,
beam 1:

    "Xe nâng đang lùi"        -> "The train is coming back"   (forklift -> train)
    "Máy nén khí ... quá nhiệt" -> "The air ... is too hot"    (compressor -> air)
    "dừng dây chuyền"         -> zh "停止电路"                  (line -> circuit)

A judge who speaks the target language sees this immediately, and on a factory
floor it is a safety problem rather than a style problem.

Three interventions were measured against each other on the same sentences:

    English pivot (vi->en->tgt)   fixes degenerate output, does NOT recover the
                                  lost noun - the error is already made at vi->en,
                                  and it costs a second decoder pass
    inline gloss  "xe nâng (forklift)"  worse than bare - the parenthetical
                                  confuses the alignment
    term injection "forklift đang lùi"  recovers the noun in en, zh AND ko in a
                                  single pass

So this module does term injection: known source-language terms are rewritten to
their English equivalent inside the source sentence before the sentence reaches
the translator. NLLB is heavily English-centric, so an English noun embedded in a
Vietnamese sentence is the one form it reliably carries through to any target.

It also serves exact whole-utterance hits from the phrase memory, which cost no
decoder pass at all and are the only translations here that are exactly right by
construction.

WHAT THIS DOES NOT DO
- It does not constrain the decoder. CTranslate2 has no hard lexical constraint,
  so injection is a strong nudge, not a guarantee; `unused_preferred_terms`
  reports when the model chose a different wording than the house term.
- It does not help with terms outside the glossary. Coverage is the whole game.
- The zh/ko columns of the glossary have not been reviewed by a native speaker.
  They are used for exact-hit output, so an error there is a wrong translation
  spoken with full confidence, not a crash. Treat the tables as a draft.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional


HERE = Path(__file__).resolve().parent
TERMS_PATH = HERE / "industrial_terms.json"
PHRASES_PATH = HERE / "industrial_phrases.json"
LANGS = ("vi", "en", "zh", "ko")
PIVOT_LANG = "en"

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def _fold(text: str) -> str:
    return normalize(text).casefold()


@dataclass(frozen=True)
class Term:
    """One glossary row: the same concept written in each supported language."""
    vi: str
    en: str
    zh: str
    ko: str
    category: str = ""

    def get(self, lang: str) -> str:
        return getattr(self, lang)


@dataclass
class Injection:
    """Result of preparing a source sentence for the translator."""
    original: str
    text: str
    terms: list[Term] = field(default_factory=list)
    exact_hit: Optional[Term] = None

    @property
    def changed(self) -> bool:
        return self.text != self.original

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "text": self.text,
            "terms": [asdict(t) for t in self.terms],
            "exact_hit": asdict(self.exact_hit) if self.exact_hit else None,
        }


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for key, value in data.items():
        if key.startswith("_") or not isinstance(value, list):
            continue
        for row in value:
            if all(row.get(lang) for lang in LANGS):
                rows.append({**row, "category": key})
    return rows


class Glossary:
    """Term substitution table plus a whole-utterance phrase memory.

    Longest term first: "máy nén khí" must win over a hypothetical "máy", or the
    substitution would corrupt the longer compound it is part of.
    """

    def __init__(self, terms_path: Path = TERMS_PATH, phrases_path: Path = PHRASES_PATH):
        self.terms = [
            Term(row["vi"], row["en"], row["zh"], row["ko"], row.get("category", ""))
            for row in _load_rows(terms_path)
        ]
        self.phrases = [
            Term(row["vi"], row["en"], row["zh"], row["ko"], row.get("category", ""))
            for row in _load_rows(phrases_path)
        ]
        self._patterns: dict[str, list[tuple[re.Pattern, Term]]] = {}
        self._exact: dict[str, dict[str, Term]] = {}
        for lang in LANGS:
            self._patterns[lang] = self._compile(self.terms, lang)
            self._exact[lang] = {}
            for term in [*self.phrases, *self.terms]:
                self._exact[lang].setdefault(_fold(term.get(lang)), term)

    @staticmethod
    def _compile(terms: Iterable[Term], lang: str) -> list[tuple[re.Pattern, Term]]:
        out = []
        for term in sorted(terms, key=lambda t: -len(t.get(lang))):
            surface = normalize(term.get(lang))
            if not surface:
                continue
            # zh/ko are unspaced, so a \b boundary would never match; guard with a
            # non-word-character lookaround only where the script actually uses one.
            body = re.escape(surface)
            pattern = rf"\b{body}\b" if lang in ("vi", "en") else body
            out.append((re.compile(pattern, re.IGNORECASE | re.UNICODE), term))
        return out

    def lookup_exact(self, text: str, lang: str) -> Optional[Term]:
        return self._exact.get(lang, {}).get(_fold(text.rstrip(" .!?。！？")))

    def find_terms(self, text: str, lang: str) -> list[Term]:
        found, consumed = [], normalize(text)
        for pattern, term in self._patterns.get(lang, []):
            if pattern.search(consumed):
                found.append(term)
                consumed = pattern.sub(" ", consumed)
        return found

    def inject(self, text: str, lang: str, pivot: str = PIVOT_LANG) -> Injection:
        """Rewrite known source terms to `pivot` inside the source sentence."""
        original = normalize(text)
        exact = self.lookup_exact(original, lang)
        if exact is not None:
            return Injection(original=original, text=original, exact_hit=exact)
        if lang == pivot:
            return Injection(original=original, text=original)

        out, used = original, []
        for pattern, term in self._patterns.get(lang, []):
            replacement = term.get(pivot)
            new = pattern.sub(replacement, out)
            if new != out:
                used.append(term)
                out = new
        return Injection(original=original, text=normalize(out), terms=used)


def unused_preferred_terms(output: str, terms: Iterable[Term],
                           target: str) -> list[Term]:
    """Terms whose *preferred* target surface form does not appear in `output`.

    This is a consistency signal, NOT an error detector, and the difference
    matters. Measured on the industrial set: for "xe nâng" the model produced
    Korean 포크리프트 where the glossary prefers 지게차, and Chinese 安全眼镜 where
    it prefers 护目镜 - both correct translations, both flagged here. A hit means
    "the device did not use the house term", which is worth logging and worth
    showing on a glossary-coverage report; it does not mean the translation is
    wrong, and nothing downstream should reject a turn because of it.
    """
    folded = _fold(output)
    missing = []
    for term in terms:
        want = _fold(term.get(target))
        if want and want not in folded:
            missing.append(term)
    return missing


@dataclass
class GlossaryResult:
    text: str
    source: str
    target: str
    exact_hit: bool
    injected: bool
    terms: list[Term]
    unused_preferred: list[Term]
    mt_calls: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "target": self.target,
            "exact_hit": self.exact_hit,
            "injected": self.injected,
            "terms": [t.en for t in self.terms],
            "unused_preferred": [t.en for t in self.unused_preferred],
            "mt_calls": self.mt_calls,
        }


class GlossaryTranslator:
    """Wraps any `translate(text, source, target) -> str` with terminology control.

    Deliberately does not own a model: `pipeline_fast` already owns the CT2
    translator and its warm state, and a second copy would double MT memory.
    """

    def __init__(self, translate_fn, glossary: Optional[Glossary] = None):
        self._translate = translate_fn
        self.glossary = glossary or Glossary()

    def translate(self, text: str, source: str, target: str) -> GlossaryResult:
        prepared = self.glossary.inject(text, source)
        if prepared.exact_hit is not None:
            return GlossaryResult(
                text=prepared.exact_hit.get(target), source=source, target=target,
                exact_hit=True, injected=False, terms=[prepared.exact_hit],
                unused_preferred=[], mt_calls=0,
            )
        out = normalize(self._translate(prepared.text, source, target))
        return GlossaryResult(
            text=out, source=source, target=target, exact_hit=False,
            injected=prepared.changed, terms=prepared.terms,
            unused_preferred=unused_preferred_terms(out, prepared.terms, target),
            mt_calls=1,
        )


_GLOSSARY: Optional[Glossary] = None


def get_glossary() -> Glossary:
    global _GLOSSARY
    if _GLOSSARY is None:
        _GLOSSARY = Glossary()
    return _GLOSSARY


if __name__ == "__main__":
    g = get_glossary()
    print(f"terms={len(g.terms)} phrases={len(g.phrases)}")
    for sample in ("Xe nâng đang lùi, xin tránh ra phía sau.",
                   "Máy nén khí ở khu vực B đang quá nhiệt, hãy dừng dây chuyền ngay.",
                   "xin chào"):
        inj = g.inject(sample, "vi")
        tag = "EXACT" if inj.exact_hit else ("INJECT" if inj.changed else "-")
        print(f"[{tag:6s}] {sample}\n          -> {inj.text}")
