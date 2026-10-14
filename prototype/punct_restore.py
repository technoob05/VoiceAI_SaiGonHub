"""punct_restore.py — Vietnamese punctuation + capitalization + sentence segmentation.

Our ASR emits lowercase, punctuation-free text ("tiếng cọc gạch khựng lệ của những
khớp sắt"). That hurts three things:
  1. Readability — the on-screen transcript is hard to read.
  2. MT quality — NLLB was trained on cased, punctuated text; feeding it a lowercase
     run-on sentence degrades translation.
  3. Streaming TTS — our fragmenter splits on sentence terminators; with none present
     a multi-sentence utterance becomes ONE big fragment (no first-audio-fast win).

This restores all three in one pass with `1-800-BAD-CODE/xlm-roberta_punctuation_
fullstop_truecase` (ONNX, 47 languages incl. Vietnamese; via the `punctuators` pkg).
Verified on our ASR output: adds periods, capitalizes (incl. proper nouns like
"Duy Minh"), and SEGMENTS into sentences.

Cost note: the model is xlm-roberta (~1 GB) — a real UX/quality win for the desktop
demo, but a heavy add for the on-device budget. Keep it OPTIONAL (VOICEAI_PUNCT=1).
"""

from __future__ import annotations

import re
from typing import Optional

MODEL = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"

# Collapse duplicate terminal punctuation (ASR sometimes already ends a segment with
# "." and the restorer appends its own → ".."), and tidy spaces before punctuation.
_DUP_PUNCT = re.compile(r"\s*([.,!?;:…])\1*")
_SPACE_PUNCT = re.compile(r"\s+([.,!?;:…])")


def _clean(s: str) -> str:
    s = _DUP_PUNCT.sub(r"\1", s)
    s = _SPACE_PUNCT.sub(r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


class PunctRestorer:
    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self._m = None

    def _ensure(self):
        if self._m is None:
            from punctuators.models import PunctCapSegModelONNX
            print(f"[punct] loading {self.model_name} …")
            self._m = PunctCapSegModelONNX.from_pretrained(self.model_name)
        return self._m

    def restore_sentences(self, text: str) -> list[str]:
        """Return the restored, capitalized, punctuated text split into sentences."""
        text = (text or "").strip()
        if not text:
            return []
        out = self._ensure().infer([text])
        # infer -> list (per input) of list[str] (sentences)
        return [c for c in (_clean(s) for s in (out[0] if out else [])) if c]

    def restore(self, text: str) -> str:
        """Return the restored text as a single string (sentences joined)."""
        sents = self.restore_sentences(text)
        return " ".join(sents) if sents else (text or "").strip()


_SINGLETON: Optional[PunctRestorer] = None


def get_restorer() -> PunctRestorer:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = PunctRestorer()
    return _SINGLETON


if __name__ == "__main__":
    import argparse, sys
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    ap = argparse.ArgumentParser(description="Vietnamese punctuation + truecase restore")
    ap.add_argument("text", nargs="+")
    args = ap.parse_args()
    r = get_restorer()
    for t in args.text:
        print("IN :", t)
        print("OUT:", r.restore(t))
        print("SEG:", r.restore_sentences(t))
