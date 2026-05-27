"""
Coarse-to-fine MT routing — tiny draft + NLLB verifier.

Idea (paper §3.9 candidate): NLLB-200-distilled-600M is overkill for ~80% of
short factory/construction phrases. A tiny Helsinki-NLP opus-mt-vi-en model
(~300 MB, INT8 ~80 MB) handles these in 100-300 ms. We route by a cheap
entropy probe on the draft output: if confidence is low, escalate to NLLB.

Coverage limitation: opus-mt-vi-en is Vi→En only. For Vi→Zh / Vi→Ko we always
use NLLB. So this module wraps Vi→En specifically.

Implementation:
  1. Load opus-mt-vi-en (HF transformers, dyn-INT8) + NllbTokenizerFast + CT2 NLLB INT8.
  2. translate(text): run opus-mt fast path. Compute mean-token logprob.
  3. If mean_logprob ≥ -1.5 → return opus-mt result. Else → run NLLB and return.
  4. Track per-call routing so we can report % fast-path hits.

Self-test on 5 Vi phrases:
  short common phrases → expected fast path
  noisy or ambiguous   → expected NLLB fallback
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROJECT = HERE.parent
HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
os.environ.setdefault("HF_HOME",            str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE",       str(HF_CACHE / "hub"))
os.environ.setdefault("HF_XET_CACHE",       str(HF_CACHE / "xet"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "hub"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass


@dataclass
class C2FStats:
    calls: int = 0
    fast_path_hits: int = 0
    nllb_fallback: int = 0
    fast_path_ms_total: float = 0.0
    nllb_ms_total: float = 0.0

    @property
    def fast_rate(self) -> float:
        return self.fast_path_hits / max(self.calls, 1)

    def summary(self) -> dict:
        return {
            "calls":              self.calls,
            "fast_path_hits":     self.fast_path_hits,
            "nllb_fallback":      self.nllb_fallback,
            "fast_rate":          round(self.fast_rate, 3),
            "fast_path_ms_mean":  round(self.fast_path_ms_total / max(self.fast_path_hits, 1), 1),
            "nllb_ms_mean":       round(self.nllb_ms_total / max(self.nllb_fallback, 1), 1),
        }


@dataclass
class C2FConfig:
    opus_mt_id:    str   = "Helsinki-NLP/opus-mt-vi-en"
    confidence_min: float = -1.5      # mean-token-logprob threshold (less negative = more confident)
    quantize_opus:  bool  = True       # apply torch dyn INT8 to opus-mt Linear layers
    max_new_tokens: int   = 256


class CoarseToFineMT:
    """Wraps a tiny opus-mt-vi-en (fast path) and your CT2-NLLB (verifier).
    Use `translate_vi_en(text)` for Vi→En only.
    """

    def __init__(self, nllb_translate_fn, cfg: C2FConfig | None = None):
        """
        nllb_translate_fn: callable(text) → str. Plug in your FastCascadeS2ST.translate
        bound to (target='en', source='vi') so this class only needs to call it.
        """
        self.cfg = cfg or C2FConfig()
        self._nllb = nllb_translate_fn
        self._opus_tok = None
        self._opus_model = None
        self.stats = C2FStats()

    def _ensure_opus(self) -> None:
        if self._opus_model is not None:
            return
        import torch
        from transformers import MarianTokenizer, MarianMTModel
        print(f"[load] coarse MT (opus-mt-vi-en): {self.cfg.opus_mt_id}")
        # Direct MarianTokenizer/MarianMTModel — AutoTokenizer factory broken
        # in transformers 5.0.0 for some legacy model types (NLLB, Marian).
        self._opus_tok = MarianTokenizer.from_pretrained(self.cfg.opus_mt_id)
        m = MarianMTModel.from_pretrained(self.cfg.opus_mt_id).eval()
        if self.cfg.quantize_opus:
            try:
                m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
                print("[load] opus-mt dyn-INT8 quantized")
            except Exception as e:
                print(f"[warn] opus-mt INT8 failed ({e}); using FP32")
        self._opus_model = m

    def translate_vi_en(self, text: str) -> tuple[str, dict]:
        """Vi → En with adaptive routing. Returns (translation, info)."""
        import torch
        self.stats.calls += 1
        if not text or not text.strip():
            return "", {"path": "empty"}

        self._ensure_opus()
        t0 = time.perf_counter()
        inputs = self._opus_tok(text, return_tensors="pt")
        with torch.inference_mode():
            out = self._opus_model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                num_beams=1,
            )
        opus_ms = (time.perf_counter() - t0) * 1000

        # Mean-token logprob (negative log-likelihood per generated token).
        # `scores` is a tuple of (batch, vocab) logits per generation step.
        token_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        if len(out.scores) > 0:
            logprobs = []
            for step_idx, score in enumerate(out.scores):
                if step_idx >= len(token_ids):
                    break
                logp = torch.log_softmax(score[0], dim=-1)
                logprobs.append(logp[int(token_ids[step_idx])].item())
            mean_logprob = float(sum(logprobs) / max(len(logprobs), 1)) if logprobs else -10.0
        else:
            mean_logprob = -10.0

        opus_text = self._opus_tok.decode(token_ids, skip_special_tokens=True).strip()

        if mean_logprob >= self.cfg.confidence_min and opus_text:
            self.stats.fast_path_hits += 1
            self.stats.fast_path_ms_total += opus_ms
            return opus_text, {
                "path":         "opus-mt-fast",
                "mean_logprob": round(mean_logprob, 3),
                "opus_ms":      round(opus_ms, 1),
            }

        # Fallback to NLLB
        t0n = time.perf_counter()
        nllb_text = self._nllb(text)
        nllb_ms = (time.perf_counter() - t0n) * 1000
        self.stats.nllb_fallback += 1
        self.stats.nllb_ms_total += nllb_ms
        return nllb_text, {
            "path":         "nllb-fallback",
            "mean_logprob": round(mean_logprob, 3),
            "opus_ms":      round(opus_ms, 1),
            "nllb_ms":      round(nllb_ms, 1),
            "draft":        opus_text,
        }


# Self-test
if __name__ == "__main__":
    print("--- coarse-to-fine MT self-test ---")
    # Stub NLLB function for offline self-test
    def stub_nllb(text: str) -> str:
        return f"[NLLB-stub] {text}"
    c2f = CoarseToFineMT(stub_nllb)

    test_phrases = [
        "an toàn lao động",
        "xin chào",
        "dừng dây chuyền ngay lập tức",
        "cám ơn rất nhiều",
        "nguy hiểm cao",
    ]
    for vi in test_phrases:
        try:
            tgt, info = c2f.translate_vi_en(vi)
            print(f"  vi='{vi}'  -> '{tgt}'  via {info['path']}  (logp={info.get('mean_logprob','-'):.2f})")
        except Exception as e:
            print(f"  vi='{vi}'  ERROR: {type(e).__name__}: {e}")
            break
    print("\nstats:", c2f.stats.summary())
