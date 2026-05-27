"""
Translation memory — text-level phrase cache for cascade re-use.

Companion to utterance_cache.py. Where utterance_cache fingerprints AUDIO and
hits on exact-phrase repeats (cos >= 0.95), translation_memory fingerprints
the Vi TRANSCRIPT (text) and hits on near-paraphrases (cos >= 0.80). On a
text-level hit we **reuse the translation** (skip MT and audio synthesis if
TTS audio is also cached, otherwise re-run TTS only).

Why this exists separately from the audio cache:
  • Audio cache hits only when the *user's voice* is acoustically similar.
    Different speakers → different audio fingerprints → cache miss.
  • Translation cache hits when the *Vietnamese transcript* is similar.
    Different speakers asking the same question still hit because the ASR
    output text is the same/similar.

Combined effect: audio cache catches "same speaker repeats", translation
cache catches "many speakers say the same thing" — both pure wins in an
industrial multi-worker setting.

Fingerprint design (256-dim, computed in <2 ms):
  • Hashed n-gram bag over the Vi text (n=1..3 character n-grams), like a
    cheap MinHash. Fixed dim = 256, deterministic, locale-stable.
  • Length-normalized so short and long phrases compare reasonably.

This is intentionally not an LLM embedding — we want a sub-millisecond
deterministic fingerprint that survives ASR jitter (case, diacritic noise,
trailing punctuation) without any neural-network dependency.
"""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np


_DIM = 256
_PUNCT_RE = re.compile(r"[^\w\s\xC0-ỹ]+", flags=re.UNICODE)
_WS_RE    = re.compile(r"\s+")


def _normalize(text: str) -> str:
    if not text:
        return ""
    # Strip diacritics for the n-gram bag stays the same across light ASR jitter.
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _ngrams(s: str, n_range=(1, 2, 3)) -> list[str]:
    grams: list[str] = []
    s = f" {s} "
    for n in n_range:
        if len(s) >= n:
            grams.extend(s[i:i+n] for i in range(len(s) - n + 1))
    return grams


def text_fingerprint(text: str, dim: int = _DIM) -> np.ndarray:
    """Return a normalized dim-d vector representing the text via hashed n-grams."""
    norm = _normalize(text)
    if not norm:
        return np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    for g in _ngrams(norm):
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest(), "little")
        sign = 1.0 if (h >> 31) & 1 else -1.0
        vec[h % dim] += sign
    n = np.linalg.norm(vec)
    if n > 1e-6:
        vec /= n
    return vec


@dataclass
class TMStats:
    lookups: int = 0
    hits:    int = 0
    misses:  int = 0
    insertions: int = 0
    evictions:  int = 0
    fp_ms_total: float = 0.0
    fp_calls:    int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.lookups, 1)

    @property
    def fp_ms_mean(self) -> float:
        return self.fp_ms_total / max(self.fp_calls, 1)


class TranslationMemory:
    """LRU text-level retrieval cache.

    Stored value per key: dict(vi_text, tgt_text, target_lang, audio_optional).
    Lookup returns the best cosine match above threshold, plus the cache
    entry so caller can decide to (a) reuse vi+tgt and re-run TTS, or (b)
    reuse vi+tgt+audio fully.
    """

    def __init__(self, max_entries: int = 256, cosine_hit: float = 0.80):
        self.max_entries = max_entries
        self.cosine_hit  = cosine_hit
        self._store: OrderedDict[int, dict] = OrderedDict()
        self._next_id   = 0
        self.stats      = TMStats()

    def _fp(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        v = text_fingerprint(text)
        self.stats.fp_calls += 1
        self.stats.fp_ms_total += (time.perf_counter() - t0) * 1000
        return v

    def lookup(self, vi_text: str, target_lang: str) -> Optional[dict]:
        self.stats.lookups += 1
        if not vi_text or not vi_text.strip():
            self.stats.misses += 1
            return None
        fp = self._fp(vi_text)
        best_id, best_score = None, -1.0
        for k, e in self._store.items():
            if e["target_lang"] != target_lang:
                continue
            score = float(np.dot(fp, e["fingerprint"]))
            if score > best_score:
                best_score, best_id = score, k
        if best_id is not None and best_score >= self.cosine_hit:
            entry = self._store.pop(best_id)
            self._store[best_id] = entry        # LRU move-to-end
            entry["hits"] += 1
            self.stats.hits += 1
            return {**entry, "cosine": best_score}
        self.stats.misses += 1
        return None

    def put(self, vi_text: str, tgt_text: str, target_lang: str,
            audio: Optional[tuple] = None) -> None:
        if not vi_text or not vi_text.strip():
            return
        fp = self._fp(vi_text)
        self._next_id += 1
        self._store[self._next_id] = {
            "fingerprint": fp,
            "vi_text":     vi_text,
            "tgt_text":    tgt_text,
            "target_lang": target_lang,
            "audio":       audio,
            "hits":        0,
            "inserted":    time.time(),
        }
        self.stats.insertions += 1
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self.stats.evictions += 1

    def summary(self) -> dict:
        return {
            "n_entries":  len(self._store),
            "max_entries": self.max_entries,
            "cosine_hit": self.cosine_hit,
            "lookups":    self.stats.lookups,
            "hits":       self.stats.hits,
            "misses":     self.stats.misses,
            "hit_rate":   round(self.stats.hit_rate, 4),
            "insertions": self.stats.insertions,
            "evictions":  self.stats.evictions,
            "fp_ms_mean": round(self.stats.fp_ms_mean, 3),
        }


# Self-test
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tm = TranslationMemory(max_entries=8, cosine_hit=0.80)
    samples = [
        ("an toàn lao động",                "occupational safety"),
        ("xin chào các bạn",                "hello everyone"),
        ("dừng dây chuyền ngay",            "stop the production line now"),
        ("tôi đi học, sau đó về nhà",       "I went to school, then went home"),
    ]
    for vi, tg in samples:
        tm.put(vi, tg, "en", audio=None)
        h = tm.lookup(vi, "en")
        print(f"  store={vi!r}   self-lookup cos={h['cosine']:.3f}   reused={h['tgt_text']!r}")

    # Near-paraphrase tests
    print("\n--- near-paraphrase lookups ---")
    queries = [
        "an toan lao dong",                              # diacritic-stripped (ASR may produce)
        "an toàn lao động!",                              # trailing punctuation
        "an toàn",                                        # truncated
        "xin chào bạn",                                   # similar but shorter
        "stop now please",                                # totally unrelated
        "tôi đi học và sau đó về nhà",                    # paraphrased "và" added
    ]
    for q in queries:
        h = tm.lookup(q, "en")
        if h:
            print(f"  q={q!r:<45} HIT cos={h['cosine']:.3f}  -> {h['tgt_text']!r}")
        else:
            print(f"  q={q!r:<45} miss")
    print()
    print("stats:", tm.summary())
