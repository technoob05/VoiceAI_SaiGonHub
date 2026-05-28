"""
Translation memory v2 — two-stream hashed n-gram T2 cache with AND-gate.

Motivation (paper §6.6, Round 25): the v1 single-stream T2 fingerprint
(diacritic-stripped 256-d hashed n-gram) collides at 16.48% on 3,000 VIVOS
transcripts, because aggressive normalization (NFD + strip combining marks
+ punct collapse) lets segmentally similar but tonally / lexically distinct
forms hash close on the unit sphere.

Fix, in direct analogy to the audio two-stream voting cache (§3.5):
two parallel 256-d hashed n-gram streams keyed on the SAME text but with
different normalization:

  • Stream P — diacritic-PRESERVING. Lowercase + collapse punctuation /
    whitespace, but do NOT NFD-strip combining marks. Tone-bearing
    differences (e.g. "bay" vs "bảy", "ma" vs "má") flow into different
    bins. Threshold τ_P = 0.85.

  • Stream S — diacritic-STRIPPED. Identical to v1 normalization.
    Catches paraphrastic variation (case noise, punctuation drift, light
    function-word substitution). Threshold τ_S = 0.80.

  • Hit rule (existence-AND): return entry e iff there exists e in store
    with cos_P(q,e) ≥ τ_P AND cos_S(q,e) ≥ τ_S. Tie-break by sum.

The veto direction is the load-bearing claim: a diacritic-conflation false
hit (different tonal/lexical forms that strip to the same sequence) has
high cos_S but low cos_P, so the AND-gate vetoes it. A genuine paraphrase
(same diacritics, light surface variation) has both cos_S and cos_P high,
so it still hits.
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


def _normalize_stripped(text: str) -> str:
    """Stream S: lowercase + NFD + strip combining marks + collapse punct/ws."""
    if not text:
        return ""
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _normalize_preserving(text: str) -> str:
    """Stream P: lowercase + collapse punct/ws, keep diacritics.
    NFC-normalize so that 'á' and 'á' hash identically."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text.lower())
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _ngrams(s: str, n_range=(1, 2, 3)) -> list[str]:
    grams: list[str] = []
    s = f" {s} "
    for n in n_range:
        if len(s) >= n:
            grams.extend(s[i:i+n] for i in range(len(s) - n + 1))
    return grams


def _hash_bag(norm: str, dim: int, salt: bytes) -> np.ndarray:
    """Sign-hashed n-gram bag into dim float32 bins, L2-normalized.
    The salt parameter decorrelates the two streams' hashing so a
    coincidental bin collision in S is not echoed in P."""
    if not norm:
        return np.zeros(dim, dtype=np.float32)
    vec = np.zeros(dim, dtype=np.float32)
    for g in _ngrams(norm):
        h = int.from_bytes(
            hashlib.blake2b(g.encode("utf-8"), digest_size=4, salt=salt).digest(),
            "little",
        )
        sign = 1.0 if (h >> 31) & 1 else -1.0
        vec[h % dim] += sign
    n = np.linalg.norm(vec)
    if n > 1e-6:
        vec /= n
    return vec


_SALT_S = b"t2-strip"
_SALT_P = b"t2-keep_"


def text_fingerprint_stripped(text: str, dim: int = _DIM) -> np.ndarray:
    return _hash_bag(_normalize_stripped(text), dim, _SALT_S)


def text_fingerprint_preserving(text: str, dim: int = _DIM) -> np.ndarray:
    return _hash_bag(_normalize_preserving(text), dim, _SALT_P)


@dataclass
class TMStatsV2:
    lookups: int = 0
    hits:    int = 0
    misses:  int = 0
    insertions: int = 0
    evictions:  int = 0
    fp_ms_total: float = 0.0
    fp_calls:    int = 0
    veto_p_only: int = 0   # cos_S ≥ τ_S but cos_P < τ_P  (the diacritic veto)
    veto_s_only: int = 0   # cos_P ≥ τ_P but cos_S < τ_S  (rare; both diverge)

    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.lookups, 1)


class TranslationMemoryV2:
    """LRU two-stream text-retrieval cache with AND-gated hit rule.

    Stored value per key: dict(fp_p, fp_s, vi_text, tgt_text, target_lang,
    audio_optional). Lookup returns the entry whose (cos_P, cos_S) jointly
    clear (τ_P, τ_S), tie-breaking by max(cos_P + cos_S).
    """

    def __init__(self, max_entries: int = 256,
                 tau_p: float = 0.85, tau_s: float = 0.80):
        self.max_entries = max_entries
        self.tau_p = tau_p
        self.tau_s = tau_s
        self._store: OrderedDict[int, dict] = OrderedDict()
        self._next_id = 0
        self.stats   = TMStatsV2()

    def _fp_pair(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        p = text_fingerprint_preserving(text)
        s = text_fingerprint_stripped(text)
        self.stats.fp_calls += 1
        self.stats.fp_ms_total += (time.perf_counter() - t0) * 1000
        return p, s

    def lookup(self, vi_text: str, target_lang: str) -> Optional[dict]:
        self.stats.lookups += 1
        if not vi_text or not vi_text.strip():
            self.stats.misses += 1
            return None
        fp_p, fp_s = self._fp_pair(vi_text)

        best_id, best_score = None, -1.0
        any_s_pass = False
        any_p_pass = False
        for k, e in self._store.items():
            if e["target_lang"] != target_lang:
                continue
            cs = float(np.dot(fp_s, e["fp_s"]))
            cp = float(np.dot(fp_p, e["fp_p"]))
            s_pass = cs >= self.tau_s
            p_pass = cp >= self.tau_p
            any_s_pass |= s_pass
            any_p_pass |= p_pass
            if s_pass and p_pass:
                score = cs + cp
                if score > best_score:
                    best_score, best_id = score, k
                    best_cs, best_cp = cs, cp  # noqa: F841
        if best_id is not None:
            entry = self._store.pop(best_id)
            self._store[best_id] = entry
            entry["hits"] += 1
            self.stats.hits += 1
            return {**entry, "cos_p": float(np.dot(fp_p, entry["fp_p"])),
                    "cos_s": float(np.dot(fp_s, entry["fp_s"]))}
        # diagnostic counters: which gate did the veto come from?
        if any_s_pass and not any_p_pass:
            self.stats.veto_p_only += 1
        elif any_p_pass and not any_s_pass:
            self.stats.veto_s_only += 1
        self.stats.misses += 1
        return None

    def put(self, vi_text: str, tgt_text: str, target_lang: str,
            audio: Optional[tuple] = None) -> None:
        if not vi_text or not vi_text.strip():
            return
        fp_p, fp_s = self._fp_pair(vi_text)
        self._next_id += 1
        self._store[self._next_id] = {
            "fp_p":        fp_p,
            "fp_s":        fp_s,
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
            "n_entries":   len(self._store),
            "max_entries": self.max_entries,
            "tau_p":       self.tau_p,
            "tau_s":       self.tau_s,
            "lookups":     self.stats.lookups,
            "hits":        self.stats.hits,
            "misses":      self.stats.misses,
            "hit_rate":    round(self.stats.hit_rate, 4),
            "insertions":  self.stats.insertions,
            "evictions":   self.stats.evictions,
            "veto_p_only": self.stats.veto_p_only,
            "veto_s_only": self.stats.veto_s_only,
            "fp_ms_mean":  round(self.stats.fp_ms_total
                                  / max(self.stats.fp_calls, 1), 3),
        }


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tm = TranslationMemoryV2(max_entries=16, tau_p=0.85, tau_s=0.80)
    seeds = [
        ("an toàn lao động",          "occupational safety"),
        ("dừng dây chuyền ngay",      "stop the production line now"),
        ("tôi muốn uống nước",        "I want some water"),
        ("máy đang chạy",             "the machine is running"),
        ("cảm ơn rất nhiều",          "thank you very much"),
    ]
    for vi, en in seeds:
        tm.put(vi, en, "en", audio=None)

    print("--- true paraphrase probes (should HIT) ---")
    para = [
        "An Toàn Lao Động",
        "an toàn lao động!",
        "Dừng dây chuyền ngay.",
        "  máy đang chạy  ",
    ]
    for q in para:
        h = tm.lookup(q, "en")
        tag = (f"HIT cosP={h['cos_p']:.3f} cosS={h['cos_s']:.3f} -> {h['tgt_text']!r}"
               if h else "miss")
        print(f"  q={q!r:<40} {tag}")

    print("\n--- diacritic-conflation probes (should MISS via AND-gate) ---")
    confl = [
        "an toan lao dong",        # diacritic-stripped variant
        "may dang chay",
        "tôi muốn mưa hai cái",     # tonal-distinct from any stored
    ]
    for q in confl:
        h = tm.lookup(q, "en")
        tag = (f"HIT cosP={h['cos_p']:.3f} cosS={h['cos_s']:.3f} -> {h['tgt_text']!r}"
               if h else "miss (vetoed)")
        print(f"  q={q!r:<40} {tag}")

    print("\nstats:", tm.summary())
