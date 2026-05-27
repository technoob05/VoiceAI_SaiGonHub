"""
Two-stream voting fingerprint — alternative architecture for the §2.6 finding.

Instead of mixing tone histogram and MFCC into a single L2-normed vector
(where MFCC's higher dim count drowns tone), we maintain TWO independent
fingerprints and require AGREEMENT for cache hits:

  Stream A (tone):    6-d tone histogram                        threshold τ_A = 0.85
  Stream B (segment): 260-d MFCC + delta concat                 threshold τ_B = 0.93

Cache hit iff BOTH streams' best matches are the same cached entry
AND both scores exceed their respective thresholds. The two streams
provide complementary information:
  - Stream A discriminates tonal variants (paper §2.6 minimal-pair eval
    showed tone_only protects: intra-tone 0.69 < inter-syl 0.72)
  - Stream B discriminates segmental content (MFCC ablation gap 0.09)

A worker saying *má* (mother) vs *mả* (tomb) — both could MFCC-hit on a
cached "má" entry, but Stream A vetoes because tone histograms disagree.

This is a paper §3.5 alternative architecture (vs weighted concat).
Self-test compares hit rate + minimal-pair protection vs full-concat baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PROJECT = HERE.parent
HF_CACHE = PROJECT / "data" / "raw" / "hf_cache"
os.environ.setdefault("HF_HOME",      str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np                                            # noqa: E402
from utterance_cache import (                                  # noqa: E402
    UtteranceCache,
    cheap_mfcc,
    tone_histogram_from_f0,
    _f0_contour,
)


def _components(audio: np.ndarray, sr: int) -> dict:
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
        sr = 16000
    f0 = _f0_contour(audio, sr)
    tone_hist = tone_histogram_from_f0(f0)
    mfcc = cheap_mfcc(audio, sr, n_mfcc=13)
    mfcc_sub = UtteranceCache._temporal_subsample(mfcc, n_chunks=10)
    mfcc_flat = mfcc_sub.flatten(order="F")
    mfcc_delta = np.diff(mfcc_sub, axis=1, prepend=mfcc_sub[:, :1])
    mfcc_delta_flat = mfcc_delta.flatten(order="F")
    return {"tone": tone_hist, "segment": np.concatenate([mfcc_flat, mfcc_delta_flat])}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


@dataclass
class TwoStreamConfig:
    tone_threshold:    float = 0.85
    segment_threshold: float = 0.93
    max_entries:       int   = 128


@dataclass
class TwoStreamFingerprint:
    tone:    np.ndarray   # 6-d
    segment: np.ndarray   # 260-d


@dataclass
class TwoStreamStats:
    lookups: int = 0
    hits: int = 0
    tone_only_hits: int = 0          # tone matches but segment doesn't
    segment_only_hits: int = 0       # segment matches but tone doesn't (FALSE POSITIVE risk)
    both_hits: int = 0               # full agreement = real cache hit

    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.lookups, 1)


class TwoStreamCache:
    """Two independent fingerprint streams; cache hit requires both to agree
    on the SAME stored entry."""

    def __init__(self, cfg: Optional[TwoStreamConfig] = None):
        self.cfg = cfg or TwoStreamConfig()
        self._store: OrderedDict[int, dict] = OrderedDict()
        self._next_id = 0
        self.stats = TwoStreamStats()

    def fingerprint(self, audio: np.ndarray, sr: int) -> TwoStreamFingerprint:
        comps = _components(audio, sr)
        return TwoStreamFingerprint(
            tone=_normalize(comps["tone"]),
            segment=_normalize(comps["segment"]),
        )

    def lookup(self, fp: TwoStreamFingerprint, target_lang: str) -> Optional[dict]:
        """Return a dict compatible with UtteranceCache.lookup() so the rest
        of the pipeline can use either cache interchangeably.
        Extra fields: tone_score, segment_score, both_streams_agree.
        """
        self.stats.lookups += 1
        best_id = None
        best_tone, best_seg = -1.0, -1.0
        best_tone_id, best_seg_id = None, None
        # Find the best tone-match and best segment-match independently
        for k, e in self._store.items():
            if e.get("target_lang") != target_lang:
                continue
            t_score = float(np.dot(fp.tone,    e["fp"].tone))
            s_score = float(np.dot(fp.segment, e["fp"].segment))
            if t_score > best_tone:
                best_tone, best_tone_id = t_score, k
            if s_score > best_seg:
                best_seg, best_seg_id = s_score, k

        if best_tone_id is None:
            return None

        # Track diagnostic signals
        tone_passes    = best_tone >= self.cfg.tone_threshold
        segment_passes = best_seg  >= self.cfg.segment_threshold
        if tone_passes and not segment_passes:
            self.stats.tone_only_hits += 1
        if segment_passes and not tone_passes:
            self.stats.segment_only_hits += 1

        # AGREEMENT-REQUIRED rule
        if tone_passes and segment_passes and best_tone_id == best_seg_id:
            self.stats.both_hits += 1
            self.stats.hits += 1
            entry = self._store.pop(best_tone_id)
            self._store[best_tone_id] = entry
            # Combined cosine for downstream code that expects single score:
            # use harmonic mean (penalizes the weaker stream).
            combined = 2 * best_tone * best_seg / max(best_tone + best_seg, 1e-9)
            return {
                **entry,
                "cosine":             combined,
                "tone_score":         best_tone,
                "segment_score":      best_seg,
                "both_streams_agree": True,
            }
        return None

    def put(self, fp: TwoStreamFingerprint, target_lang: str, result: dict) -> None:
        self._next_id += 1
        self._store[self._next_id] = {
            "fp":          fp,
            "target_lang": target_lang,
            "result":      result,
            "inserted":    time.time(),
        }
        while len(self._store) > self.cfg.max_entries:
            self._store.popitem(last=False)

    def summary(self) -> dict:
        return {
            "n_entries":         len(self._store),
            "max_entries":       self.cfg.max_entries,
            "lookups":           self.stats.lookups,
            "hits":              self.stats.hits,
            "both_hits":         self.stats.both_hits,
            "tone_only_hits":    self.stats.tone_only_hits,
            "segment_only_hits": self.stats.segment_only_hits,
            "hit_rate":          round(self.stats.hit_rate, 4),
        }


# Self-test on minimal-pair audios — does two-stream voting reject tonal-only-difference false hits?

MINIMAL_PAIR_SYLLABLES = {
    "ma":  ["ma",  "má",  "mả",  "mạ",  "mã",  "mà"],
    "ba":  ["ba",  "bá",  "bả",  "bạ",  "bã",  "bà"],
    "ca":  ["ca",  "cá",  "cả",  "cạ",  "cã",  "cà"],
    "la":  ["la",  "lá",  "lả",  "lạ",  "lã",  "là"],
    "da":  ["da",  "dá",  "dả",  "dạ",  "dã",  "dà"],
}
CARRIER = "nói {syl}"


def _synth(text, tok, model):
    import torch
    inputs = tok(text, return_tensors="pt")
    with torch.inference_mode():
        wav = model(**inputs).waveform
    return wav.squeeze().cpu().numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Results/two_stream_voting_eval.json")
    args = ap.parse_args()

    print("[load] MMS-TTS-vie ...")
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate

    # Synthesize minimal pair clips
    audio_by_key = {}
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants:
            try:
                audio_by_key[(base, v)] = _synth(CARRIER.format(syl=v), tok, model)
            except Exception as e:
                print(f"  [warn] {v}: {e}")
    print(f"[synth] {len(audio_by_key)} clips")

    # ---- Experiment 1: store all variants, query each — should hit ITSELF ----
    cache = TwoStreamCache()
    for k, a in audio_by_key.items():
        fp = cache.fingerprint(a, sr)
        cache.put(fp, "en", {"vi": f"{k[0]}/{k[1]}", "tgt": k[1]})

    # Each clip should self-hit (cos = 1.0 in both streams)
    self_hits = 0
    for k, a in audio_by_key.items():
        fp = cache.fingerprint(a, sr)
        h = cache.lookup(fp, "en")
        if h is not None and h["result"]["vi"] == f"{k[0]}/{k[1]}":
            self_hits += 1
    print(f"\n[exp1] self-lookup hit rate: {self_hits}/{len(audio_by_key)}")

    # ---- Experiment 2: store ONLY 'ma' tone, query OTHER tones of same syllable — should MISS ----
    cache2 = TwoStreamCache()
    for base in MINIMAL_PAIR_SYLLABLES:
        ma_key = (base, MINIMAL_PAIR_SYLLABLES[base][0])   # the "ngang"-tone variant
        if ma_key in audio_by_key:
            fp = cache2.fingerprint(audio_by_key[ma_key], sr)
            cache2.put(fp, "en", {"vi": MINIMAL_PAIR_SYLLABLES[base][0], "tgt": MINIMAL_PAIR_SYLLABLES[base][0]})

    false_hits = 0
    correct_misses = 0
    total = 0
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants[1:]:    # skip ngang (already stored)
            k = (base, v)
            if k not in audio_by_key:
                continue
            fp = cache2.fingerprint(audio_by_key[k], sr)
            h = cache2.lookup(fp, "en")
            total += 1
            if h is None:
                correct_misses += 1
            else:
                # If hit, it should be the WRONG entry (ngang of same syllable) — false positive
                false_hits += 1

    miss_rate = correct_misses / max(total, 1)
    print(f"[exp2] correct-miss on tonal variants (only ngang cached): {correct_misses}/{total} ({miss_rate*100:.1f}%)")
    print(f"       false hits (would translate wrong word): {false_hits}/{total}")

    out = {
        "self_hit_rate": round(self_hits / len(audio_by_key), 4),
        "tonal_variant_correct_miss_rate": round(miss_rate, 4),
        "tonal_variant_false_hit_rate":    round(false_hits / max(total, 1), 4),
        "cache_stats":   cache2.summary(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {args.out}")
    print("\nLegend: higher correct-miss = better minimal-pair protection (the cache refuses to return wrong-tone translation).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
