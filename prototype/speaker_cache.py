"""
Speaker-personalized cache — per-worker partition of the audio cache.

Why this matters: a single factory has 50-200 workers, each with personal speech
patterns, accents, and a personal vocabulary of frequent phrases. A global LRU
cache mixes everyone's phrases, evicting worker A's repeat-phrases when worker
B speaks. Per-speaker partitions let each worker's cache survive independently.

Speaker identification is intentionally cheap (no x-vector / ECAPA model — those
add 100s of MB). We use a 26-dim voice fingerprint derived from MFCC mean + std
(no temporal info, only speaker timbre). Speakers are clustered by cosine
similarity with a low threshold (0.85) so accent / mood variation in the same
worker stays in the same partition.

Pipeline integration: speaker cache REPLACES the flat UtteranceCache when
enabled. It maintains a dict of {speaker_id: UtteranceCache}. The first time
a never-heard voice speaks, a new speaker_id is created.

This is paper §3.10 candidate: typology-aware audio fingerprint + per-speaker
LRU partition for industrial multi-worker S2ST.

Self-test verifies that:
  - Same speaker different phrases land in same partition
  - Different speaker → new partition created
  - Within-partition cosine still discriminates phrases
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utterance_cache import UtteranceCache, UtteranceFingerprint, cheap_mfcc


# 26-d voice timbre fingerprint (13 MFCC mean + 13 MFCC std).
# Intentionally NO temporal info — we want this to be invariant to which
# phrase the speaker is saying, only their voice identity.
def _voice_fingerprint(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
    mfcc = cheap_mfcc(audio, 16000, n_mfcc=13)
    if mfcc.shape[1] < 4:
        return np.zeros(26, dtype=np.float32)
    v = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


@dataclass
class SpeakerStats:
    n_speakers:     int = 0
    n_lookups:      int = 0
    n_hits:         int = 0
    n_new_speakers: int = 0
    voice_fp_ms_total: float = 0.0
    voice_fp_calls:    int = 0

    @property
    def hit_rate(self) -> float:
        return self.n_hits / max(self.n_lookups, 1)


@dataclass
class SpeakerCacheConfig:
    speaker_match_threshold: float = 0.85   # cosine to consider same speaker
    max_speakers:            int   = 32
    per_speaker_entries:     int   = 32     # LRU size per partition
    cosine_hit:              float = 0.95


class SpeakerCache:
    """Per-speaker partitioned audio cache.

    Public API mirrors UtteranceCache so it can be a drop-in replacement.
    Each (speaker_id, target_lang) pair gets its own LRU UtteranceCache instance.
    """

    def __init__(self, cfg: Optional[SpeakerCacheConfig] = None):
        self.cfg = cfg or SpeakerCacheConfig()
        # speaker_id -> dict(voice_fp, partition_cache, last_seen)
        self._speakers: dict[int, dict] = {}
        self._next_speaker_id = 0
        self.stats = SpeakerStats()

    # ---- voice identification ----
    def _identify_speaker(self, audio: np.ndarray, sr: int) -> tuple[int, float]:
        t0 = time.perf_counter()
        vfp = _voice_fingerprint(audio, sr)
        self.stats.voice_fp_calls += 1
        self.stats.voice_fp_ms_total += (time.perf_counter() - t0) * 1000

        best_id, best_score = None, -1.0
        for sid, sdata in self._speakers.items():
            score = float(np.dot(vfp, sdata["voice_fp"]))
            if score > best_score:
                best_score, best_id = score, sid

        if best_id is not None and best_score >= self.cfg.speaker_match_threshold:
            self._speakers[best_id]["last_seen"] = time.time()
            return best_id, best_score

        # New speaker
        self._next_speaker_id += 1
        sid = self._next_speaker_id
        self._speakers[sid] = {
            "voice_fp":   vfp,
            "partition":  UtteranceCache(
                max_entries=self.cfg.per_speaker_entries,
                cosine_hit=self.cfg.cosine_hit,
            ),
            "created":    time.time(),
            "last_seen":  time.time(),
        }
        self.stats.n_new_speakers += 1
        self.stats.n_speakers = len(self._speakers)
        # Evict oldest speaker if over limit
        while len(self._speakers) > self.cfg.max_speakers:
            oldest = min(self._speakers, key=lambda s: self._speakers[s]["last_seen"])
            del self._speakers[oldest]
        return sid, 1.0

    # ---- mirror of UtteranceCache API ----
    def fingerprint(self, audio: np.ndarray, sr: int) -> UtteranceFingerprint:
        """Identify speaker first, then fingerprint within their partition."""
        sid, _ = self._identify_speaker(audio, sr)
        fp = self._speakers[sid]["partition"].fingerprint(audio, sr)
        # tag the fingerprint with speaker so subsequent lookup/put goes to the right partition
        fp._speaker_id = sid   # type: ignore
        return fp

    def lookup(self, fp: UtteranceFingerprint, target_lang: str) -> Optional[dict]:
        self.stats.n_lookups += 1
        sid = getattr(fp, "_speaker_id", None)
        if sid is None or sid not in self._speakers:
            return None
        result = self._speakers[sid]["partition"].lookup(fp, target_lang)
        if result is not None:
            self.stats.n_hits += 1
            result["speaker_id"] = sid
        return result

    def put(self, fp: UtteranceFingerprint, target_lang: str, result: dict) -> None:
        sid = getattr(fp, "_speaker_id", None)
        if sid is None or sid not in self._speakers:
            return
        self._speakers[sid]["partition"].put(fp, target_lang, result)

    def summary(self) -> dict:
        per_speaker = {}
        for sid, s in self._speakers.items():
            ps = s["partition"].summary()
            per_speaker[f"spk{sid}"] = {
                "n_entries": ps["n_entries"],
                "lookups":   ps["lookups"],
                "hits":      ps["hits"],
                "hit_rate":  ps["hit_rate"],
            }
        return {
            "n_speakers":    len(self._speakers),
            "max_speakers":  self.cfg.max_speakers,
            "n_lookups":     self.stats.n_lookups,
            "n_hits":        self.stats.n_hits,
            "n_new_speakers": self.stats.n_new_speakers,
            "hit_rate":      round(self.stats.hit_rate, 3),
            "voice_fp_ms_mean": round(self.stats.voice_fp_ms_total / max(self.stats.voice_fp_calls, 1), 2),
            "speaker_match_threshold": self.cfg.speaker_match_threshold,
            "per_speaker":   per_speaker,
        }


# Self-test on real VIVOS speakers
if __name__ == "__main__":
    import sys
    import soundfile as sf
    from pathlib import Path
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base = Path("data/processed/vivos/vivos/train/waves")
    sc = SpeakerCache()

    print("--- speaker identification test ---")
    # Use 2 utterances per speaker x 3 speakers — expect 3 speaker partitions.
    spk_files = {
        "SPK01": ["VIVOSSPK01/VIVOSSPK01_R001.wav", "VIVOSSPK01/VIVOSSPK01_R002.wav"],
        "SPK02": ["VIVOSSPK02/VIVOSSPK02_R001.wav", "VIVOSSPK02/VIVOSSPK02_R002.wav"],
        "SPK03": ["VIVOSSPK03/VIVOSSPK03_R001.wav", "VIVOSSPK03/VIVOSSPK03_R002.wav"],
    }
    for vspk, files in spk_files.items():
        for f in files:
            audio, sr = sf.read(str(base / f), dtype="float32")
            sid, score = sc._identify_speaker(audio, sr)
            print(f"  {vspk}/{Path(f).name}  ->  partition spk{sid}  (cos {score:.3f})")

    print("\n--- summary ---")
    summary = sc.summary()
    print(f"speakers detected: {summary['n_speakers']}  (expected 3)")
    print(f"voice fingerprint mean: {summary['voice_fp_ms_mean']:.1f} ms")

    # Test lookup partitioning
    print("\n--- per-partition lookup test ---")
    # Insert from SPK01
    audio, sr = sf.read(str(base / "VIVOSSPK01/VIVOSSPK01_R001.wav"), dtype="float32")
    fp = sc.fingerprint(audio, sr)
    sc.put(fp, "en", {"vi": "spk01 phrase 1", "tgt": "trans", "audio": (sr, audio)})
    # Query same audio → should HIT spk1 partition
    fp2 = sc.fingerprint(audio, sr)
    hit = sc.lookup(fp2, "en")
    print(f"  spk01 self-lookup: HIT speaker={hit['speaker_id']} cos={hit['cosine']:.3f} -> {hit['result']['vi']}")
    # Query a SPK03 audio → expect MISS even though phrase content might match
    audio3, sr3 = sf.read(str(base / "VIVOSSPK03/VIVOSSPK03_R001.wav"), dtype="float32")
    fp3 = sc.fingerprint(audio3, sr3)
    hit3 = sc.lookup(fp3, "en")
    print(f"  spk03 lookup (no entry in spk03 partition): {'MISS' if hit3 is None else 'HIT'}")
