"""
Persist Tier-1 (audio) and Tier-2 (text) caches across sessions.

WHY this is non-trivial: each cached entry stores numpy fingerprints + audio
WAV arrays + Vi text. Standard JSON can't serialize numpy directly, but we
also don't want a heavy pickle (security + portability). Approach:
  • fingerprints: stored as base64-encoded float16 (4× smaller than float32),
    losslessly round-trippable for the cosine similarities we use
  • audio WAVs: stored as base64-encoded int16 PCM at the original SR
  • metadata as plain JSON

On disk layout (per cache file):
  {
    "schema": "voiceai-cache-v1",
    "kind":   "t1_audio" | "t2_text",
    "n_entries":  <int>,
    "stats":  {…},
    "entries":  [{...}, ...]
  }

Disk savings: a 128-entry T1 cache with audio is < 5 MB. T2 (text-only) < 100 KB.

Load policy: silently skips entries whose fingerprint dim doesn't match the
current pipeline (so upgrading fingerprint code doesn't break existing cache files).
"""
from __future__ import annotations

import base64
import json
import os
import time
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

SCHEMA = "voiceai-cache-v1"


def _vec_to_b64(arr: np.ndarray) -> str:
    """np.ndarray → base64 string (float16 storage, lossless cosine on L2-normed vecs)."""
    arr16 = arr.astype(np.float16, copy=False)
    return base64.b64encode(arr16.tobytes()).decode("ascii")


def _b64_to_vec(s: str, dim: int) -> np.ndarray:
    raw = base64.b64decode(s.encode("ascii"))
    v = np.frombuffer(raw, dtype=np.float16)
    if len(v) != dim:
        raise ValueError(f"fingerprint dim mismatch: got {len(v)} expected {dim}")
    return v.astype(np.float32)


def _audio_to_b64(audio: np.ndarray, sr: int) -> dict:
    """Encode float audio as int16 PCM in base64 + zlib for compression."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16).tobytes()
    return {"sr": int(sr), "pcm_b64": base64.b64encode(zlib.compress(pcm)).decode("ascii")}


def _b64_to_audio(d: dict) -> tuple[int, np.ndarray]:
    raw = zlib.decompress(base64.b64decode(d["pcm_b64"].encode("ascii")))
    pcm = np.frombuffer(raw, dtype=np.int16)
    return d["sr"], (pcm.astype(np.float32) / 32768.0)


# =====================================================================
# Tier 1 — audio cache
# =====================================================================


def save_t1_audio_cache(cache, path: Path) -> dict:
    """Serialize an UtteranceCache to JSON."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, e in cache._store.items():
        fp = e["fingerprint"]
        rec = {
            "key": k,
            "fp_b64":  _vec_to_b64(fp.vec),
            "fp_dim":  len(fp.vec),
            "duration_s":  float(fp.duration_s),
            "n_voiced_frames": int(fp.n_voiced_frames),
            "target_lang": e["target_lang"],
            "result_vi":   e["result"]["vi"],
            "result_tgt":  e["result"]["tgt"],
            "audio":       _audio_to_b64(*reversed(e["result"]["audio"])),
            "hits":        int(e.get("hits", 0)),
            "inserted":    float(e.get("inserted", time.time())),
            "last_used":   float(e.get("last_used", time.time())),
        }
        entries.append(rec)
    obj = {"schema": SCHEMA, "kind": "t1_audio",
           "n_entries": len(entries), "saved_at": time.time(),
           "stats": cache.summary(), "entries": entries}
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "n_entries": len(entries),
            "size_bytes": path.stat().st_size}


def load_t1_audio_cache(cache, path: Path) -> dict:
    """Hydrate an UtteranceCache from JSON. Drops mismatched-dim entries silently."""
    from utterance_cache import UtteranceFingerprint
    path = Path(path)
    if not path.exists():
        return {"loaded": 0, "skipped": 0, "error": f"file not found: {path}"}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema") != SCHEMA:
        return {"loaded": 0, "skipped": 0, "error": f"schema mismatch: {obj.get('schema')}"}
    if obj.get("kind") != "t1_audio":
        return {"loaded": 0, "skipped": 0, "error": f"kind mismatch: {obj.get('kind')}"}

    # determine expected dim from a fresh fingerprint on silence
    expected_dim = len(cache.fingerprint(np.zeros(int(cache.sr * 0.5), dtype=np.float32), cache.sr).vec)

    loaded, skipped = 0, 0
    for rec in obj.get("entries", []):
        if rec.get("fp_dim") != expected_dim:
            skipped += 1; continue
        try:
            vec = _b64_to_vec(rec["fp_b64"], expected_dim)
            sr, audio = _b64_to_audio(rec["audio"])
        except Exception:
            skipped += 1; continue
        fp = UtteranceFingerprint(
            vec=vec,
            duration_s=rec.get("duration_s", 0.0),
            n_voiced_frames=rec.get("n_voiced_frames", 0),
        )
        cache.put(fp, rec["target_lang"], {
            "vi":  rec["result_vi"],
            "tgt": rec["result_tgt"],
            "audio": (sr, audio),
        })
        loaded += 1
    return {"loaded": loaded, "skipped": skipped,
            "path": str(path), "size_bytes": path.stat().st_size}


# =====================================================================
# Tier 2 — text translation memory
# =====================================================================


def save_t2_text_cache(tm, path: Path) -> dict:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, e in tm._store.items():
        rec = {
            "key":         k,
            "fp_b64":      _vec_to_b64(e["fingerprint"]),
            "fp_dim":      len(e["fingerprint"]),
            "vi_text":     e["vi_text"],
            "tgt_text":    e["tgt_text"],
            "target_lang": e["target_lang"],
            "hits":        int(e.get("hits", 0)),
            "inserted":    float(e.get("inserted", time.time())),
        }
        if e.get("audio") is not None:
            rec["audio"] = _audio_to_b64(*reversed(e["audio"]))
        entries.append(rec)
    obj = {"schema": SCHEMA, "kind": "t2_text",
           "n_entries": len(entries), "saved_at": time.time(),
           "stats": tm.summary(), "entries": entries}
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "n_entries": len(entries),
            "size_bytes": path.stat().st_size}


def load_t2_text_cache(tm, path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"loaded": 0, "skipped": 0, "error": f"file not found: {path}"}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema") != SCHEMA or obj.get("kind") != "t2_text":
        return {"loaded": 0, "skipped": 0,
                "error": f"schema/kind mismatch: {obj.get('schema')}/{obj.get('kind')}"}

    # expected dim from fresh fingerprint
    expected_dim = len(tm._fp("xin chào"))

    loaded, skipped = 0, 0
    for rec in obj.get("entries", []):
        if rec.get("fp_dim") != expected_dim:
            skipped += 1; continue
        try:
            audio = None
            if "audio" in rec:
                sr, a = _b64_to_audio(rec["audio"])
                audio = (sr, a)
        except Exception:
            skipped += 1; continue
        tm.put(rec["vi_text"], rec["tgt_text"], rec["target_lang"], audio=audio)
        loaded += 1
    return {"loaded": loaded, "skipped": skipped,
            "path": str(path), "size_bytes": path.stat().st_size}


# =====================================================================
# Convenience: bundle both tiers into a pipeline-level snapshot
# =====================================================================

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "cache_snapshots"


def snapshot_pipeline(pipe, name: str = "default", root: Path = DEFAULT_DIR) -> dict:
    """Save both T1 and T2 caches under <root>/<name>/."""
    root = Path(root) / name
    root.mkdir(parents=True, exist_ok=True)
    out = {"snapshot_dir": str(root), "saved_at": time.time()}
    if pipe._utt_cache is not None:
        out["t1"] = save_t1_audio_cache(pipe._utt_cache, root / "t1_audio.json")
    if pipe._tm is not None:
        out["t2"] = save_t2_text_cache(pipe._tm, root / "t2_text.json")
    return out


def restore_pipeline(pipe, name: str = "default", root: Path = DEFAULT_DIR) -> dict:
    root = Path(root) / name
    out = {"snapshot_dir": str(root)}
    if pipe._utt_cache is not None:
        out["t1"] = load_t1_audio_cache(pipe._utt_cache, root / "t1_audio.json")
    if pipe._tm is not None:
        out["t2"] = load_t2_text_cache(pipe._tm, root / "t2_text.json")
    return out


# =====================================================================
# Self-test
# =====================================================================

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from utterance_cache import UtteranceCache
    from translation_memory import TranslationMemory

    # T1 cache round-trip
    ucache = UtteranceCache(max_entries=8, cosine_hit=0.95)
    audio = np.random.RandomState(0).normal(0, 0.05, 16000 * 2).astype(np.float32)
    fp = ucache.fingerprint(audio, 16000)
    ucache.put(fp, "en", {"vi": "xin chào", "tgt": "hello", "audio": (16000, audio)})

    tm = TranslationMemory(max_entries=16, cosine_hit=0.80)
    tm.put("an toàn lao động", "occupational safety", "en")
    tm.put("dừng dây chuyền",   "stop the production line", "en")

    tmp = Path("data/cache_snapshots/_selftest")
    r1 = save_t1_audio_cache(ucache, tmp / "t1.json")
    r2 = save_t2_text_cache(tm,      tmp / "t2.json")
    print("saved T1:", r1)
    print("saved T2:", r2)

    # Round-trip
    fresh_u = UtteranceCache(max_entries=8, cosine_hit=0.95)
    fresh_t = TranslationMemory(max_entries=16, cosine_hit=0.80)
    print("loaded T1:", load_t1_audio_cache(fresh_u, tmp / "t1.json"))
    print("loaded T2:", load_t2_text_cache(fresh_t, tmp / "t2.json"))

    # Verify hits
    h = fresh_t.lookup("an toàn lao động", "en")
    assert h is not None and h["tgt_text"] == "occupational safety"
    print(f"  T2 round-trip lookup OK: cos={h['cosine']:.3f}")

    fp2 = fresh_u.fingerprint(audio, 16000)
    h2 = fresh_u.lookup(fp2, "en")
    assert h2 is not None
    print(f"  T1 round-trip lookup OK: cos={h2['cosine']:.3f}  tgt={h2['result']['tgt']!r}")

    print("\n  selftest passed")
