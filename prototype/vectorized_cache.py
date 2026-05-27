"""
VectorizedUtteranceCache — drop-in subclass of UtteranceCache that replaces the
Python `for entry in self._store.items()` comparison loop with a single
(N x D) @ (D,) matmul.

Motivation (from Results/cache_scalability_bench.json):
  - At N=1024 the baseline comparison loop is ~7.79 ms p50 / 9.67 ms p95.
  - The fingerprint cost (~77 ms p50) still dominates total latency, but the
    comparison loop is pure-Python overhead that we get back "for free" by
    delegating one matmul to BLAS.

Mechanism:
  - Maintain `_fp_matrix: np.ndarray of shape (N, D)` alongside `_store`.
  - Maintain `_ids: list[int]` row-aligned to `_fp_matrix` (so row i ↔ id `_ids[i]`).
  - Maintain `_target_langs: np.ndarray` of str row-aligned (used as a mask).
  - lookup(): scores = _fp_matrix @ fp.vec   (one BLAS call)
              mask out rows whose target_lang != query target_lang
              argmax, compare to cosine threshold, mutate LRU on hit.
  - put(): append a row; on overflow, pop row 0 + sync `_store.popitem(last=False)`.
  - LRU on hit: row stays in place (we identify by id, not row index, and the
    row-index → id mapping is updated only on put/evict). This keeps the matrix
    contiguous and avoids O(N) reshuffles per hit.

Correctness invariant:
  After every put() / lookup() / eviction, for every i in range(len(_ids)):
      _fp_matrix[i] == _store[_ids[i]]["fingerprint"].vec
      _target_langs[i] == _store[_ids[i]]["target_lang"]
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from utterance_cache import (
    UtteranceCache,
    UtteranceFingerprint,
)


FP_DIM = 272


class VectorizedUtteranceCache(UtteranceCache):
    """Same public API as UtteranceCache; comparison loop replaced by a matmul."""

    def __init__(self, max_entries: int = 128, cosine_hit: float = 0.95,
                 min_duration_s: float = 0.5, sr: int = 16000,
                 fp_dim: int = FP_DIM):
        super().__init__(max_entries=max_entries, cosine_hit=cosine_hit,
                         min_duration_s=min_duration_s, sr=sr)
        self._fp_dim = fp_dim
        # Pre-allocate to max_entries; track logical length via _n_rows.
        self._fp_matrix: np.ndarray = np.zeros((max_entries, fp_dim), dtype=np.float32)
        self._ids: list[int] = []
        self._target_langs: list[str] = []

    # ---- put ----
    def put(self, fp: UtteranceFingerprint, target_lang: str, result: dict) -> None:
        if fp.duration_s < self.min_duration_s:
            return
        self._next_id += 1
        entry_id = self._next_id
        self._store[entry_id] = {
            "fingerprint": fp,
            "target_lang": target_lang,
            "result":      result,
            "last_used":   time.time(),
            "hits":        0,
            "inserted":    time.time(),
        }
        self.stats.insertions += 1

        # Grow matrix if we somehow exceed pre-allocation (defensive).
        if len(self._ids) >= self._fp_matrix.shape[0]:
            grown = np.zeros((self._fp_matrix.shape[0] * 2, self._fp_dim), dtype=np.float32)
            grown[:len(self._ids)] = self._fp_matrix[:len(self._ids)]
            self._fp_matrix = grown

        self._fp_matrix[len(self._ids)] = fp.vec
        self._ids.append(entry_id)
        self._target_langs.append(target_lang)

        # Evict oldest while over capacity (mirror parent semantics).
        while len(self._store) > self.max_entries:
            old_id, _ = self._store.popitem(last=False)
            self.stats.evictions += 1
            # The oldest entry is at the head of the row mapping; row-shift up by one.
            row_idx = self._ids.index(old_id)   # usually 0; robust against re-ordering
            if row_idx == 0:
                # Cheap path: shift everything up by one.
                n = len(self._ids)
                self._fp_matrix[:n-1] = self._fp_matrix[1:n]
                self._fp_matrix[n-1].fill(0.0)
            else:
                # Defensive: shift the trailing rows up to fill the gap.
                n = len(self._ids)
                self._fp_matrix[row_idx:n-1] = self._fp_matrix[row_idx+1:n]
                self._fp_matrix[n-1].fill(0.0)
            self._ids.pop(row_idx)
            self._target_langs.pop(row_idx)

    # ---- lookup ----
    def lookup(self, fp: UtteranceFingerprint, target_lang: str) -> Optional[dict]:
        self.stats.lookups += 1
        if fp.duration_s < self.min_duration_s or fp.n_voiced_frames < 4:
            self.stats.misses += 1
            return None
        n = len(self._ids)
        if n == 0:
            self.stats.misses += 1
            return None

        # Single matmul over the live slab.
        scores = self._fp_matrix[:n] @ fp.vec        # shape (n,)

        # Mask out rows with mismatched target_lang. Building a numpy array
        # of str is O(n) but cheap; alternative is to keep a parallel ndarray.
        tl_arr = np.asarray(self._target_langs)
        mask = (tl_arr == target_lang)
        if not mask.any():
            self.stats.misses += 1
            return None
        masked = np.where(mask, scores, -np.inf)
        best_row = int(np.argmax(masked))
        best_score = float(masked[best_row])

        if best_score >= self.cosine_hit:
            best_id = self._ids[best_row]
            entry = self._store.pop(best_id)
            self._store[best_id] = entry            # LRU move-to-end on _store
            entry["last_used"] = time.time()
            entry["hits"] += 1
            self.stats.hits += 1
            return {**entry, "cosine": best_score}
        self.stats.misses += 1
        return None

    # ---- introspection ----
    def matrix_shape(self) -> tuple[int, int]:
        return (len(self._ids), self._fp_dim)


# ---------------- Self-test ----------------
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rng = np.random.default_rng(0)

    # Equivalence test: feed the same audio sequence to both caches, verify hits match.
    from utterance_cache import UtteranceCache as BaseCache

    base = BaseCache(max_entries=8, cosine_hit=0.90)
    vec  = VectorizedUtteranceCache(max_entries=8, cosine_hit=0.90)

    sig1 = (rng.normal(size=16000) * 0.1).astype(np.float32)
    sig2 = (rng.normal(size=20000) * 0.1).astype(np.float32)
    sig3 = (rng.normal(size=12000) * 0.1).astype(np.float32)
    sig1_again = sig1 + (rng.normal(size=16000) * 0.005).astype(np.float32)

    for name, sig in [("ut1", sig1), ("ut2", sig2), ("ut3", sig3), ("ut1_jitter", sig1_again)]:
        fp_b = base.fingerprint(sig, sr=16000)
        fp_v = vec.fingerprint(sig, sr=16000)
        assert np.allclose(fp_b.vec, fp_v.vec, atol=1e-7), f"fingerprints diverged on {name}"

        hit_b = base.lookup(fp_b, target_lang="en")
        hit_v = vec.lookup(fp_v, target_lang="en")

        # Both should agree on hit/miss
        assert (hit_b is None) == (hit_v is None), f"hit/miss disagreement on {name}: base={hit_b is not None} vec={hit_v is not None}"
        if hit_b is not None:
            assert abs(hit_b["cosine"] - hit_v["cosine"]) < 1e-5, f"cosine disagreement on {name}"
            print(f"  {name}: HIT cosine={hit_v['cosine']:.4f}  OK")
        else:
            base.put(fp_b, target_lang="en", result={"vi": f"transcript_{name}"})
            vec.put(fp_v,  target_lang="en", result={"vi": f"transcript_{name}"})
            print(f"  {name}: miss -> cached  matrix={vec.matrix_shape()}")

    # Eviction equivalence test: fill past max_entries.
    for i in range(12):
        s = (rng.normal(size=16000 + i*100) * 0.1).astype(np.float32)
        fp_b = base.fingerprint(s, sr=16000)
        fp_v = vec.fingerprint(s, sr=16000)
        base.put(fp_b, target_lang="en", result={"vi": f"extra_{i}"})
        vec.put(fp_v,  target_lang="en", result={"vi": f"extra_{i}"})

    assert len(base) == len(vec) == 8, f"size mismatch: base={len(base)}, vec={len(vec)}"
    assert vec.matrix_shape() == (8, FP_DIM), f"matrix shape wrong: {vec.matrix_shape()}"
    # Same ids should remain (LRU semantics identical)
    base_ids = list(base._store.keys())
    vec_ids  = list(vec._store.keys())
    assert base_ids == vec_ids, f"id ordering diverged: base={base_ids}, vec={vec_ids}"
    print(f"\n  eviction OK: both caches have ids {base_ids}")
    print(f"\n  base summary: {base.summary()}")
    print(f"  vec  summary: {vec.summary()}")
    print("  matrix shape:", vec.matrix_shape())
    print("\nself-test PASSED")
