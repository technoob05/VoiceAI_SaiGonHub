"""
PQ-compressed Two-Stream cache — memory-efficient variant of TwoStreamCache.

At 1000+ cached entries the raw segment stream costs 260 * 4 = 1040 bytes
per entry (~1 MB total). Product quantization with M=13 sub-quantizers and
K=256 codes drops that to 13 bytes — ~80x memory reduction. The tone stream
stays raw as float16 (6 * 2 = 12 bytes), since it's already tiny and tone
loss directly defeats the §2.6 minimal-pair purpose.

Interface mirrors TwoStreamCache (prototype/two_stream_voting.py):
  - .fingerprint(audio, sr) -> TwoStreamFingerprint
  - .lookup(fp, target_lang)  -> dict | None  (cosine, result, ...)
  - .put(fp, target_lang, result)

PQ training: first N=200 segments inserted before quantization are buffered
raw; once the buffer hits the train_size or a lookup is requested with the
buffer full, the PQ is trained on the buffer, all buffered fingerprints are
encoded, and from then on every put encodes immediately. Tone stream is
stored as float16 always.

Asymmetric cosine: the query stays float32; the stored entry is a (M,)
uint8 code; ProductQuantizer.asymmetric_cosine gives an O(M) score.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
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

from two_stream_voting import (                                # noqa: E402
    TwoStreamFingerprint,
    TwoStreamStats,
    _components,
    _normalize,
)
from pq_compress import ProductQuantizer, PQConfig            # noqa: E402


@dataclass
class PQTwoStreamConfig:
    tone_threshold:    float = 0.85
    segment_threshold: float = 0.93
    max_entries:       int   = 4096
    pq_train_size:     int   = 200     # buffer this many raw segments before training
    pq_n_sub_vectors:  int   = 13      # M — 260 / 13 = 20 sub-dim
    pq_n_codes:        int   = 256     # K — uint8 codes
    pq_train_iter:     int   = 15
    pq_train_seed:     int   = 0
    # After training, the segment-stream threshold is auto-loosened by
    #   k_sigma * std + mean   of holdout encoding error.
    # Reason: asymmetric_cosine of a vector's encoding is < 1.0 by exactly
    # the PQ approximation error. With under-trained codebooks (few train
    # samples vs K=256 centroids per sub-quantizer) the gap is non-negligible.
    # Set k_sigma=1.0 (conservative) — too loose hurts minimal-pair safety.
    pq_threshold_relax_k_sigma: float = 1.0
    pq_threshold_floor:         float = 0.88


class PQTwoStreamCache:
    """Two-stream cache where the segment stream is PQ-compressed.

    Storage per entry after training:
      tone:    6  * 2 = 12 bytes (float16)
      segment: 13 * 1 = 13 bytes (uint8 codes)
      total:           25 bytes  (vs 1064 bytes raw — 42x reduction)
    """

    def __init__(self, cfg: Optional[PQTwoStreamConfig] = None):
        self.cfg = cfg or PQTwoStreamConfig()
        self._store: OrderedDict[int, dict] = OrderedDict()
        self._next_id = 0
        self.stats = TwoStreamStats()
        # PQ machinery
        self._pq: Optional[ProductQuantizer] = None
        self._train_buffer: list[np.ndarray] = []   # raw L2-normed segment vectors
        self._trained = False
        # Effective segment threshold = cfg.segment_threshold pre-training,
        # then loosened post-training by measured PQ self-error.
        self._seg_threshold_effective: float = self.cfg.segment_threshold
        self._pq_self_error: dict = {}

    # -- PQ lifecycle ------------------------------------------------------
    def _maybe_train(self) -> None:
        if self._trained:
            return
        if len(self._train_buffer) < self.cfg.pq_train_size:
            return
        train = np.stack(self._train_buffer, axis=0).astype(np.float32)
        N = train.shape[0]
        # Train on 80% / hold out 20% for honest encoding-error measurement.
        # Self-error on the training set is misleadingly low (trivially 0 when
        # N <= K because each sample becomes its own centroid). Holdout queries
        # land on a non-self centroid, exposing the realistic gap.
        if N >= 10:
            split = int(N * 0.8)
            train_fit = train[:split]
            holdout   = train[split:]
        else:
            train_fit = train
            holdout   = train
        # Cap K so k-means is well-determined: more centroids than samples
        # leaves K-N centroids at random init → catastrophic encoding error.
        effective_K = min(self.cfg.pq_n_codes, max(train_fit.shape[0], 16))
        self._pq = ProductQuantizer(PQConfig(
            n_sub_vectors=self.cfg.pq_n_sub_vectors,
            n_codes=effective_K,
            train_seed=self.cfg.pq_train_seed,
        ))
        self._pq.train(train_fit, n_iter=self.cfg.pq_train_iter)
        # Calibrate PQ encoding error on holdout: each sample is encoded by
        # the trained codebook, then asymmetric_cosine measures the gap.
        errs = []
        for v in holdout:
            codes = self._pq.encode(v)
            score = self._pq.asymmetric_cosine(v, codes)
            errs.append(1.0 - score)
        mu  = float(np.mean(errs))
        sig = float(np.std(errs))
        self._pq_self_error = {
            "mean": mu, "std": sig, "max": float(np.max(errs)), "n": len(errs),
        }
        # Loosen the segment threshold so that a true self-match (with PQ
        # approximation) still passes. The relaxation is bounded by floor.
        delta = mu + self.cfg.pq_threshold_relax_k_sigma * sig
        self._seg_threshold_effective = max(
            self.cfg.pq_threshold_floor,
            self.cfg.segment_threshold - delta,
        )
        # Encode every buffered entry currently in _store that still has raw fp
        for k, e in self._store.items():
            if e.get("seg_codes") is None and e.get("seg_raw") is not None:
                e["seg_codes"] = self._pq.encode(e["seg_raw"])
                e["seg_raw"] = None     # release raw vector
        self._train_buffer.clear()
        self._trained = True

    # -- public API --------------------------------------------------------
    def fingerprint(self, audio: np.ndarray, sr: int) -> TwoStreamFingerprint:
        comps = _components(audio, sr)
        return TwoStreamFingerprint(
            tone=_normalize(comps["tone"]).astype(np.float32),
            segment=_normalize(comps["segment"]).astype(np.float32),
        )

    def put(self, fp: TwoStreamFingerprint, target_lang: str, result: dict) -> None:
        self._next_id += 1
        entry: dict = {
            "tone":        fp.tone.astype(np.float16),
            "seg_raw":     None,
            "seg_codes":   None,
            "target_lang": target_lang,
            "result":      result,
            "inserted":    time.time(),
        }
        if self._trained and self._pq is not None:
            entry["seg_codes"] = self._pq.encode(fp.segment)
        else:
            # store raw until PQ is trained
            entry["seg_raw"] = fp.segment.astype(np.float32)
            self._train_buffer.append(fp.segment.astype(np.float32))
        self._store[self._next_id] = entry
        while len(self._store) > self.cfg.max_entries:
            self._store.popitem(last=False)
        self._maybe_train()

    def lookup(self, fp: TwoStreamFingerprint, target_lang: str) -> Optional[dict]:
        self.stats.lookups += 1
        # Ensure PQ training happens if we have enough buffered, regardless of put pattern.
        self._maybe_train()

        best_tone, best_tone_id = -1.0, None
        best_seg,  best_seg_id  = -1.0, None

        q_tone = fp.tone.astype(np.float32)
        q_seg  = fp.segment.astype(np.float32)

        for k, e in self._store.items():
            if e.get("target_lang") != target_lang:
                continue
            t_score = float(np.dot(q_tone, e["tone"].astype(np.float32)))
            if e["seg_codes"] is not None and self._pq is not None:
                s_score = float(self._pq.asymmetric_cosine(q_seg, e["seg_codes"]))
            else:
                # pre-training fallback: raw vector still in buffer
                s_score = float(np.dot(q_seg, e["seg_raw"]))
            if t_score > best_tone:
                best_tone, best_tone_id = t_score, k
            if s_score > best_seg:
                best_seg,  best_seg_id  = s_score, k

        if best_tone_id is None:
            return None

        tone_passes    = best_tone >= self.cfg.tone_threshold
        segment_passes = best_seg  >= self._seg_threshold_effective
        if tone_passes and not segment_passes:
            self.stats.tone_only_hits += 1
        if segment_passes and not tone_passes:
            self.stats.segment_only_hits += 1

        if tone_passes and segment_passes and best_tone_id == best_seg_id:
            self.stats.both_hits += 1
            self.stats.hits += 1
            entry = self._store.pop(best_tone_id)
            self._store[best_tone_id] = entry
            combined = 2 * best_tone * best_seg / max(best_tone + best_seg, 1e-9)
            return {
                "result":             entry["result"],
                "target_lang":        entry["target_lang"],
                "inserted":           entry["inserted"],
                "cosine":             combined,
                "tone_score":         best_tone,
                "segment_score":      best_seg,
                "both_streams_agree": True,
            }
        return None

    # -- introspection -----------------------------------------------------
    def stored_bytes(self) -> dict:
        """Count bytes actually occupied by tone + segment payloads."""
        tone_b = 0
        seg_b  = 0
        for e in self._store.values():
            tone_b += int(e["tone"].nbytes)
            if e["seg_codes"] is not None:
                seg_b += int(e["seg_codes"].nbytes)
            elif e["seg_raw"] is not None:
                seg_b += int(e["seg_raw"].nbytes)
        return {
            "n_entries":   len(self._store),
            "tone_bytes":  tone_b,
            "segment_bytes": seg_b,
            "total_bytes": tone_b + seg_b,
            "trained":     self._trained,
        }

    def summary(self) -> dict:
        s = self.stored_bytes()
        return {
            **s,
            "max_entries":            self.cfg.max_entries,
            "lookups":                self.stats.lookups,
            "hits":                   self.stats.hits,
            "both_hits":              self.stats.both_hits,
            "tone_only_hits":         self.stats.tone_only_hits,
            "segment_only_hits":      self.stats.segment_only_hits,
            "hit_rate":               round(self.stats.hit_rate, 4),
            "seg_threshold_cfg":      self.cfg.segment_threshold,
            "seg_threshold_effective":round(self._seg_threshold_effective, 4),
            "pq_self_error":          self._pq_self_error,
        }


# Self-test ---------------------------------------------------------------

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


def _round_trip_test(cache: PQTwoStreamCache, audios: list[np.ndarray], sr: int) -> dict:
    """Insert each audio with a unique label, then re-query — expect self-hit."""
    for i, a in enumerate(audios):
        fp = cache.fingerprint(a, sr)
        cache.put(fp, "en", {"id": i, "tgt": f"entry_{i}"})

    hits, cos_sum, n = 0, 0.0, 0
    for i, a in enumerate(audios):
        fp = cache.fingerprint(a, sr)
        h = cache.lookup(fp, "en")
        n += 1
        if h is not None and h["result"]["id"] == i:
            hits += 1
            cos_sum += h["cosine"]
    return {
        "n":            n,
        "hits":         hits,
        "hit_rate":     hits / max(n, 1),
        "mean_cosine":  cos_sum / max(hits, 1),
    }


def _memory_test(n_entries: int = 200) -> dict:
    """Push n_entries synthetic L2-normed fingerprints, report stored bytes."""
    cfg = PQTwoStreamConfig(pq_train_size=min(n_entries, 200))
    cache = PQTwoStreamCache(cfg)
    rng = np.random.default_rng(42)
    for i in range(n_entries):
        tone = rng.normal(0, 1, size=6).astype(np.float32);    tone /= np.linalg.norm(tone) + 1e-9
        seg  = rng.normal(0, 1, size=260).astype(np.float32);  seg  /= np.linalg.norm(seg)  + 1e-9
        fp = TwoStreamFingerprint(tone=tone, segment=seg)
        cache.put(fp, "en", {"id": i})
    # ensure PQ trained
    cache._maybe_train()
    sb = cache.stored_bytes()
    raw_per   = 6 * 4 + 260 * 4          # 1064
    pq_per    = 6 * 2 + cfg.pq_n_sub_vectors  # 12 + 13 = 25
    raw_total = raw_per * n_entries
    return {
        "n_entries":           sb["n_entries"],
        "tone_bytes":          sb["tone_bytes"],
        "segment_bytes":       sb["segment_bytes"],
        "total_bytes":         sb["total_bytes"],
        "raw_per_entry":       raw_per,
        "pq_per_entry":        pq_per,
        "raw_total_estimate":  raw_total,
        "compression_ratio":   raw_total / max(sb["total_bytes"], 1),
        "trained":             sb["trained"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Results/pq_two_stream_cache.json")
    args = ap.parse_args()

    # ---- Test 2: memory test (no audio needed, runs first / cheap) ----
    print("[memory] inserting 200 synthetic entries to measure PQ footprint ...")
    mem = _memory_test(n_entries=200)
    print(f"  n={mem['n_entries']}   raw≈{mem['raw_total_estimate']:,} B   pq={mem['total_bytes']:,} B   "
          f"ratio={mem['compression_ratio']:.1f}x   trained={mem['trained']}")

    # ---- Synthesize Vi minimal-pair clips for tests 1 & 3 ----
    print("\n[load] MMS-TTS-vie ...")
    from transformers import AutoTokenizer, VitsModel
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
    sr = model.config.sampling_rate

    audio_by_key: dict[tuple, np.ndarray] = {}
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants:
            try:
                audio_by_key[(base, v)] = _synth(CARRIER.format(syl=v), tok, model)
            except Exception as e:
                print(f"  [warn] {v}: {e}")
    print(f"[synth] {len(audio_by_key)} clips")

    # ---- Test 1: round-trip with 20 entries ----
    # Synthesize 20 acoustically distinct industrial phrases + a warmup pool.
    # PQ must train on IN-DISTRIBUTION fingerprints (MFCC manifold) — training
    # on random Gaussians produced 0/20 hit rate because the codebook never saw
    # the real fingerprint geometry.
    print("\n[synth] 20 round-trip phrases + warmup pool ...")
    RT_PHRASES = [
        "máy đang chạy", "hãy dừng lại", "kiểm tra động cơ", "mở van số ba",
        "đóng cửa kho", "nhiệt độ cao", "áp suất bình thường", "có người bị thương",
        "gọi bác sĩ", "tắt nguồn ngay", "khẩn cấp di tản", "thay ca tiếp theo",
        "vận chuyển hàng", "đo lưu lượng nước", "ghi nhận sự cố", "bật quạt thông gió",
        "kiểm tra chất lượng", "đóng gói thành phẩm", "lên lịch bảo trì", "báo cáo cuối ngày",
    ]
    WARMUP_PHRASES = [
        "công nhân làm việc", "lò hơi ổn định", "máy in đang sửa", "bình chứa đầy",
        "cảnh báo khói lửa", "thợ điện cần thang", "kho lạnh nhiệt thấp",
        "bộ phận quản lý", "phòng họp tầng hai", "thang máy hỏng",
        "xe nâng cần dầu", "bãi đỗ phía bắc", "cổng kiểm soát mở",
        "camera mất tín hiệu", "kết nối mạng yếu", "máy chủ khởi động",
        "in báo cáo giờ làm", "đồng hồ đo điện", "bồn dầu rò rỉ",
        "lưới điện quá tải", "máy bơm yếu áp", "xe tải đến nơi",
        "băng chuyền dừng", "vận hành khẩn cấp", "an toàn lao động",
        "tổ trưởng kíp ba", "vị trí số bảy", "trạm xăng đóng cửa",
        "nhà máy nghỉ trưa", "đường ống lạnh", "cần thêm vật tư",
        "kéo dài hợp đồng", "khu vực cấm vào", "ghi lại nhật ký",
        "kiểm kê hàng tháng", "đào tạo nhân viên",
        # Pool 2: numbers and codes
        "số một", "số hai", "số ba", "số bốn", "số năm", "số sáu", "số bảy",
        "tầng trệt", "tầng một", "tầng hai", "tầng ba", "tầng bốn", "tầng năm",
        "phòng A một", "phòng A hai", "phòng B một", "phòng B hai", "phòng C ba",
        # Pool 3: actions
        "khởi động", "tắt máy", "tạm dừng", "tiếp tục", "kiểm tra",
        "ghi nhận", "báo cáo", "thông báo", "cảnh báo", "phê duyệt",
        "từ chối", "chấp nhận", "xác nhận", "hủy bỏ", "đăng nhập", "đăng xuất",
        # Pool 4: longer
        "buổi sáng hôm nay trời mát mẻ",
        "công việc đang tiến triển tốt",
        "kết quả kiểm tra đạt yêu cầu",
        "đề nghị xem xét lại quy trình",
        "thiết bị cần được hiệu chỉnh",
        "có nhiều khách đến tham quan",
        "lịch họp thay đổi vào ngày mai",
        "vui lòng chờ trong giây lát",
        "xin liên hệ tổ trưởng",
        "đường truyền dữ liệu ổn định",
        "máy cắt laser đang vận hành",
        "thông tin được ghi lại chính xác",
    ]
    rt_audios = []
    for p in RT_PHRASES:
        try: rt_audios.append(_synth(p, tok, model))
        except Exception as e: print(f"  [warn] {p}: {e}")
    warmup_audios = []
    for p in WARMUP_PHRASES:
        try: warmup_audios.append(_synth(p, tok, model))
        except Exception as e: print(f"  [warn] {p}: {e}")
    print(f"  rt={len(rt_audios)}  warmup={len(warmup_audios)}")
    audios = rt_audios[:20]
    n_train = len(warmup_audios)
    cache_rt = PQTwoStreamCache(PQTwoStreamConfig(pq_train_size=n_train))
    for w in warmup_audios:
        fp_w = cache_rt.fingerprint(w, sr)
        cache_rt.put(fp_w, "zz_warmup", {"warmup": True})
    cache_rt._maybe_train()
    assert cache_rt._trained, f"PQ should be trained after {n_train} warmup entries"
    # Now insert the 20 real audios and query — PQ-encoded path.
    rt = _round_trip_test(cache_rt, audios, sr)
    print(f"\n[round-trip] n={rt['n']}  hits={rt['hits']}  hit_rate={rt['hit_rate']*100:.1f}%  "
          f"mean_cos={rt['mean_cosine']:.4f}  (PQ-encoded path)")

    # ---- Test 3: minimal-pair safety — store only "ngang" tone variant, query others ----
    # Use the same warmup pool so PQ trains on the real MFCC distribution.
    cache_mp = PQTwoStreamCache(PQTwoStreamConfig(pq_train_size=len(warmup_audios)))
    for w in warmup_audios:
        fp_w = cache_mp.fingerprint(w, sr)
        cache_mp.put(fp_w, "zz_warmup", {"warmup": True})
    cache_mp._maybe_train()
    # Store all "ngang" variants (under target_lang="en")
    for base in MINIMAL_PAIR_SYLLABLES:
        ngang_key = (base, MINIMAL_PAIR_SYLLABLES[base][0])
        if ngang_key in audio_by_key:
            fp = cache_mp.fingerprint(audio_by_key[ngang_key], sr)
            cache_mp.put(fp, "en", {"vi": ngang_key[1], "tgt": ngang_key[1]})
        if cache_mp.cfg.max_entries <= len(cache_mp._store):
            break

    correct_misses, false_hits, total = 0, 0, 0
    for base, variants in MINIMAL_PAIR_SYLLABLES.items():
        for v in variants[1:]:
            k = (base, v)
            if k not in audio_by_key:
                continue
            fp = cache_mp.fingerprint(audio_by_key[k], sr)
            h = cache_mp.lookup(fp, "en")
            total += 1
            if h is None:
                correct_misses += 1
            else:
                false_hits += 1
    miss_rate = correct_misses / max(total, 1)
    print(f"\n[minimal-pair] correct_misses={correct_misses}/{total} ({miss_rate*100:.1f}%)   "
          f"false_hits={false_hits}/{total}")

    out = {
        "round_trip": {
            "n_entries":      rt["n"],
            "hits":           rt["hits"],
            "hit_rate":       round(rt["hit_rate"], 4),
            "mean_cosine":    round(rt["mean_cosine"], 4),
        },
        "memory": {
            "n_entries":          mem["n_entries"],
            "raw_per_entry":      mem["raw_per_entry"],
            "pq_per_entry":       mem["pq_per_entry"],
            "raw_total_bytes":    mem["raw_total_estimate"],
            "pq_total_bytes":     mem["total_bytes"],
            "compression_ratio":  round(mem["compression_ratio"], 2),
            "trained":            mem["trained"],
        },
        "minimal_pair_safety": {
            "total":              total,
            "correct_misses":     correct_misses,
            "false_hits":         false_hits,
            "correct_miss_rate":  round(miss_rate, 4),
        },
        "cache_summary": cache_mp.summary(),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {args.out}")
    print("\nSummary:")
    print(f"  round-trip hit_rate          = {out['round_trip']['hit_rate']*100:.1f}%   (target = 100%, cos >= 0.99)")
    print(f"  memory compression ratio     = {out['memory']['compression_ratio']:.1f}x  "
          f"({out['memory']['raw_total_bytes']:,} B raw -> {out['memory']['pq_total_bytes']:,} B PQ)")
    print(f"  minimal-pair correct-miss    = {out['minimal_pair_safety']['correct_miss_rate']*100:.1f}%   "
          "(target >= 90%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
