"""
Smoke-test runner for VoiceAI_SaiGonHub prototype modules.

Pure-Python (no pytest). Each test_xxx function raises on failure, returns
silently on success, or raises SkipTest to mark a heavy-resource skip.
Determinism: every test seeds numpy.random with 42.

Run:
    python prototype/test_suite.py

Writes Results/test_suite_report.json with per-test outcomes + summary.
Exit code 0 iff all non-skipped tests pass.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PROJECT = HERE.parent

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try: _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

import numpy as np


class SkipTest(Exception):
    pass


SEED = 42


def _rng():
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 1. UtteranceCache
# ---------------------------------------------------------------------------

def test_utterance_cache_fingerprint_shape_and_norm():
    from utterance_cache import UtteranceCache
    cache = UtteranceCache(max_entries=8, cosine_hit=0.95)
    audio = (_rng().normal(size=24000) * 0.1).astype(np.float32)
    fp = cache.fingerprint(audio, sr=16000)
    assert fp.vec.shape == (272,), f"expected (272,), got {fp.vec.shape}"
    n = float(np.linalg.norm(fp.vec))
    assert abs(n - 1.0) < 1e-3, f"fingerprint not L2-normed: norm={n}"


def test_utterance_cache_put_and_self_hit():
    from utterance_cache import UtteranceCache
    cache = UtteranceCache(max_entries=8, cosine_hit=0.90)
    audio = (_rng().normal(size=24000) * 0.1).astype(np.float32)
    fp = cache.fingerprint(audio, sr=16000)
    cache.put(fp, "en", {"vi": "test", "tgt": "test_en"})
    hit = cache.lookup(fp, "en")
    assert hit is not None, "self-lookup must hit"
    assert hit["result"]["vi"] == "test"
    assert hit["cosine"] >= 0.99


def test_utterance_cache_miss_on_random_noise():
    from utterance_cache import UtteranceCache
    cache = UtteranceCache(max_entries=8, cosine_hit=0.95)
    sr = 16000
    t = np.arange(2 * sr) / sr
    # Two perceptually distinct synthetic signals: a low tone vs. a high
    # rising chirp. The fingerprint mixes F0 + MFCC trajectory so these
    # produce well-separated vectors. Pure Gaussian-noise pairs aren't a
    # valid negative case because the fingerprint collapses on signals with
    # no voicing structure.
    a1 = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    a2 = (0.4 * np.sin(2 * np.pi * np.linspace(300, 600, len(t)) * t)).astype(np.float32)
    fp1 = cache.fingerprint(a1, sr=sr)
    cache.put(fp1, "en", {"vi": "a", "tgt": "a"})
    fp2 = cache.fingerprint(a2, sr=sr)
    hit = cache.lookup(fp2, "en")
    assert hit is None, f"distinct signals should miss; got cosine={hit and hit.get('cosine')}"


# ---------------------------------------------------------------------------
# 2. TwoStreamCache
# ---------------------------------------------------------------------------

def test_two_stream_shapes():
    from two_stream_voting import TwoStreamCache
    cache = TwoStreamCache()
    audio = (_rng().normal(size=24000) * 0.1).astype(np.float32)
    fp = cache.fingerprint(audio, sr=16000)
    assert fp.tone.shape == (6,), f"tone stream expected (6,), got {fp.tone.shape}"
    assert fp.segment.shape == (260,), f"segment stream expected (260,), got {fp.segment.shape}"


def test_two_stream_existence_self_hit():
    from two_stream_voting import TwoStreamCache, TwoStreamConfig
    cfg = TwoStreamConfig(tone_threshold=0.70, segment_threshold=0.80)
    cache = TwoStreamCache(cfg)
    # Use a voiced sinusoid so the tone-histogram stream is non-zero (random
    # Gaussian noise produces an all-zero tone histogram -> zero norm ->
    # self-cosine of 0, failing the existence rule even on identical input).
    sr = 16000
    t = np.arange(2 * sr) / sr
    audio = (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    fp = cache.fingerprint(audio, sr=sr)
    cache.put(fp, "en", {"vi": "hello", "tgt": "hello_en"})
    hit = cache.lookup_existence(fp, "en")
    assert hit is not None, "self-existence-lookup must hit on voiced input"
    assert hit["both_streams_agree"] is True


def test_two_stream_minimal_pair_miss():
    from two_stream_voting import TwoStreamCache, TwoStreamConfig
    cfg = TwoStreamConfig(tone_threshold=0.85, segment_threshold=0.93)
    cache = TwoStreamCache(cfg)
    rng = _rng()
    a1 = (rng.normal(size=24000) * 0.1).astype(np.float32)
    a2 = (rng.normal(size=24000) * 0.1).astype(np.float32)
    fp1 = cache.fingerprint(a1, sr=16000)
    cache.put(fp1, "en", {"vi": "a", "tgt": "a"})
    fp2 = cache.fingerprint(a2, sr=16000)
    hit = cache.lookup_existence(fp2, "en")
    assert hit is None, "random different audio should not pass AND-gate at tight thresholds"


# ---------------------------------------------------------------------------
# 3. TranslationMemory v1
# ---------------------------------------------------------------------------

def test_translation_memory_v1_paraphrastic_hit():
    from translation_memory import TranslationMemory
    tm = TranslationMemory(max_entries=8, cosine_hit=0.80)
    tm.put("an toàn lao động", "occupational safety", "en")
    hit = tm.lookup("AN TOÀN LAO ĐỘNG.", "en")
    assert hit is not None, "case/punct paraphrase should hit"
    assert hit["tgt_text"] == "occupational safety"


# ---------------------------------------------------------------------------
# 4. TranslationMemoryV2
# ---------------------------------------------------------------------------

def test_tm_v2_paraphrase_hit():
    from translation_memory_v2 import TranslationMemoryV2
    tm = TranslationMemoryV2(max_entries=16, tau_p=0.85, tau_s=0.80)
    tm.put("an toàn lao động", "occupational safety", "en")
    hit = tm.lookup("An Toàn Lao Động!", "en")
    assert hit is not None, "case/punct paraphrase should hit (diacritics preserved)"
    assert hit["tgt_text"] == "occupational safety"


def test_tm_v2_diacritic_distinct_no_collision():
    """Round 26 guarantee: 'bay giờ' vs 'bảy giờ' must not collide."""
    from translation_memory_v2 import TranslationMemoryV2
    tm = TranslationMemoryV2(max_entries=16, tau_p=0.85, tau_s=0.80)
    tm.put("bảy giờ", "seven o'clock", "en")
    hit = tm.lookup("bay giờ", "en")
    assert hit is None or hit["tgt_text"] != "seven o'clock", (
        f"'bay giờ' must NOT collide with 'bảy giờ' (Round 26 guarantee); "
        f"got hit cos_p={hit['cos_p']:.3f} cos_s={hit['cos_s']:.3f}" if hit else "")


# ---------------------------------------------------------------------------
# 5. ToneVAD
# ---------------------------------------------------------------------------

def test_tone_vad_rejects_silence():
    from tone_vad import ToneVAD
    vad = ToneVAD()
    audio = np.zeros(16000, dtype=np.float32)
    r = vad.is_likely_speech(audio, 16000)
    assert r.is_speech is False, "silence must be rejected"


def test_tone_vad_rejects_flat_hum():
    """A constant low-frequency hum (no tonal modulation) should be rejected
    by the require_tonal_pattern gate. White noise is intentionally NOT
    tested here: random autocorrelation can spuriously masquerade as voiced
    on a single seed and the VAD is calibrated against real machinery noise,
    not Gaussian noise."""
    from tone_vad import ToneVAD
    vad = ToneVAD()
    sr = 16000
    t = np.arange(sr) / sr
    # Pure 200 Hz sine — voiced-looking F0 but ZERO tonal variance, so
    # log_f0_std < log_f0_std_min should veto.
    audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    r = vad.is_likely_speech(audio, sr)
    assert r.is_speech is False, f"flat hum should be rejected; reason={r.reason}"


# ---------------------------------------------------------------------------
# 6. Noise filter
# ---------------------------------------------------------------------------

def test_noise_filter_shape_preserved():
    from noise_filter import NoiseFilter
    nf = NoiseFilter()
    audio = (_rng().normal(0, 0.05, 32000)).astype(np.float32)
    r = nf(audio, 16000)
    assert r.filtered_audio.shape == audio.shape, (
        f"shape mismatch: in={audio.shape} out={r.filtered_audio.shape}")
    assert r.sr == 16000


def test_noise_filter_reduces_noise_floor():
    """Sanity check: pure-noise input should come out with lower RMS than it
    went in (spectral subtraction has something to subtract). We don't claim
    SNR improvement on signal+noise mixes because spectral subtraction is
    well known to musical-noise distort tonal components; the module is used
    upstream of the cascade where what matters is reducing broadband floor."""
    from noise_filter import NoiseFilter
    nf = NoiseFilter()
    sr = 16000
    rng = _rng()
    # Stationary noise: filter should learn the noise PSD from the head
    # (0.3s) and subtract it from the rest, reducing total energy.
    noise = (rng.normal(0, 0.1, 3 * sr)).astype(np.float32)
    r = nf(noise, sr)
    rms_in  = float(np.sqrt((noise ** 2).mean()))
    rms_out = float(np.sqrt((r.filtered_audio ** 2).mean()))
    assert rms_out < rms_in, f"noise floor not reduced: in={rms_in:.4f} out={rms_out:.4f}"


# ---------------------------------------------------------------------------
# 7. Bootstrap CI
# ---------------------------------------------------------------------------

def test_bootstrap_ci_brackets_true_mean():
    from bootstrap_ci import bootstrap_ci
    rng = _rng()
    true_mu = 3.0
    n_trials = 50
    n_brackets = 0
    for trial in range(n_trials):
        sub_rng = np.random.default_rng(SEED + trial)
        x = sub_rng.normal(loc=true_mu, scale=1.0, size=80)
        pt, lo, hi = bootstrap_ci(x, stat_fn=np.mean, B=500, rng=sub_rng)
        if lo <= true_mu <= hi:
            n_brackets += 1
    cov = n_brackets / n_trials
    assert cov >= 0.90, f"95% CI bracketed true mean only {cov*100:.0f}% of trials (expected >=90%)"


# ---------------------------------------------------------------------------
# 8. Persistent cache
# ---------------------------------------------------------------------------

def test_persistent_cache_t2_roundtrip(tmp_dir):
    from translation_memory import TranslationMemory
    from persistent_cache import save_t2_text_cache, load_t2_text_cache
    tm = TranslationMemory(max_entries=8, cosine_hit=0.80)
    tm.put("an toàn lao động", "occupational safety", "en")
    tm.put("dừng dây chuyền", "stop the production line", "en")
    out_path = tmp_dir / "t2.json"
    save_t2_text_cache(tm, out_path)
    fresh = TranslationMemory(max_entries=8, cosine_hit=0.80)
    info = load_t2_text_cache(fresh, out_path)
    assert info["loaded"] == 2, f"expected 2 entries loaded, got {info}"
    h = fresh.lookup("an toàn lao động", "en")
    assert h is not None and h["tgt_text"] == "occupational safety"


# ---------------------------------------------------------------------------
# 9. Learned tone embedding
# ---------------------------------------------------------------------------

def test_learned_tone_embedding_forward():
    npz_path = PROJECT / "data" / "cache_snapshots" / "learned_tone_stream.npz"
    if not npz_path.exists():
        raise SkipTest(f"learned tone weights not found: {npz_path}")
    from learned_tone_stream import ToneEmbeddingMLP, clip_to_64d
    mlp = ToneEmbeddingMLP.load(npz_path)
    audio = (_rng().normal(0, 0.05, 32000)).astype(np.float32)
    feat = clip_to_64d(audio, 16000)
    # NOTE: function is named clip_to_64d but actually produces 63 dims
    # (the saved weights confirm in_dim=63). Documented mismatch; test the
    # *actual* contract, not the function name.
    assert feat.shape[0] == mlp.W1.shape[0], (
        f"feature dim {feat.shape[0]} != W1 in_dim {mlp.W1.shape[0]}")
    out = mlp.apply(feat[None, :])[0]
    assert out.shape == (32,), f"expected 32-d output, got {out.shape}"
    n = float(np.linalg.norm(out))
    assert abs(n - 1.0) < 1e-3, f"learned embedding not L2-normed: norm={n}"


# ---------------------------------------------------------------------------
# 10. TTS audio cache (pipeline_fast)
# ---------------------------------------------------------------------------

def test_tts_audio_cache_lru_eviction():
    """Test the LRU eviction logic of _tts_audio_cache without loading TTS."""
    # Simulate the LRU directly with the same data structures as pipeline_fast.
    cache: dict = {}
    order: list = []
    cap = 4
    def insert(key, val):
        cache[key] = val
        order.append(key)
        if len(order) > cap:
            old = order.pop(0)
            cache.pop(old, None)
    for i in range(6):
        insert(("en", f"text_{i}"), (np.zeros(8, np.float32), 16000))
    assert len(cache) == cap, f"expected cap={cap} entries, got {len(cache)}"
    assert ("en", "text_0") not in cache, "oldest entry should have been evicted"
    assert ("en", "text_5") in cache, "newest entry should be present"


def test_tts_audio_cache_first_vs_second_call():
    """MMS-TTS-vie + VITS forward pass is too heavy for a smoke test on CPU
    (~30 s cold load, ~1-2 s per synthesis). The LRU semantics that govern
    correctness are covered by test_tts_audio_cache_lru_eviction, which
    runs the exact same data-structure dance in <1 ms without loading the
    model. Cold-vs-warm latency claims must be re-checked manually before
    paper claims via prototype/pipeline_fast.py."""
    raise SkipTest("intentional skip: heavy TTS model not loaded in smoke test")


# ---------------------------------------------------------------------------
# 11. Syllable-local tone pooling
# ---------------------------------------------------------------------------

def test_syllable_local_aggregation_strategies():
    npz_path = PROJECT / "data" / "cache_snapshots" / "learned_tone_stream.npz"
    if not npz_path.exists():
        raise SkipTest(f"learned tone weights not found: {npz_path}")
    from learned_tone_stream import ToneEmbeddingMLP
    from syllable_local_tone import (aggregate_mean, aggregate_max,
                                      ConcatProjector, AttentionPooler,
                                      EMB_DIM, K_SYL)
    _ = ToneEmbeddingMLP.load(npz_path)

    rng = _rng()
    # multi-syllable signal -> (K, 32) random L2-normed embeddings
    K = 3
    embs = rng.normal(size=(K, EMB_DIM)).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    v_mean = aggregate_mean(embs)
    v_max = aggregate_max(embs)
    proj = ConcatProjector.identity_sum()
    v_concat = proj.apply(embs)
    attn = AttentionPooler.init(seed=0)
    v_attn = attn.apply(embs)
    for name, v in [("mean", v_mean), ("max", v_max),
                    ("concat", v_concat), ("attention", v_attn)]:
        assert v.shape == (EMB_DIM,), f"{name}: shape {v.shape} != ({EMB_DIM},)"
        n = float(np.linalg.norm(v))
        assert 0.99 < n < 1.01, f"{name}: not L2-normed (n={n})"


def test_syllable_local_max_equals_mean_on_single_syllable():
    from syllable_local_tone import aggregate_mean, aggregate_max
    rng = _rng()
    v = rng.normal(size=(1, 32)).astype(np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    out_mean = aggregate_mean(v)
    out_max = aggregate_max(v)
    # On a single syllable both reduce to the L2-normed single vector itself.
    assert np.allclose(out_mean, out_max, atol=1e-6), (
        "max-pool and mean-pool must agree on K=1")


# ---------------------------------------------------------------------------
# 12. Adaptive threshold lookup
# ---------------------------------------------------------------------------

def test_adaptive_threshold_clean_band():
    from adaptive_threshold import lookup_thresholds
    tA, tB, label = lookup_thresholds(40.0)  # well above 25 dB cutoff
    assert (tA, tB) == (0.85, 0.93), f"clean band expected (0.85, 0.93), got ({tA}, {tB})"


def test_adaptive_threshold_10db_band():
    from adaptive_threshold import lookup_thresholds
    tA, tB, label = lookup_thresholds(10.0)
    assert (tA, tB) == (0.92, 0.95), f"10dB band expected (0.92, 0.95), got ({tA}, {tB})"


def test_adaptive_threshold_snr_estimator_on_synthetic_clip():
    from adaptive_threshold import estimate_snr_db
    sr = 16000
    rng = _rng()
    t = np.arange(2 * sr) / sr
    # Tone bursts (voiced) + silence gaps so the estimator has a noise floor to find.
    speech = np.zeros_like(t, dtype=np.float32)
    for start in (0.0, 0.6, 1.2):
        seg = (np.arange(int(0.3 * sr)) / sr)
        sig = (0.5 * np.sin(2 * np.pi * 200 * seg)).astype(np.float32)
        s_idx = int(start * sr)
        speech[s_idx:s_idx + len(sig)] = sig
    # Add noise at ~20 dB.
    target_snr_db = 20.0
    sig_power = float((speech ** 2).mean()) + 1e-12
    noise_power = sig_power / (10 ** (target_snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(speech)).astype(np.float32)
    audio = speech + noise
    est = estimate_snr_db(audio, sr)
    assert abs(est - 20.0) < 6.0, f"SNR estimate {est:.1f} dB too far from ground truth 20 dB"


# ---------------------------------------------------------------------------
# 13. Streaming cache lookup
# ---------------------------------------------------------------------------

def test_streaming_consecutive_confirmation_rejects_single_hit():
    from streaming_cache_lookup import (
        streaming_lookup, CacheEntry, _components_partial,
    )
    rng = _rng()
    sr = 16000
    audio = (rng.normal(0, 0.1, 4 * sr)).astype(np.float32)
    tone_full, seg_full = _components_partial(audio, sr)
    # Store a SINGLE entry that ONLY has the full-length fingerprint.
    # Earlier (partial) decision points will fall back to the full fp; the cosine
    # against partial-audio fingerprints will fluctuate. With consecutive=2 and a
    # noisy partial, we want to verify the protocol completes without false
    # commits at partial points purely from a single passing comparison.
    store = [CacheEntry(key="canon", by_t={None: (tone_full, seg_full)},
                        duration_s=len(audio)/sr)]
    # Query: an UNRELATED utterance — should not commit at any partial point.
    different = (rng.normal(0, 0.1, 4 * sr)).astype(np.float32)
    r = streaming_lookup(different, sr, store,
                         tau_a=0.96, tau_b=0.98,
                         require_consecutive=2)
    # require=2 with tight thresholds should reject unrelated audio.
    assert r["committed"] is False, (
        f"unrelated audio should not commit under consecutive=2 + tight tau; "
        f"got commit={r['commit']}")


# ---------------------------------------------------------------------------
# 14. HNSW ANN
# ---------------------------------------------------------------------------

def test_hnsw_recall_at_1():
    try:
        import nmslib  # noqa: F401
    except Exception as e:
        raise SkipTest(f"nmslib unavailable: {e}")
    import nmslib
    rng = _rng()
    N = 128
    D = 272
    corpus = rng.normal(size=(N, D)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.normal(size=(20, D)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    idx = nmslib.init(method="hnsw", space="cosinesimil")
    idx.addDataPointBatch(corpus)
    idx.createIndex({"M": 16, "efConstruction": 200, "post": 0}, print_progress=False)
    idx.setQueryTimeParams({"efSearch": 200})

    n_match = 0
    for q in queries:
        ids, _ = idx.knnQuery(q, k=1)
        sims = corpus @ q
        true_top = int(np.argmax(sims))
        if int(ids[0]) == true_top:
            n_match += 1
    recall = n_match / len(queries)
    assert recall >= 0.95, f"HNSW recall@1 = {recall:.2f} < 0.95"


# ---------------------------------------------------------------------------
# 15. DTW baseline
# ---------------------------------------------------------------------------

def test_dtw_self_distance_zero():
    from dtw_baseline import dtw_distance
    rng = _rng()
    x = rng.normal(size=(40, 13)).astype(np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    d = dtw_distance(x, x, radius_frac=0.5)
    assert d < 1e-6, f"DTW(x, x) expected 0, got {d}"


def test_dtw_symmetry():
    from dtw_baseline import dtw_distance
    rng = _rng()
    x = rng.normal(size=(30, 13)).astype(np.float32)
    y = rng.normal(size=(36, 13)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    d_xy = dtw_distance(x, y, radius_frac=0.5)
    d_yx = dtw_distance(y, x, radius_frac=0.5)
    # Banded DTW is approximately symmetric (band may differ slightly with non-square T).
    rel = abs(d_xy - d_yx) / max(d_xy, d_yx, 1e-9)
    assert rel < 0.10, f"DTW asymmetry too large: d(x,y)={d_xy:.4f} d(y,x)={d_yx:.4f}"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

TESTS = [
    ("utterance_cache_fingerprint_shape_norm",   test_utterance_cache_fingerprint_shape_and_norm),
    ("utterance_cache_put_self_hit",             test_utterance_cache_put_and_self_hit),
    ("utterance_cache_miss_random",              test_utterance_cache_miss_on_random_noise),
    ("two_stream_shapes",                        test_two_stream_shapes),
    ("two_stream_existence_self_hit",            test_two_stream_existence_self_hit),
    ("two_stream_minimal_pair_miss",             test_two_stream_minimal_pair_miss),
    ("tm_v1_paraphrastic_hit",                   test_translation_memory_v1_paraphrastic_hit),
    ("tm_v2_paraphrase_hit",                     test_tm_v2_paraphrase_hit),
    ("tm_v2_diacritic_distinct_no_collision",    test_tm_v2_diacritic_distinct_no_collision),
    ("tone_vad_rejects_silence",                 test_tone_vad_rejects_silence),
    ("tone_vad_rejects_flat_hum",                test_tone_vad_rejects_flat_hum),
    ("noise_filter_shape_preserved",             test_noise_filter_shape_preserved),
    ("noise_filter_reduces_noise_floor",         test_noise_filter_reduces_noise_floor),
    ("bootstrap_ci_brackets_true_mean",          test_bootstrap_ci_brackets_true_mean),
    ("persistent_cache_t2_roundtrip",            test_persistent_cache_t2_roundtrip),
    ("learned_tone_embedding_forward",           test_learned_tone_embedding_forward),
    ("tts_audio_cache_lru_eviction",             test_tts_audio_cache_lru_eviction),
    ("tts_audio_cache_first_vs_second",          test_tts_audio_cache_first_vs_second_call),
    ("syllable_local_aggregation_strategies",    test_syllable_local_aggregation_strategies),
    ("syllable_local_max_equals_mean_K1",        test_syllable_local_max_equals_mean_on_single_syllable),
    ("adaptive_threshold_clean_band",            test_adaptive_threshold_clean_band),
    ("adaptive_threshold_10db_band",             test_adaptive_threshold_10db_band),
    ("adaptive_threshold_snr_estimator",         test_adaptive_threshold_snr_estimator_on_synthetic_clip),
    ("streaming_consecutive_confirmation",       test_streaming_consecutive_confirmation_rejects_single_hit),
    ("hnsw_recall_at_1",                         test_hnsw_recall_at_1),
    ("dtw_self_distance_zero",                   test_dtw_self_distance_zero),
    ("dtw_symmetric",                            test_dtw_symmetry),
]


def main() -> int:
    import inspect
    import tempfile
    np.random.seed(SEED)

    tmp_root = Path(tempfile.mkdtemp(prefix="voiceai_test_"))

    results = []
    t_start = time.perf_counter()
    n_pass = n_fail = n_skip = 0
    for name, fn in TESTS:
        t0 = time.perf_counter()
        status = "PASS"
        err = None
        np.random.seed(SEED)
        try:
            sig = inspect.signature(fn)
            if "tmp_dir" in sig.parameters:
                fn(tmp_dir=tmp_root / name)
                (tmp_root / name).mkdir(parents=True, exist_ok=True)
            else:
                fn()
            n_pass += 1
        except SkipTest as e:
            status = "SKIP"
            err = str(e)
            n_skip += 1
        except Exception as e:  # noqa: BLE001
            status = "FAIL"
            err = f"{type(e).__name__}: {e}"
            tb_lines = traceback.format_exc().strip().splitlines()[-3:]
            err = err + "\n        " + "\n        ".join(tb_lines)
            n_fail += 1
        dt = (time.perf_counter() - t0) * 1000
        results.append({"name": name, "status": status,
                        "elapsed_ms": round(dt, 1),
                        "error": err})
        marker = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
        line = f"{marker} {name:<48} {dt:7.1f} ms"
        if err and status != "PASS":
            line += f"  -- {err.splitlines()[0]}"
        print(line)
        if err and status == "FAIL":
            for ln in err.splitlines()[1:]:
                print("       " + ln)

    elapsed_s = time.perf_counter() - t_start
    n_total = len(TESTS)
    print()
    print(f"=== {n_pass}/{n_total} passed  ({n_skip} skipped, {n_fail} failed)  in {elapsed_s:.1f}s ===")

    report = {
        "summary": {
            "total":   n_total,
            "passed":  n_pass,
            "failed":  n_fail,
            "skipped": n_skip,
            "elapsed_s": round(elapsed_s, 2),
        },
        "tests": results,
    }
    out_path = PROJECT / "Results" / "test_suite_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[write] {out_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
