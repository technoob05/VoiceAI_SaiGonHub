# Pipeline cache-miss profile (cache-only mode)

Note: host had <1 GB free RAM during profiling, so PhoWhisper-small INT8 + NLLB-200-distilled-600M-INT8 + MMS-TTS could not be co-resident under cProfile instrumentation. Cold-path latencies below are pulled from the existing reference run `Results/bench_compare_en_n10.json` (fast/INT8 aggregate over 10 VIVOS utterances).

## Reference cascade latencies (bench_compare_en_n10, fast INT8)
- ASR  p50=6861ms  p95=10951ms
- MT   p50=1075ms  p95=3673ms
- TTS  p50=3150ms  p95=7380ms
- Total p50=11940ms  p95=20960ms

## Cache stage micro-benchmarks (this run)
- T1 fingerprint (272-d): p50=28.56ms  p95=66.13ms
- T1 lookup linear scan N=128: p50=0.92ms  p95=1.26ms
- T2 fingerprint + lookup: p50=2.84ms  p95=4.82ms

## Top fingerprint hot spots
- `utterance_cache.py:292:fingerprint` cum=1.518s tot=0.052s calls=20
- `_pocketfft.py:58:_raw_fft` cum=0.874s tot=0.874s calls=100
- `utterance_cache.py:75:_f0_contour` cum=0.855s tot=0.118s calls=20
- `_pocketfft.py:324:rfft` cum=0.613s tot=0.001s calls=60
- `utterance_cache.py:208:cheap_mfcc` cum=0.362s tot=0.126s calls=20

## Bottleneck recommendations (after TTS audio cache)

1. **ASR is the new dominant stage**: p50=6861ms vs MT 1075ms vs TTS 3150ms. Even after TTS caching, ASR alone takes 6.9s for a 3-4s utterance. Options: (a) PhoWhisper-tiny INT8 (4x smaller, expect ~2x faster, evaluate TER cost), (b) raise CT2 intra_threads above auto (`VOICEAI_CT2_INTRA_THREADS=4` or 8 on Snapdragon), (c) drop `no_repeat_ngram_size=3` + `repetition_penalty=1.2` on the streaming path - both add per-beam compute and only help in the loop-failure mode, (d) reduce `compression_ratio_threshold` re-decoding from 1.8 to 2.4 to avoid retries on hard inputs.
2. **MT is well-tuned** (p50=1075ms): beam=1, dynamic max_decoding_length, INT8. Only further win: parallelise NLLB encoder over sentences while decoder runs greedy.
3. **TTS already gated by audio cache** at warm path. First-synth p50=3150ms remains. Pre-tokenize the 42 industrial phrases at startup so tokenizer is hot for first hit (see prototype/industrial_phrases.json).
4. **T1 fingerprint hot spot**: the autocorrelation loop in `_f0_contour` (p50=28.56ms over 4-s audio) is pure Python per-frame. Vectorise with `np.lib.stride_tricks.sliding_window_view` + a single FFT-based autocorrelation across all frames. Target: <10ms.
5. **T1 lookup is O(N) linear scan**. Stack stored fingerprints into a (N, 272) numpy matrix and replace the for-loop in `UtteranceCache.lookup` with one matrix-vector product. Currently p50=0.92ms at N=128; vectorised should be <0.5ms.
6. **CT2 thread underuse**: `ct2_intra_threads=0` (auto) on a 4-core machine often picks 2. Explicitly pin `VOICEAI_CT2_INTRA_THREADS=N_CORES` for the cold path.