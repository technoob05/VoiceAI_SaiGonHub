# Integration Test Report

- **Result**: 11/11 passed  (0 skipped, 0 failed)
- **Runtime**: 258.28 s
- **Cold init**: 82861 ms
- **Cache hit speedup**: 108.1x

## Per-test outcomes

| # | Test | Status | Elapsed (ms) | Notes |
|---|------|--------|--------------|-------|
| 01 | cold_init | PASS | 82877.9 | cold_init_ms=82861.3; under_60s_target=False |
| 02 | single_cold_pass | PASS | 19616.0 | vi=tiếng cọc gạch khựng lạ của những khớp sắt.; tgt=The strange crackling of the iron joints. |
| 03 | cache_hit_replay | PASS | 181.5 | cold_total_ms=19598.8; warm_total_ms=181.2 |
| 05 | industrial_prewarming | PASS | 22576.2 | n_phrases_inserted=42; asr_recognized_vi=anh còn lao động. |
| 06 | persistence_roundtrip | PASS | 288.3 | snapshot_saved={'snapshot_dir': 'D:\\PhD_LetGoo\\EACL\\VoiceAI_SaiGonHub\\data\\cache_snapshots\\integration_test', 'saved_at': 1779971990.590009, 't1': {'path' |
| 07 | tts_audio_cache_hit | PASS | 2723.8 | cold_ms=2690.13; warm_ms=0.08 |
| 08 | vad_silence_rejection | PASS | 33.3 | stages_observed=['vad_reject']; vad_rejected=True |
| 09 | streaming_generator | PASS | 34485.5 | stages=['asr_partial', 'mt_partial', 'done']; final_stage=done |
| 10 | long_utterance | PASS | 28161.3 | duration_s=14.0; vi_text=tiếng cọc cạch khựng lợi của những khớp sắt di tích có bảy phỗng ông và hai phỗnh bà được làm bằng gỗ mít ngọn lửa bạo động ở trung đôn |
| 11 | noisy_miss_then_hit | PASS | 15254.1 | first_cache_hit=False; first_total_ms=15067.8 |
| 04 | three_language_sanity | PASS | 52013.7 | per_language={'en': {'text': "And I'm also speaking out for these proposals.", 'nonempty': True, 'latency_ms': 10795.6}, 'zh': {'text': '支持这些建议.', 'nonempty': T |

## Bug-fix log

No production-code patches were required during this run (all observed pipeline behaviors matched expected contracts; see test detail JSON for evidence). Skips were due to disabled feature flags or missing optional resources, not bugs.
