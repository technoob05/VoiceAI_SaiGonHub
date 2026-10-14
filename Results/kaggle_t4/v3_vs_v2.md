# Run v3 vs run v2 — status: **the T4 run did not happen**

Attempted 2026-08-20. Everything needed for `bidir-full-v3` is staged and live on
Kaggle, but the kernel could not be started: the account is out of weekly GPU
quota. What follows separates what is *measured*, what is *proven without the
GPU*, and what is still *unmeasured*.

## What was staged (all under `kaggle_uploads/`)

| File | Change |
|---|---|
| `voiceai-t4-srclib/industrial_terms.json` | refreshed from `prototype/`, 61 → 93 terms |
| `voiceai-t4-bidir-config/benchmark_config.json` | `condition` → `bidir-full-v3`, `dataset_version` → 4 |
| `voiceai-t4-kernel-bidir/exp_t4_bidirectional.py` | TTS thread-override fix (below) |

Both dataset versions pushed and were **verified live by re-download** before the
kernel push: config reads `bidir-full-v3`, srclib reads 93 terms. The rest of
srclib already matched `prototype/`, including the `_threads_for` version of
`local_tts_piper.py`. `cache_cap=4` and the four-voice pre-warm are untouched —
a cold voice load is still not charged to `tts_ms`.

## Blocker

```
kaggle kernels push -p kaggle_uploads/voiceai-t4-kernel-bidir
Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.
```

This is **not** the 2-concurrent-session limit. No kernels were running — all of
`voiceai-t4-noise-robustness`, `voiceai-t4-zh-script-ab`, `voiceai-t4-bidirectional`
and `voiceai-t4-inference-full` reported `COMPLETE`. The four T4 runs on 19–20 Aug
consumed the week's 30 hours. Retried on a loop for ~80 min with no change; Kaggle's
weekly quota resets Saturday 00:00 UTC, so the run has to be re-pushed then. The
five alternate accounts in `kaggle_for_research/` all return 403 on these private
datasets, and re-uploading the inputs is ~20 GB of models and speech, so
cross-account was not a viable route.

**To finish the run, nothing needs editing — just re-push the kernel.**

---

## Finding 1 (confirmed, no GPU needed): the benchmark was overriding the Mandarin thread budget

`prototype/local_tts_piper.py`:

```python
self.num_threads = num_threads or int(os.environ.get("VOICEAI_TTS_THREADS", "4"))
self._thread_override = num_threads is not None
```

and `_threads_for` opens with `if self._thread_override: return self.num_threads`.

The benchmark constructed `PiperTTS(num_threads=2, ...)`, which sets that flag, so
`_threads_for` returned 2 for **every** voice and the per-language Mandarin budget
added after v2 was dead code inside this benchmark. Verified locally:

| Constructor | en | vi | zh | ko | `_thread_override` |
|---|---:|---:|---:|---:|---|
| `PiperTTS(num_threads=2, ...)` (v2) | 2 | 2 | 2 | 2 | `True` |
| `PiperTTS(...)` with `VOICEAI_TTS_THREADS=2` (v3) | 2 | 2 | **8** | 2 | `False` |

**This is a reportable finding on its own**: v2's 824 ms Mandarin p50 was measured
with the fix disabled, and *any* re-run that kept the explicit `num_threads=2`
would have reproduced 824 ms and been misread as "the thread change did nothing".

The staged fix routes the 2 through the environment default instead of the
constructor, so `en`/`vi`/`ko` stay at the 2 threads v2 used — those rows remain
comparable — and only `zh` takes its own `min(8, cpu_count)`. A Kaggle T4 kernel
has 4 vCPU, so **zh would run at 4 threads, not 8**. The script now records
`tts_threads_per_lang` and `cpu_count` in the report so this can never again be
assumed rather than checked.

Expected effect, from the dev-box sweep in the v2 write-up (2 → 1211, 4 → 880 ms
per second of output audio): roughly a 25–30 % cut in `tts_ms` for `vi→zh`. That
sweep ran on a contended 12-core box where a repeat measurement moved 698 → 1412,
so the *ordering* is the trustworthy part and the magnitude is not. **Unmeasured
on T4.**

## Finding 2 (confirmed, no GPU needed): the 32 new glossary terms cannot move sentence recovery

The probe set is built from the glossary itself, so before spending a run I checked
whether the new terms can even reach the sentence probes. `Glossary.find_terms` over
the 12 fixed industrial sentences returns the **same 34 terms** under both tables —
34 × 3 targets = 102, exactly the denominator v2 reported. Not one of the 32 new
terms intersects a probe sentence.

Confirmed empirically with a controlled A/B on the same NLLB-200 CT2 int8 model
(CPU, `glossary_61_vs_93_cpu_ab.py`, both arms same machine and session):

| Glossary | probes | terms | recovery OFF | recovery ON |
|---|---:|---:|---:|---:|
| 61 terms (what v2 ran) | 36 | 102 | 0.1569 | 0.4510 |
| 93 terms (current) | 36 | 102 | 0.1569 | **0.4510** |

All **36/36 sentence rows are byte-identical** between the two arms — same `off`
string, same `on` string, same expected-term list. The delta is not "small", it is
exactly zero, and it is zero by construction rather than by luck.

(These CPU figures sit near but not on v2's GPU figures — 0.167/0.441 there,
0.157/0.451 here — because greedy decode under `int8` on CPU differs slightly from
`int8_float16` on CUDA. That offset is irrelevant to the comparison above, which is
internally controlled: both arms ran on the same box in the same session.)

### Is the v2-vs-v3 comparison even apples-to-apples? No.

Two reasons, and both matter:

1. **The sentence probes are fixed text.** Growing the glossary changes the
   denominator only if a new term appears in one of those 12 sentences. None does.
   The measurement is structurally blind to the expansion.
2. **The bare-term probes are `terms[:24]` — a positional slice.** Inserting 32
   terms shifted that window, so v3 would probe a *different* 24 bare terms than v2
   did. Seven safety terms (`đồ bảo hộ`, `kính bảo hộ`, `mũ bảo hộ`, `găng tay`,
   `giày bảo hộ`, `dây đai an toàn`, `nút dừng khẩn cấp`) drop out and seven
   machinery terms (`máy ép`, `máy cắt`, `máy khoan`, `băng tải`, `vòng bi`,
   `dầu bôi trơn`, `khí nén`) enter. Those are exact whole-utterance hits that never
   reach the decoder, so they score 1.000 either way — but the **all-probes OFF**
   figure does move on this alone (0.1897 → 0.1839 locally), which is an artifact of
   the slice, not a change in behaviour.

So the honest reading: **the README's claim that "coverage, not the mechanism, is
the limit" was not tested by adding 32 terms, and cannot be tested by this probe
set.** A glossary built from its own probe set measures the mechanism against
itself. To actually price coverage, the probes have to come from somewhere else —
held-out industrial sentences written without reference to the table, ideally with
the terms chosen before the glossary was extended.

## What is still unmeasured

Everything that needs the GPU. `vi→zh` TTS latency under the corrected thread
budget is the one number this run existed to get, and it is the one number not in
hand. Also unmeasured, and worth watching when v3 does run, since a shared T4
drifts run-to-run:

| Metric | v2 | v3 |
|---|---:|---|
| `vi→zh` TTS p50 | 824.3 ms | pending |
| `vi→zh` total p50 | 1108.2 ms | pending |
| `vi→en` / `vi→ko` / `en→vi` / `ko→vi` / `zh→vi` TTS p50 | 141.6 / 200.6 / 430.9 / 396.1 / 397.6 ms | pending — should be flat, these keep 2 threads |
| PhoWhisper WER / CER | 0.1259 / 0.0718 | pending — no input changed, should be flat |
| FLEURS CER en / ko / zh | 0.0320 / 0.0732 / 0.2216 | pending — should be flat |
| Streaming first audio (fast / whole) | 207.1 / 465.3 ms | pending |
| Peak GPU | 2865 MiB | pending |
| Term recovery, translated sentences | 0.441 | **provably unchanged** (see above) |

When v3 lands, treat a small `vi→zh` improvement with suspicion: the other five
directions are the control. If they drift by a similar proportion, the zh delta is
run-to-run noise on a shared T4 and not the thread change. Only a zh improvement
that is large relative to the drift in the untouched directions is evidence.

## Files

| File | Contents |
|---|---|
| `glossary_61_vs_93_cpu_ab.json` | Full 61-vs-93 A/B: every probe, both arms, both glossaries |
| `glossary_61_vs_93_cpu_ab.py` | The script that produced it |
| `_terms_old61.json` | The exact 61-term table v2 ran (recovered from srclib dataset v3) |
