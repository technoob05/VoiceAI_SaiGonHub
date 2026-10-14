# v4 — the Mandarin thread budget, actually applied (`voiceai_bidir-full-v4.json`)

Third attempt at this measurement. v2 ran it with the budget bypassed by a
constructor argument; v3 ran it bypassed again after I overwrote the fixed staged
copy from `exp/`. v4 emits the applied thread counts so the treatment is visible in
the output instead of assumed:

```
cpu_count: 4
tts_threads_per_lang: {"en": 2, "vi": 2, "zh": 4, "ko": 2}
```

Mandarin got its own budget (4, capped by the kernel's 4 vCPU); English, Vietnamese
and Korean stayed at 2 as a control group.

## Result: no detectable gain

| Direction | v3 TTS p50 | v4 TTS p50 | change |
|---|---:|---:|---:|
| **vi→zh** (treated) | 907 ms | 846 ms | **−6.7 %** |
| vi→en (control) | 155 ms | 146 ms | −5.8 % |
| vi→ko (control) | 220 ms | 219 ms | −0.5 % |
| en→vi (control) | 474 ms | 447 ms | −5.7 % |
| ko→vi (control) | 434 ms | 416 ms | −4.1 % |
| zh→vi (control) | 440 ms | 421 ms | −4.3 % |

**The treated direction moved −6.7 % while untreated directions moved −0.5 % to
−5.8 %.** The zh change is inside the band of run-to-run drift on a shared T4, so the
honest conclusion is that doubling Mandarin's threads from 2 to 4 produced **no
effect this measurement can distinguish from noise**.

The control group is what makes that statement possible. Without it, −6.7 % would
have looked like a win.

`vi→zh` under-2 s did read 100 % in v4 against 96.7 % in v3, but that is 2 of 60
utterances crossing a threshold and moves with the same drift.

## Why the dev-box sweep suggested otherwise

The local sweep that motivated this measured 2 → 1211, 4 → 880, 6 → 922, 8 → 786
ms per second of output audio on a **12-core** box. The Kaggle kernel has **4 vCPU**,
so `min(8, cpu_count)` resolves to 4 — the sweep's own 4-thread point, not its
8-thread one. The large gain in that sweep was 2 → 8, which this hardware cannot
reach. On the target device the setting is worth keeping (it costs nothing and the
budget is honest about core count), but no speedup should be claimed from it.

Mandarin TTS remains the system bottleneck: 846 ms p50 against 146–447 ms for the
other three voices, and 6127 ms to load the voice.
