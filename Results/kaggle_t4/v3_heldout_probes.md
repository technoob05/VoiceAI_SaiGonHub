# v3 — the held-out probe set (`voiceai_bidir-full-v3.json`)

Run on `kit567` after the `daosyduyminh` weekly GPU quota ran out. Same script,
same models; the change is that experiment D now draws its probes from
`prototype/glossary_probe_sentences.json` — 32 industrial sentences authored
independently of the glossary — instead of from the glossary itself.

## Why the probe set had to change

v2 reported term recovery 0.167 → 0.441 and the write-up concluded "coverage, not
the mechanism, is the limit". That conclusion was untestable as stated: the probes
came from the table being measured, so `find_terms` returned the identical 34 terms
whether the table held 61 entries or 93. Growing it changed the number by exactly
zero — 36/36 rows byte-identical in a controlled CPU A/B.

## Result

| Probe class | probes | terms | recovery OFF | recovery ON |
|---|---:|---:|---:|---:|
| **Held-out sentences** | 96 | 279 | 0.194 | **0.462** |
| Exact whole-utterance hits | 18 | 18 | 0.111 | 1.000 |
| All probes | 114 | 297 | 0.189 | 0.495 |

96 = 32 sentences × 3 targets. Glossary at 93 terms.

**The headline number survives the fix.** 0.194 → 0.462 on independent sentences is
within noise of v2's 0.167 → 0.441 on glossary-derived ones. So the circular probe
set was not inflating the effect — it was only blind to *changes* in coverage. The
measurement is now capable of detecting a coverage improvement; nothing has yet
demonstrated one.

Recovery still counts the glossary's *preferred* surface form, so a correct synonym
scores as a miss. It measures house-term consistency, not adequacy.

## Six directions, unchanged within noise

| Direction | total p50 | RTF p50 | TTS p50 | under 2 s |
|---|---:|---:|---:|---:|
| vi→en | 459 ms | 0.137 | 155 ms | 100 % |
| vi→zh | 1175 ms | 0.366 | **907 ms** | 96.7 % |
| vi→ko | 507 ms | 0.159 | 220 ms | 100 % |
| en→vi | 874 ms | 0.093 | 474 ms | 100 % |
| ko→vi | 888 ms | 0.073 | 434 ms | 100 % |
| zh→vi | 847 ms | 0.092 | 440 ms | 100 % |

## What this run did NOT measure, and why

**The Mandarin TTS thread budget was still bypassed.** The fix had been applied to
the kernel's staged copy of the script; I then overwrote that copy from `exp/` while
wiring the probe set, silently reverting it. The report fields that would have
exposed it (`tts_threads_per_lang`, `cpu_count`) came back `None`, and vi→zh TTS
read 907 ms — v2 territory, i.e. still `num_threads=2`.

The fix now lives in `exp/exp_t4_bidirectional.py` itself, where a copy-from-source
cannot revert it, and the run emits `tts_threads_per_lang` so the treatment is
visible in the output rather than assumed. Re-measured as `bidir-full-v4`.

This is the second time in two days that a treatment silently failed to apply — the
first was the constructor argument itself. The lesson holds: **assert in the output
that the treatment was applied, or the A/B measures nothing.**
