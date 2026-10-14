# Production-readiness verification — 2026-08-20

This record covers software gates executed on the Windows development host. It is not
Snapdragon, QNN, thermal, power, microphone or acoustic evidence.

## Integrated regression suite

```text
python -m pytest prototype/test_live_translate.py prototype/test_asr_router.py
  prototype/test_pipeline_corrector_integration.py device scripts -q

209 passed, 1 skipped in 23.33s
```

The suite includes pre-ASR source pinning, stale-speculation rejection, ASR routing,
privacy-by-default telemetry, configuration/degradation policy, HTTP surface,
concurrency guards, watchdog logic, acceptance evidence and offline bootstrap tests.

One oversized-body HTTP test transiently failed once with Windows
`ConnectionAbortedError`; the isolated rerun passed, followed by the complete passing
run recorded above.

## Model provenance

```text
python scripts/offline_bootstrap.py verify-models --profile six-direction --json

ok: true
mode: sha256
checked_assets: 46
checked_bytes: 2189076740
findings: []
```

## Current environment blocker

The combined production preflight correctly remains non-ready in the current Python
environment because `sounddevice==0.5.6` is not installed. It also requires the
fail-closed runtime variables to be explicitly set. The policy-only gate passed after
all twelve safe values were supplied. No package was downloaded or installed during
this verification.

## Unexecuted gates

- No real microphone or speaker-loop test.
- No multi-hour acceptance/soak output.
- No Linux/ARM/systemd boot test.
- No Snapdragon/QNN/Hexagon latency, memory, power or thermal result.
