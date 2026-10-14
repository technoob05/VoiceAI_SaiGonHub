# Privacy and Data Handling

This document describes the current submission runtime, not a future cloud service.
The default deployment is local-only and does not require an API or network connection
after model preparation.

## Default data flow

- Microphone frames are consumed in memory by VAD and ASR. The application does not
  intentionally write captured audio to disk.
- Transcripts and translations are emitted to the local UI and speaker because they are
  the product output.
- Persistent structured logs are metadata-only by default. They retain direction,
  character counts, stage state and latency, but not transcript or translation text.
- T1/T2 text caches are disabled in the submission configuration. No conversation
  history is required for translation.
- Runtime model download is disabled, and Hugging Face/Transformers are placed in
  offline mode after assets have been staged and verified.

## Explicit diagnostic opt-in

`VOICEAI_LOG_CONTENT=1` adds bounded 120-character source and target previews to mic
turn records. This setting is for an explicitly consented diagnostic session only. It
must remain `0` for judging, field trials and unattended operation.

The active posture is exposed by `/healthz.privacy`:

```json
{
  "content_logging": false,
  "structured_log_policy": "metadata-only"
}
```

## Retention and deletion

`device/devlog.py` rotates each JSONL stream at 8 MiB and keeps three backups. This is
a storage bound, not a time-based retention guarantee. Before a field release, the
operator must define a time limit and a deletion procedure appropriate to the site.
Diagnostic logs can be removed after stopping the supervisor; no model artifact depends
on them.

## Security boundary and known limitations

- The HTTP service binds to `127.0.0.1` by default and has no remote authentication.
  Do not expose it on a LAN/WAN without an authenticated local gateway.
- Application-level log encryption is not implemented. Use OS full-disk encryption and
  restrict filesystem permissions on the device account.
- The no-audio-file statement covers application code, not operating-system swap,
  crash dumps, sound-driver diagnostics or third-party debugging tools.
- A formal data-protection impact assessment and the organizer's retention requirements
  remain open deployment gates.
