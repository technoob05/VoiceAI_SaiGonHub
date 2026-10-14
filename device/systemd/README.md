# systemd units (Linux/ARM target) — UNTESTED

None of these has been run: development happened on Windows 11, which has no
systemd. They are written against the behaviour of `device/supervisor.py`,
`device/watchdog.py` and `device/healthcheck.py`, all of which HAVE been run on
Windows (see `../DEMO_RUNBOOK.md` for the measured numbers).

Install (as root, repo at `/opt/voiceai`):

    cp device/systemd/*.service device/systemd/*.timer /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable --now voiceai-supervisor.service
    systemctl enable --now voiceai-watchdog.timer

Then verify with something that actually probes the model:

    python3 device/healthcheck.py --level full

`systemctl is-active voiceai-supervisor` is NOT a health check. It reports the
process, not the model. That exact confusion is documented in
`../../references/Summer_School_AIEye/aibox_eyes/systemd/geniex-keepalive.conf`.

## Boot persistence

`systemctl enable` gives boot persistence on the Linux target. On the Windows
dev box there is no equivalent in this directory — use Task Scheduler
("At startup", run `device\run_device.ps1`) if you need it; that is a manual
step and is not scripted here.

## Keepalive

There is no keepalive drop-in analogous to `geniex-keepalive.conf`, because the
failure it fought does not exist here: CPython does not unload a loaded
CTranslate2/onnxruntime model after an idle period. The residency risk on this
stack is the OS paging model memory out, which `supervisor.py --keepalive`
(default: one light probe every 120 s) and `vm.swappiness=10` in
`../max_performance.sh` address instead.
