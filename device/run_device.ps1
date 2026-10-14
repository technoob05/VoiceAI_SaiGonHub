<#
run_device.ps1 — the Windows dev-box launcher. Same mandatory order and the same
pass/fail readiness line as device/run_device.sh, so what you rehearse on the
laptop is what runs on the box.

  0. refuse if something already listens on the port
  1. verify the model manifest (SHA-256)     -> device/provenance.py
  2. start the supervisor (loads every model once, never unloads, prewarms
     ASR -> corrector -> MT -> TTS -> frontend with real inferences)
  3. block until a REAL end-to-end probe passes, print PASS/FAIL
  4. optionally start the watchdog

What is deliberately NOT here: the CPU-governor / C-state pinning from
device/max_performance.sh. Windows has no cpufreq sysfs; the closest equivalent
is a power plan, and this script does not silently change the machine's power
settings. If you want it: powercfg /setactive SCHEME_MIN (High performance).

Usage:
  powershell -ExecutionPolicy Bypass -File device\run_device.ps1
  powershell -ExecutionPolicy Bypass -File device\run_device.ps1 -Target zh -QuickVerify -Watchdog
#>
param(
  [int]$Port = 8731,
  [ValidateSet("en","zh","ko","vi")][string]$Target = "en",
  [switch]$QuickVerify,
  [switch]$SkipVerify,
  [switch]$Watchdog,
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$Project = Split-Path -Parent $Here
Set-Location $Project
New-Item -ItemType Directory -Force -Path "$Here\logs","$Here\evidence" | Out-Null

Write-Host "== 0. resource conflict check (port $Port) =="
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  $owner = (Get-Process -Id $busy[0].OwningProcess -ErrorAction SilentlyContinue)
  Write-Host "   REFUSING TO START: port $Port is already owned by pid $($busy[0].OwningProcess) $($owner.ProcessName)"
  Write-Host "   Check it first:  $Python device\healthcheck.py --port $Port"
  exit 3
}
Write-Host "   OK"

$verifyArgs = @()
if ($QuickVerify) { $verifyArgs += "--quick-verify" }
if ($SkipVerify)  { $verifyArgs += "--skip-verify" }

Write-Host "== 1-6. supervisor: verify manifest, load models once, prewarm every stage =="
Write-Host "   Cold start is MINUTES, not seconds (measured on this dev box: see"
Write-Host "   device\DEMO_RUNBOOK.md). Start it before the judges arrive."
$log = "$Here\logs\supervisor.stdout.log"
$args = @("-u", "$Here\supervisor.py", "--port", "$Port", "--target", "$Target") + $verifyArgs
$sup = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $Project `
        -RedirectStandardOutput $log -RedirectStandardError "$Here\logs\supervisor.stderr.log" `
        -PassThru -WindowStyle Hidden
Write-Host "   supervisor pid=$($sup.Id)  log=$log"

Write-Host "== 7. waiting for a REAL end-to-end probe to pass =="
& $Python "$Here\healthcheck.py" --port $Port --level status --wait 900 | Out-Null
$statusRc = $LASTEXITCODE
if ($statusRc -eq 0) {
  & $Python "$Here\healthcheck.py" --port $Port --level full --timeout 300
  $rc = $LASTEXITCODE
} else {
  $rc = $statusRc
}

if ($rc -eq 0) {
  Write-Host "----------------------------------------------------------------------"
  Write-Host "DEVICE READY - PASS.  http://127.0.0.1:$Port/healthz   target=$Target"
  Write-Host "Pre-flight numbers and triage: device\DEMO_RUNBOOK.md"
  Write-Host "----------------------------------------------------------------------"
} else {
  Write-Host "----------------------------------------------------------------------"
  Write-Host "DEVICE NOT READY - FAIL (rc=$rc). Do not start the demo."
  Write-Host "Look at: $log   then device\DEMO_RUNBOOK.md section 'Triage'"
  Write-Host "----------------------------------------------------------------------"
  exit 4
}

if ($Watchdog) {
  Write-Host "== 8. watchdog (attach mode, probe every 45 s) =="
  $wd = Start-Process -FilePath $Python -ArgumentList @("-u","$Here\watchdog.py","--port","$Port","--target","$Target","--attach","--interval","45","--deep-every","10") `
        -WorkingDirectory $Project -RedirectStandardOutput "$Here\logs\watchdog.stdout.log" `
        -RedirectStandardError "$Here\logs\watchdog.stderr.log" -PassThru -WindowStyle Hidden
  Write-Host "   watchdog pid=$($wd.Id)  log=$Here\logs\watchdog.log"
}
