[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Wheelhouse,
    [string]$Venv = ".venv",
    [string]$Requirements = "requirements.txt",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$wheelhousePath = (Resolve-Path -LiteralPath $Wheelhouse).Path
$requirementsCandidate = if ([System.IO.Path]::IsPathRooted($Requirements)) {
    $Requirements
} else {
    Join-Path $projectRoot $Requirements
}
$requirementsPath = (Resolve-Path -LiteralPath $requirementsCandidate).Path
$venvCandidate = if ([System.IO.Path]::IsPathRooted($Venv)) {
    $Venv
} else {
    Join-Path $projectRoot $Venv
}
$venvPath = [System.IO.Path]::GetFullPath($venvCandidate)
$bootstrap = Join-Path $PSScriptRoot "offline_bootstrap.py"

$arguments = @(
    $bootstrap,
    "install",
    "--wheelhouse", $wheelhousePath,
    "--requirements", $requirementsPath,
    "--venv", $venvPath
)
if ($DryRun) {
    $arguments += "--dry-run"
}

& python @arguments
exit $LASTEXITCODE
