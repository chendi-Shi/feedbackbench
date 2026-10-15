$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Run Python 3.12 scripts/bootstrap.py --download first.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $projectPython (Join-Path $PSScriptRoot 'scripts\run.py') -m embodied_research @args
    $runExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $runExitCode
