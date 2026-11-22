param([string]$Python='py')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        if ($Python -eq 'py') { & py -3.12 -m venv .venv } else { & $Python -m venv .venv }
        if ($LASTEXITCODE -ne 0) {throw 'Pass -Python with an installed Python 3.10-3.12 executable.'}
    }
    & .\.venv\Scripts\python.exe -m pip install -e '.[live]' .\vendor\contest-harness-sdk-1.0.4.tar.gz
    if ($LASTEXITCODE -ne 0) {throw 'Live dependency installation failed.'}
    & .\.venv\Scripts\python.exe scripts\check.py
    if ($LASTEXITCODE -ne 0) {throw 'Offline tests failed.'}
} finally {Pop-Location}
