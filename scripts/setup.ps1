param([string]$Python='py')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (Test-Path -LiteralPath '.venv') {throw '.venv already exists; reuse it or choose a clean environment.'}
    if ($Python -eq 'py') { & py -3.12 -m venv .venv } else { & $Python -m venv .venv }
    if ($LASTEXITCODE -ne 0) {throw 'Python 3.10-3.12 is required; pass -Python with its executable path.'}
    & .\.venv\Scripts\python.exe -m pip install -e . .\vendor\contest-harness-sdk-1.0.4.tar.gz
    if ($LASTEXITCODE -ne 0) {throw 'Dependency installation failed.'}
    & .\.venv\Scripts\python.exe scripts\check.py
    if ($LASTEXITCODE -ne 0) {throw 'Offline verification failed.'}
} finally {Pop-Location}
