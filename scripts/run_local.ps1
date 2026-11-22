param([int]$Rounds=10, [string]$Python='python')
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $env:PYTHONPATH=Join-Path $projectRoot 'src'
    $env:PYTHONDONTWRITEBYTECODE='1'
    & $Python -m gas_a --config configs/a.json run --rounds $Rounds
    if($LASTEXITCODE -ne 0){throw 'A lifecycle run failed'}
} finally {Pop-Location}

