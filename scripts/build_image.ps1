param([string]$Image='gas-a-harness:1.0.4', [string]$Archive='../gas1/output/docker/gas-a-harness.tar', [string]$Docker='docker', [string]$Proxy='', [ValidateRange(1,4096)][double]$MaxArchiveMiB=60)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$previousHttpProxy=$env:HTTP_PROXY
$previousHttpsProxy=$env:HTTPS_PROXY
if($Proxy){
    $proxyUri=[uri]$Proxy
    if(-not $proxyUri.IsAbsoluteUri -or $proxyUri.Scheme -notin @('http','https')){throw 'Proxy must be an absolute HTTP/HTTPS URL'}
    $env:HTTP_PROXY=$Proxy
    $env:HTTPS_PROXY=$Proxy
}
Push-Location -LiteralPath $projectRoot
try {
    & $Docker build --platform linux/amd64 -t $Image .
    if($LASTEXITCODE -ne 0){throw 'Docker build failed'}
    & (Join-Path $PSScriptRoot 'verify_image.ps1') -Image $Image -Docker $Docker
    $archivePath=if([IO.Path]::IsPathRooted($Archive)){[IO.Path]::GetFullPath($Archive)}else{[IO.Path]::GetFullPath((Join-Path $projectRoot $Archive))}
    New-Item -ItemType Directory -Path (Split-Path -Parent $archivePath) -Force | Out-Null
    & $Docker save -o $archivePath $Image
    if($LASTEXITCODE -ne 0){throw 'Docker save failed'}
    $archiveBytes=(Get-Item -LiteralPath $archivePath).Length
    if($archiveBytes -gt $MaxArchiveMiB*1MB){throw ('Archive exceeds '+$MaxArchiveMiB+' MiB: '+$archiveBytes+' bytes')}
    Write-Output ('Upload archive: {0:N2} MiB' -f ($archiveBytes/1MB))
    Get-FileHash -LiteralPath $archivePath -Algorithm SHA256
} finally {
    Pop-Location
    $env:HTTP_PROXY=$previousHttpProxy
    $env:HTTPS_PROXY=$previousHttpsProxy
}
