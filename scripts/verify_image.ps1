param(
    [string]$Image='gas-a-harness:1.0.4',
    [string]$Docker='docker',
    [string]$OutputDirectory=''
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
if(-not $OutputDirectory){$OutputDirectory='../gas1/output/docker-verification/'+($Image -replace '[^A-Za-z0-9_.-]','_')}
$outputPath=if([IO.Path]::IsPathRooted($OutputDirectory)){[IO.Path]::GetFullPath($OutputDirectory)}else{[IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$utf8=[Text.UTF8Encoding]::new($false)
Push-Location -LiteralPath $projectRoot
try {
    $inspection=& $Docker image inspect $Image
    if($LASTEXITCODE -ne 0){throw 'Image inspection failed'}
    $metadata=($inspection | ConvertFrom-Json)[0]
    if($metadata.Os -ne 'linux' -or $metadata.Architecture -ne 'amd64'){throw 'Expected a linux/amd64 image'}
    $describeText=& $Docker run --rm --network none --read-only $Image describe
    if($LASTEXITCODE -ne 0){throw 'Harness describe failed'}
    $description=$describeText | ConvertFrom-Json
    if($description.protocol -ne 'contest.harness.workspace.v2' -or
       $description.model_access -ne 'host_gateway_rpc' -or
       $description.sdk_version -ne '1.0.4' -or
       ($description.tasks -join ',') -ne 'classification'){throw 'Unexpected Harness contract'}

    $probe=@'
import hashlib, importlib.util, json, os, platform
from pathlib import Path
import numpy
import gas_a.contest
excluded = ('live_access', 'live', 'live_commands', 'cloud_records')
assert all(importlib.util.find_spec('gas_a.' + name) is None for name in excluded)
assert os.geteuid() == 65534
assert numpy.__version__ == '1.26.4'
source = Path(gas_a.contest.__file__).parent
print(json.dumps({'python': platform.python_version(), 'numpy': numpy.__version__, 'uid': os.geteuid(),
                  'direct_clients_absent': True,
                  'source_sha256': {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(source.rglob('*.py'))}}))
'@
    $probeText=& $Docker run --rm --network none --read-only --entrypoint python3 $Image -B -c $probe
    if($LASTEXITCODE -ne 0){throw 'Image runtime isolation check failed'}
    $runtime=$probeText | ConvertFrom-Json
    foreach($entry in $runtime.source_sha256.PSObject.Properties){
        $local=Join-Path $projectRoot ('src/gas_a/' + $entry.Name)
        if((Get-FileHash -LiteralPath $local -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value){
            throw ('Image contains stale source: ' + $entry.Name)
        }
    }

    # 仅挂载测试替身和公开数据；实际执行的参赛代码、
    # SDK 与依赖均来自实际构建的镜像。
    $arguments=@('run','--rm','--network','none','--read-only','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--pids-limit','128','--memory','1g','--cpus','2',
        '--tmpfs','/tmp:rw,noexec,nosuid,size=256m',
        '--mount',('type=bind,source='+(Join-Path $projectRoot 'tests')+',target=/app/tests,readonly'),
        '--mount',('type=bind,source='+(Join-Path $projectRoot 'data/official_a')+',target=/app/data/official_a,readonly'),
        '--mount',('type=bind,source='+(Join-Path $projectRoot 'Dockerfile')+',target=/app/Dockerfile,readonly'),
        '--env','PYTHONPATH=/app/tests','--entrypoint','python3',$Image,
        '-B','-m','unittest','test_protocol','-v')
    $testOutput=& $Docker @arguments 2>&1
    $testCode=$LASTEXITCODE
    $testText=($testOutput | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine
    [IO.File]::WriteAllText((Join-Path $outputPath 'container-tests.txt'),$testText+[Environment]::NewLine,$utf8)
    Write-Output $testText
    if($testCode -ne 0){throw ('Container SDK tests failed; see '+(Join-Path $outputPath 'container-tests.txt'))}
    $report=[ordered]@{
        image=$Image; image_id=$metadata.Id; os=$metadata.Os; architecture=$metadata.Architecture
        image_bytes=$metadata.Size; entrypoint=$metadata.Config.Entrypoint; runtime=$runtime
        description=$description; passed=$true; container_sdk_tests_exit_code=$testCode
        network='none'; root_filesystem='read-only'; external_model_calls=0
        scope='Actual image: describe, non-root runtime, source hashes, independent prediction, SDK continuation/candidate/state/ack, single eligible choice, UTF-8 reason and corrected decision. Host/model replies are offline fixtures.'
        official_platform_verified=$false; created_at=[DateTime]::UtcNow.ToString('o')
    }
    [IO.File]::WriteAllText((Join-Path $outputPath 'verification.json'),($report | ConvertTo-Json -Depth 12)+[Environment]::NewLine,$utf8)
    Write-Output ('Verified image: '+$metadata.Id)
} finally {Pop-Location}
