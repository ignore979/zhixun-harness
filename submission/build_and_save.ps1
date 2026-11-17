param(
    [string]$Python = '',
    [string]$ImageTag = 'zhixun-harness:trial-v1',
    [string]$ContractPath = ''
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $ProjectRoot '指导\环境\Scripts\python.exe' }
if (-not $ContractPath) { $ContractPath = Join-Path $PSScriptRoot 'runtime_contract.json' }
$BuildStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$ContextPath = Join-Path $ProjectRoot ('指导\临时\platform-build-' + $BuildStamp)
$ExportFolder = Join-Path $ProjectRoot '指导\交付\平台试运行准备-20261116'
New-Item -ItemType Directory -Force -Path $ExportFolder | Out-Null
$ArchivePath = Join-Path $ExportFolder ('zhixun-harness-linux-amd64-' + $BuildStamp + '.tar')
if (Test-Path -LiteralPath $ArchivePath) { throw 'Export destination already exists; choose a new version.' }
& $Python -B (Join-Path $PSScriptRoot 'prepare_image_context.py') --root $ProjectRoot --destination $ContextPath --contract $ContractPath
if ($LASTEXITCODE -ne 0) { throw 'Official runtime adapter is not ready. No image was built or exported.' }
& docker version
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is unavailable. Fix WSL/Docker Desktop first.' }
& docker build --platform linux/amd64 --tag $ImageTag $ContextPath
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed; no upload archive was produced.' }
$ImagePlatform = & docker image inspect --format '{{.Os}}/{{.Architecture}}' $ImageTag
if ($LASTEXITCODE -ne 0 -or $ImagePlatform.Trim() -ne 'linux/amd64') { throw 'Image platform must be linux/amd64.' }
$Contract = Get-Content -LiteralPath $ContractPath -Raw | ConvertFrom-Json
$SmokeArgs = @($Contract.smoke_test_args)
& docker run --rm --network none $ImageTag @SmokeArgs
if ($LASTEXITCODE -ne 0) { throw 'Container smoke test failed; archive export stopped.' }
& docker save --output $ArchivePath $ImageTag
if ($LASTEXITCODE -ne 0) { throw 'docker save failed.' }
& $Python -B (Join-Path $PSScriptRoot 'validate_image_archive.py') $ArchivePath
if ($LASTEXITCODE -ne 0) { throw 'Exported image archive is invalid; do not upload it.' }
Write-Output ('Image archive exported: ' + $ArchivePath)
Write-Output 'No upload or evaluation was started. Official full-run compatibility requires separate validation.'
