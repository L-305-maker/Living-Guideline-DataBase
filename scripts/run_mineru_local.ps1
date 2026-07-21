param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [string]$OutputPath = "data/evidence_candidate/mineru_local"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$mineruExe = Join-Path $projectRoot ".venv-mineru-gpu\Scripts\mineru.exe"
$configPath = Join-Path $projectRoot "data\mineru\mineru.json"
$modelCache = Join-Path $projectRoot "data\mineru\modelscope"

if (-not (Test-Path -LiteralPath $mineruExe)) {
    throw "MinerU environment not found: $mineruExe"
}
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "MinerU model config not found: $configPath"
}

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$resolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath
} else {
    Join-Path $projectRoot $OutputPath
}
New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null

$env:MINERU_TOOLS_CONFIG_JSON = $configPath
$env:MINERU_MODEL_SOURCE = "local"
$env:MODELSCOPE_CACHE = $modelCache
$env:CUDA_VISIBLE_DEVICES = "0"
$env:MINERU_PROCESSING_WINDOW_SIZE = "1"
$env:MINERU_PDF_RENDER_THREADS = "1"
$env:MINERU_API_MAX_CONCURRENT_REQUESTS = "1"
$env:MINERU_INTRA_OP_NUM_THREADS = "8"
$env:MINERU_INTER_OP_NUM_THREADS = "2"

& $mineruExe -p $resolvedInput -o $resolvedOutput -b pipeline -m ocr -l ch
exit $LASTEXITCODE
