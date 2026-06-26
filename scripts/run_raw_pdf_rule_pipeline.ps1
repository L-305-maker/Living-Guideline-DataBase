param(
    [int]$ParseTimeout = 30,
    [int]$ProgressEvery = 25,
    [int]$ProcessLimit = 0,
    [switch]$StopAfterPdfProcess
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$ReportsDir = Join-Path $Root "data\processed\reports"
$LogPath = Join-Path $ReportsDir "raw_pdf_rule_pipeline_run.log"
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null

function Write-LogLine {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "s"), $Message
    Add-Content -Path $LogPath -Value $Line -Encoding utf8
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-LogLine "START $Name"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python -B -X utf8 @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        Write-LogLine "FAILED $Name exit=$ExitCode"
        throw "Step failed: $Name"
    }
    Write-LogLine "DONE $Name"
}

Set-Location $Root
Write-LogLine "Pipeline run started at $Root"

$ProcessArgs = @(
    "-m", "src.ingestion.crawler.processors.raw_pdf_incremental",
    "process-missing",
    "--parse-timeout", "$ParseTimeout",
    "--progress-every", "$ProgressEvery"
)
if ($ProcessLimit -gt 0) {
    $ProcessArgs += @("--limit", "$ProcessLimit")
}
Run-Step "process missing raw PDFs" $ProcessArgs
if ($StopAfterPdfProcess) {
    Write-LogLine "Pipeline run stopped after PDF processing by request"
    exit 0
}

Run-Step "merge origin files" @(
    "-m", "src.ingestion.crawler.processors.raw_pdf_incremental",
    "merge-origin"
)

Run-Step "clean merged origin" @(
    "-m", "src.pipeline.cleaning.source_cleaner",
    "--input", "data\origin\all_origin_merged.jsonl",
    "--output", "data\processed\cleaned\all_cleaned.jsonl"
)

Run-Step "gate cleaned records" @(
    "-m", "src.pipeline.cleaning.quality_gate",
    "--input", "data\processed\cleaned\all_cleaned.jsonl",
    "--ready-output", "data\processed\cleaned\all_cleaned.ready.jsonl",
    "--layout-repair-output", "data\processed\cleaned\all_cleaned.needs_layout_repair.jsonl",
    "--parse-failed-output", "data\processed\cleaned\all_cleaned.parse_failed.jsonl",
    "--skipped-output", "data\processed\cleaned\all_cleaned.skipped.jsonl",
    "--report-output", "data\processed\reports\all_cleaned_quality_gate_report.jsonl"
)

Run-Step "parse source blocks" @(
    "-m", "src.pipeline.parsing.structure_parser",
    "--input", "data\processed\cleaned\all_cleaned.ready.jsonl",
    "--output", "data\processed\blocks\all_blocks.jsonl"
)

Run-Step "route candidate inputs" @(
    "-m", "src.pipeline.extraction.routing.candidate_router",
    "--input", "data\processed\blocks\all_blocks.jsonl",
    "--output-dir", "data\processed\candidate_inputs",
    "--prefix", "all"
)

Run-Step "extract recommendation candidates" @(
    "-m", "src.pipeline.extraction.recommendation.candidate_extractor",
    "--input", "data\processed\candidate_inputs\all_recommendation_blocks.jsonl",
    "--candidates-output", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--traces-output", "data\processed\traces\all_recommendation_traces.jsonl"
)

Run-Step "extract pico questions" @(
    "-m", "src.pipeline.extraction.pico.question_extractor",
    "--input", "data\processed\candidate_inputs\all_pico_blocks.jsonl",
    "--output", "data\processed\knowledge\all_pico_questions.jsonl",
    "--traces-output", "data\processed\traces\all_pico_traces.jsonl"
)

Run-Step "extract grade candidates" @(
    "-m", "src.pipeline.extraction.grade.candidate_extractor",
    "--grade-blocks-input", "data\processed\candidate_inputs\all_grade_blocks.jsonl",
    "--recommendation-candidates-input", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--grade-candidates-output", "data\processed\candidates\all_grade_candidates.jsonl",
    "--traces-output", "data\processed\traces\all_grade_traces.jsonl"
)

Run-Step "extract evidence items" @(
    "-m", "src.pipeline.extraction.evidence.item_extractor",
    "--input", "data\processed\candidate_inputs\all_evidence_blocks.jsonl",
    "--output", "data\processed\knowledge\all_evidence_items.jsonl",
    "--traces-output", "data\processed\traces\all_evidence_traces.jsonl",
    "--picos-input", "data\processed\knowledge\all_pico_questions.jsonl",
    "--recommendations-input", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--include-unlinked"
)

Run-Step "build recommendation versions" @(
    "-m", "src.pipeline.extraction.versioning.recommendation_version_builder",
    "--recommendations-input", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--grades-input", "data\processed\candidates\all_grade_candidates.jsonl",
    "--picos-input", "data\processed\knowledge\all_pico_questions.jsonl",
    "--evidence-input", "data\processed\knowledge\all_evidence_items.jsonl",
    "--versions-output", "data\processed\knowledge\all_recommendation_versions.jsonl",
    "--report-output", "data\processed\reports\all_recommendation_version_report.jsonl"
)

Run-Step "build recommendation review queue" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "build-queue",
    "--entity-type", "recommendation_candidate",
    "--input", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--queue-output", "data\processed\manual_review\all_recommendation_candidate_queue.jsonl",
    "--summary-output", "data\processed\manual_review\all_recommendation_candidate_queue_summary.jsonl",
    "--include-pending",
    "--include-unclear-fields",
    "--confidence-threshold", "0.65"
)

Run-Step "build grade review queue" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "build-queue",
    "--entity-type", "grade_candidate",
    "--input", "data\processed\candidates\all_grade_candidates.jsonl",
    "--queue-output", "data\processed\manual_review\all_grade_candidate_queue.jsonl",
    "--summary-output", "data\processed\manual_review\all_grade_candidate_queue_summary.jsonl",
    "--include-pending",
    "--include-unclear-fields",
    "--confidence-threshold", "0.65"
)

Run-Step "build pico review queue" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "build-queue",
    "--entity-type", "pico_question",
    "--input", "data\processed\knowledge\all_pico_questions.jsonl",
    "--queue-output", "data\processed\manual_review\all_pico_question_queue.jsonl",
    "--summary-output", "data\processed\manual_review\all_pico_question_queue_summary.jsonl",
    "--include-unclear-fields",
    "--confidence-threshold", "0.65"
)

Run-Step "build evidence review queue" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "build-queue",
    "--entity-type", "evidence_item",
    "--input", "data\processed\knowledge\all_evidence_items.jsonl",
    "--queue-output", "data\processed\manual_review\all_evidence_item_queue.jsonl",
    "--summary-output", "data\processed\manual_review\all_evidence_item_queue_summary.jsonl",
    "--include-pending",
    "--include-unclear-fields",
    "--confidence-threshold", "0.65"
)

Write-LogLine "Pipeline run finished"
