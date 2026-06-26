param(
    [ValidateSet("DryRun", "Call")]
    [string]$LlmMode = "DryRun",
    [string]$OriginInput = "data\origin\all_origin_merged.jsonl",
    [string]$LlmPriorities = "P0",
    [int]$LlmLimit = 20,
    [string]$LlmModel = "deepseek-v4-flash",
    [ValidateSet("responses", "chat_completions")]
    [string]$ApiFormat = "chat_completions",
    [string]$ApiUrl = "https://api.deepseek.com",
    [string]$ApiKeyEnv = "API_KEY",
    [string]$EnvFile = ".env",
    [int]$TimeoutSeconds = 60,
    [int]$ProgressEvery = 50,
    [int]$FlushEvery = 1,
    [switch]$FreshLlmOutputs,
    [switch]$IncludeP3,
    [switch]$BuildFinalAfterManualReview,
    [string]$ReviewedRecommendationQueue = "",
    [string]$ReviewedGradeQueue = "",
    [string]$ReviewedPicoQueue = "",
    [string]$ReviewedEvidenceQueue = ""
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$ReportsDir = Join-Path $Root "data\processed\reports"
$StepLogDir = Join-Path $ReportsDir "pipeline_step_logs"
$LogPath = Join-Path $ReportsDir "guideline_review_pipeline_run.log"
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
New-Item -ItemType Directory -Force -Path $StepLogDir | Out-Null

function Write-LogLine {
    param([string]$Message)
    $Line = "[{0}] {1}" -f (Get-Date -Format "s"), $Message
    Add-Content -Path $LogPath -Value $Line -Encoding utf8
    Write-Host $Line
}

function Run-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $SafeName = ($Name -replace "[^A-Za-z0-9_.-]", "_")
    $StepLog = Join-Path $StepLogDir "$SafeName.log"
    Write-LogLine "START $Name"
    & python -B -X utf8 @Arguments *> $StepLog
    $ExitCode = $LASTEXITCODE
    if (Test-Path $StepLog) {
        Get-Content -Path $StepLog | Tee-Object -FilePath $LogPath -Append
    }
    if ($ExitCode -ne 0) {
        Write-LogLine "FAILED $Name exit=$ExitCode"
        throw "Step failed: $Name"
    }
    Write-LogLine "DONE $Name"
}

function Require-ReviewedQueue {
    param(
        [string]$Name,
        [string]$Path
    )
    if (-not $Path) {
        throw "$Name is required when -BuildFinalAfterManualReview is enabled."
    }
    if (-not (Test-Path $Path)) {
        throw "$Name does not exist: $Path"
    }
}

Set-Location $Root
Write-LogLine "Pipeline started. mode=$LlmMode origin=$OriginInput priorities=$LlmPriorities limit=$LlmLimit"

$LlmStreamingArgs = @(
    "--progress-every", "$ProgressEvery",
    "--flush-every", "$FlushEvery"
)
if (-not $FreshLlmOutputs) {
    $LlmStreamingArgs += @("--resume", "--append-output")
}

Run-Step "clean merged origin" @(
    "-m", "src.pipeline.cleaning.source_cleaner",
    "--input", $OriginInput,
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

Run-Step "rule extract recommendation candidates" @(
    "-m", "src.pipeline.extraction.recommendation.candidate_extractor",
    "--input", "data\processed\candidate_inputs\all_recommendation_blocks.jsonl",
    "--candidates-output", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--traces-output", "data\processed\traces\all_recommendation_traces.jsonl"
)

Run-Step "rule extract pico questions" @(
    "-m", "src.pipeline.extraction.pico.question_extractor",
    "--input", "data\processed\candidate_inputs\all_pico_blocks.jsonl",
    "--output", "data\processed\knowledge\all_pico_questions.jsonl",
    "--traces-output", "data\processed\traces\all_pico_traces.jsonl"
)

Run-Step "rule extract grade candidates" @(
    "-m", "src.pipeline.extraction.grade.candidate_extractor",
    "--grade-blocks-input", "data\processed\candidate_inputs\all_grade_blocks.jsonl",
    "--recommendation-candidates-input", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--grade-candidates-output", "data\processed\candidates\all_grade_candidates.jsonl",
    "--traces-output", "data\processed\traces\all_grade_traces.jsonl"
)

Run-Step "rule extract evidence items" @(
    "-m", "src.pipeline.extraction.evidence.item_extractor",
    "--input", "data\processed\candidate_inputs\all_evidence_blocks.jsonl",
    "--output", "data\processed\knowledge\all_evidence_items.jsonl",
    "--traces-output", "data\processed\traces\all_evidence_traces.jsonl",
    "--picos-input", "data\processed\knowledge\all_pico_questions.jsonl",
    "--recommendations-input", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--include-unlinked"
)

$QueueArgs = @(
    "-m", "src.pipeline.llm_review.queue.builder",
    "--recommendations-input", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--grades-input", "data\processed\candidates\all_grade_candidates.jsonl",
    "--queue-output", "data\processed\llm_review\all_llm_review_queue.jsonl",
    "--summary-output", "data\processed\llm_review\all_llm_review_queue_summary.jsonl"
)
if ($IncludeP3) {
    $QueueArgs += "--include-p3"
}
Run-Step "build llm review queue" $QueueArgs

if ($LlmMode -eq "DryRun") {
    Run-Step "dry-run recommendation llm prompts" @(
        "-m", "src.pipeline.llm_review.queue.runner",
        "--mode", "dry-run",
        "--queue-input", "data\processed\llm_review\all_llm_review_queue.jsonl",
        "--prompts-output", "data\processed\llm_review\all_recommendation_llm_prompts.jsonl",
        "--summary-output", "data\processed\llm_review\all_recommendation_llm_prompts_summary.jsonl",
        "--priorities", $LlmPriorities,
        "--task-types", "recommendation_candidate_review",
        "--limit", "$LlmLimit"
    )
    Run-Step "dry-run grade llm prompts" @(
        "-m", "src.pipeline.llm_review.queue.runner",
        "--mode", "dry-run",
        "--queue-input", "data\processed\llm_review\all_llm_review_queue.jsonl",
        "--prompts-output", "data\processed\llm_review\all_grade_llm_prompts.jsonl",
        "--summary-output", "data\processed\llm_review\all_grade_llm_prompts_summary.jsonl",
        "--priorities", $LlmPriorities,
        "--task-types", "grade_candidate_review",
        "--limit", "$LlmLimit"
    )
    Write-LogLine "DryRun finished. LLM prompts are ready; stop before human review because no LLM outputs were applied."
    exit 0
}

$RecommendationCallArgs = @(
    "-m", "src.pipeline.llm_review.queue.runner",
    "--mode", "call",
    "--queue-input", "data\processed\llm_review\all_llm_review_queue.jsonl",
    "--outputs-output", "data\processed\llm_review\all_recommendation_llm_outputs.jsonl",
    "--traces-output", "data\processed\traces\all_recommendation_llm_traces.jsonl",
    "--summary-output", "data\processed\llm_review\all_recommendation_llm_outputs_summary.jsonl",
    "--priorities", $LlmPriorities,
    "--task-types", "recommendation_candidate_review",
    "--limit", "$LlmLimit",
    "--model", $LlmModel,
    "--api-url", $ApiUrl,
    "--api-format", $ApiFormat,
    "--api-key-env", $ApiKeyEnv,
    "--env-file", $EnvFile,
    "--timeout-seconds", "$TimeoutSeconds"
)
Run-Step "call recommendation llm review" ($RecommendationCallArgs + $LlmStreamingArgs)

$GradeCallArgs = @(
    "-m", "src.pipeline.llm_review.queue.runner",
    "--mode", "call",
    "--queue-input", "data\processed\llm_review\all_llm_review_queue.jsonl",
    "--outputs-output", "data\processed\llm_review\all_grade_llm_outputs.jsonl",
    "--traces-output", "data\processed\traces\all_grade_llm_traces.jsonl",
    "--summary-output", "data\processed\llm_review\all_grade_llm_outputs_summary.jsonl",
    "--priorities", $LlmPriorities,
    "--task-types", "grade_candidate_review",
    "--limit", "$LlmLimit",
    "--model", $LlmModel,
    "--api-url", $ApiUrl,
    "--api-format", $ApiFormat,
    "--api-key-env", $ApiKeyEnv,
    "--env-file", $EnvFile,
    "--timeout-seconds", "$TimeoutSeconds"
)
Run-Step "call grade llm review" ($GradeCallArgs + $LlmStreamingArgs)

Run-Step "build guideline profiles" @(
    "-m", "src.pipeline.parsing.guideline_profiler",
    "--inputs", "data\processed\cleaned\all_cleaned.ready.jsonl",
    "--profiles-output", "data\processed\profiles\guideline_profiles.jsonl",
    "--report-output", "data\processed\reports\guideline_profiles_summary.jsonl"
)

Run-Step "recommendation llm auto qc" @(
    "-m", "src.pipeline.llm_review.auto_qc.recommendation",
    "--outputs-input", "data\processed\llm_review\all_recommendation_llm_outputs.jsonl",
    "--recommendations-input", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--grades-input", "data\processed\candidates\all_grade_candidates.jsonl",
    "--profiles-input", "data\processed\profiles\guideline_profiles.jsonl",
    "--qc-output", "data\processed\llm_review\all_recommendation_auto_qc.jsonl",
    "--summary-output", "data\processed\llm_review\all_recommendation_auto_qc_summary.jsonl"
)

Run-Step "grade llm auto qc" @(
    "-m", "src.pipeline.llm_review.auto_qc.grade",
    "--outputs-input", "data\processed\llm_review\all_grade_llm_outputs.jsonl",
    "--grades-input", "data\processed\candidates\all_grade_candidates.jsonl",
    "--recommendations-input", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--profiles-input", "data\processed\profiles\guideline_profiles.jsonl",
    "--qc-output", "data\processed\llm_review\all_grade_auto_qc.jsonl",
    "--summary-output", "data\processed\llm_review\all_grade_auto_qc_summary.jsonl"
)

Run-Step "enhance recommendation candidates" @(
    "-m", "src.pipeline.extraction.enhancement.candidate_enhancer",
    "--candidates-input", "data\processed\candidates\all_recommendation_candidates.jsonl",
    "--llm-outputs-input", "data\processed\llm_review\all_recommendation_llm_outputs.jsonl",
    "--auto-qc-input", "data\processed\llm_review\all_recommendation_auto_qc.jsonl",
    "--enhanced-output", "data\processed\candidates\all_recommendation_candidates_enhanced.jsonl",
    "--summary-output", "data\processed\reports\all_recommendation_candidate_enhancement_summary.jsonl"
)

Run-Step "enhance grade candidates" @(
    "-m", "src.pipeline.extraction.enhancement.grade_candidate_enhancer",
    "--candidates-input", "data\processed\candidates\all_grade_candidates.jsonl",
    "--llm-outputs-input", "data\processed\llm_review\all_grade_llm_outputs.jsonl",
    "--auto-qc-input", "data\processed\llm_review\all_grade_auto_qc.jsonl",
    "--enhanced-output", "data\processed\candidates\all_grade_candidates_enhanced.jsonl",
    "--summary-output", "data\processed\reports\all_grade_candidate_enhancement_summary.jsonl"
)

Run-Step "build recommendation human review queue" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "build-queue",
    "--entity-type", "recommendation_candidate",
    "--input", "data\processed\candidates\all_recommendation_candidates_enhanced.jsonl",
    "--queue-output", "data\processed\manual_review\all_recommendation_candidate_queue.jsonl",
    "--summary-output", "data\processed\manual_review\all_recommendation_candidate_queue_summary.jsonl",
    "--include-pending",
    "--include-unclear-fields",
    "--confidence-threshold", "0.65"
)

Run-Step "build grade human review queue" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "build-queue",
    "--entity-type", "grade_candidate",
    "--input", "data\processed\candidates\all_grade_candidates_enhanced.jsonl",
    "--queue-output", "data\processed\manual_review\all_grade_candidate_queue.jsonl",
    "--summary-output", "data\processed\manual_review\all_grade_candidate_queue_summary.jsonl",
    "--include-pending",
    "--include-unclear-fields",
    "--confidence-threshold", "0.65"
)

Run-Step "build pico human review queue" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "build-queue",
    "--entity-type", "pico_question",
    "--input", "data\processed\knowledge\all_pico_questions.jsonl",
    "--queue-output", "data\processed\manual_review\all_pico_question_queue.jsonl",
    "--summary-output", "data\processed\manual_review\all_pico_question_queue_summary.jsonl",
    "--include-unclear-fields",
    "--confidence-threshold", "0.65"
)

Run-Step "build evidence human review queue" @(
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

if (-not $BuildFinalAfterManualReview) {
    Write-LogLine "Pipeline stopped after human review queue creation. Review queues must be reviewed and applied before final version build."
    exit 0
}

Require-ReviewedQueue "ReviewedRecommendationQueue" $ReviewedRecommendationQueue
Require-ReviewedQueue "ReviewedGradeQueue" $ReviewedGradeQueue
Require-ReviewedQueue "ReviewedPicoQueue" $ReviewedPicoQueue
Require-ReviewedQueue "ReviewedEvidenceQueue" $ReviewedEvidenceQueue

Run-Step "apply recommendation human reviews" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "apply",
    "--entity-type", "recommendation_candidate",
    "--base-input", "data\processed\candidates\all_recommendation_candidates_enhanced.jsonl",
    "--reviewed-input", $ReviewedRecommendationQueue,
    "--output", "data\processed\candidates\all_recommendation_candidates_reviewed.jsonl",
    "--summary-output", "data\processed\manual_review\all_recommendation_candidate_apply_summary.jsonl"
)

Run-Step "apply grade human reviews" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "apply",
    "--entity-type", "grade_candidate",
    "--base-input", "data\processed\candidates\all_grade_candidates_enhanced.jsonl",
    "--reviewed-input", $ReviewedGradeQueue,
    "--output", "data\processed\candidates\all_grade_candidates_reviewed.jsonl",
    "--summary-output", "data\processed\manual_review\all_grade_candidate_apply_summary.jsonl"
)

Run-Step "apply pico human reviews" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "apply",
    "--entity-type", "pico_question",
    "--base-input", "data\processed\knowledge\all_pico_questions.jsonl",
    "--reviewed-input", $ReviewedPicoQueue,
    "--output", "data\processed\knowledge\all_pico_questions_reviewed.jsonl",
    "--summary-output", "data\processed\manual_review\all_pico_question_apply_summary.jsonl"
)

Run-Step "apply evidence human reviews" @(
    "-m", "src.pipeline.review.manual_review_gate",
    "apply",
    "--entity-type", "evidence_item",
    "--base-input", "data\processed\knowledge\all_evidence_items.jsonl",
    "--reviewed-input", $ReviewedEvidenceQueue,
    "--output", "data\processed\knowledge\all_evidence_items_reviewed.jsonl",
    "--summary-output", "data\processed\manual_review\all_evidence_item_apply_summary.jsonl"
)

Run-Step "build final recommendation versions" @(
    "-m", "src.pipeline.extraction.versioning.recommendation_version_builder",
    "--recommendations-input", "data\processed\candidates\all_recommendation_candidates_reviewed.jsonl",
    "--grades-input", "data\processed\candidates\all_grade_candidates_reviewed.jsonl",
    "--picos-input", "data\processed\knowledge\all_pico_questions_reviewed.jsonl",
    "--evidence-input", "data\processed\knowledge\all_evidence_items_reviewed.jsonl",
    "--versions-output", "data\processed\knowledge\all_recommendation_versions.jsonl",
    "--report-output", "data\processed\reports\all_recommendation_version_report.jsonl"
)

Write-LogLine "Pipeline finished with final version build."
