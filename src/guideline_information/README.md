# Guideline Information Extraction

## Quick Run

Prepare a frozen experiment batch:

```bash
python -m src.guideline_information prepare-experiment --pilot-id idsa_real_pilot_v1 --experiment-id idsa_real_model_experiment_v2
```

Run a real model pilot after model configuration is available in environment or `.env`:

```bash
python -m src.guideline_information run-pilot --pilot-id idsa_real_pilot_v1 --run-id idsa_real_model_stage_a_v2 --client openai-compatible --experiment-id idsa_real_model_experiment_v2 --input data/guideline_information/experiments/idsa_real_model_experiment_v2/stage_a_candidates.jsonl
```

Create the V0.1 human review package:

```bash
python -m src.guideline_information prepare-review --experiment-id idsa_real_model_experiment_v2 --review-batch-id idsa_stage_a_review_v1
```

Import completed human review files:

```bash
python -m src.guideline_information import-review --review-batch-id idsa_stage_a_review_v1 --samples-csv data/guideline_information/review_batches/idsa_stage_a_review_v1/completed_review_samples.csv --fields-csv data/guideline_information/review_batches/idsa_stage_a_review_v1/completed_review_fields.csv --review-origin human
```

Build Silver examples:

```bash
python -m src.guideline_information build-silver --review-batch-id idsa_stage_a_review_v1
```

Evaluate real-model, human-reviewed results:

```bash
python -m src.guideline_information evaluate-review --review-batch-id idsa_stage_a_review_v1 --real-only
```

Run the automatic part of the V0.1 loop:

```bash
python -m src.guideline_information complete-v0 --experiment-id idsa_real_model_experiment_v2 --review-batch-id idsa_stage_a_review_v1
```

If completed human review files are absent, `complete-v0` stops at `READY_FOR_HUMAN_REVIEW` and prints the files to fill.

## Data Flow

```text
Candidate
-> Extraction
-> Verification
-> Validation
-> Human Review
-> Silver
```

## File Locations

- Experiments: `data/guideline_information/experiments/<experiment_id>/`
- Real run artifacts: `data/pilots/guideline_information_real/runs/<run_id>/`
- Review batches: `data/guideline_information/review_batches/<review_batch_id>/`
- Review package: `review_batch_manifest.json`, `review_samples.csv`, `review_fields.csv`, `review_instructions.md`, `source_snapshots.jsonl`
- Imported review: `review_decisions.jsonl`, `field_review_decisions.jsonl`, `review_import_report.json`
- Silver output: `silver_recommendations.jsonl`, `silver_manifest.json`
- Evaluation output: `evaluation_v0_1.json`, `evaluation_v0_1.md`, `error_examples_v0_1.jsonl`
- Summary report: `reports/guideline_information_loop_v0_1.md`

## Limits

- Current V0.1 supports single-reviewer Silver only.
- Double-review Gold is not implemented.
- RecommendationVersion is not implemented.
- Cross-version alignment is not implemented.
- PostgreSQL review storage and Web review UI are intentionally out of scope.
- Prompts, canonical schemas, candidate builder, span medical policy, and route policy are not changed by the V0.1 review loop.
