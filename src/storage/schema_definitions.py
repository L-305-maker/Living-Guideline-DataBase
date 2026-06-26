from __future__ import annotations


STORAGE_SCHEMA_VERSION = "storage_schema_constraints_v1"


def _sql_literal_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _add_check_constraint(table: str, constraint: str, condition: str) -> str:
    return f"""
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = '{constraint}'
    ) THEN
        ALTER TABLE {table}
        ADD CONSTRAINT {constraint}
        CHECK ({condition})
        NOT VALID;
    END IF;
END $$;
"""


TABLE_DDL = [
    "                CREATE TABLE IF NOT EXISTS storage_schema_migrations (\n                    migration_id TEXT PRIMARY KEY,\n                    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS guidelines (\n                    guideline_id TEXT PRIMARY KEY,\n                    title TEXT NOT NULL,\n                    disease_area TEXT,\n                    guideline_type TEXT NOT NULL DEFAULT 'standard',\n                    status TEXT NOT NULL DEFAULT 'active',\n                    source TEXT,\n                    issuer TEXT,\n                    publication_url TEXT,\n                    pdf_url TEXT,\n                    current_version TEXT,\n                    update_frequency TEXT,\n                    published_date TEXT,\n                    last_updated_at TEXT,\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS papers (\n                    paper_id TEXT PRIMARY KEY,\n                    title TEXT NOT NULL,\n                    pmid TEXT,\n                    doi TEXT,\n                    abstract TEXT,\n                    authors JSONB NOT NULL DEFAULT '[]'::jsonb,\n                    journal TEXT,\n                    publication_date TEXT,\n                    article_type TEXT,\n                    source_database TEXT,\n                    url TEXT,\n                    first_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n                    screening_status TEXT NOT NULL DEFAULT 'unscreened'\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS cleaned_records (\n                    record_id TEXT PRIMARY KEY,\n                    guideline_id TEXT REFERENCES guidelines(guideline_id) ON UPDATE CASCADE,\n                    paper_id TEXT REFERENCES papers(paper_id) ON UPDATE CASCADE,\n                    source TEXT,\n                    issuer TEXT,\n                    title TEXT,\n                    url TEXT,\n                    published_year TEXT,\n                    raw_pdf_path TEXT,\n                    url_provenance TEXT,\n                    content TEXT NOT NULL,\n                    tables JSONB NOT NULL DEFAULT '[]'::jsonb,\n                    table_count INTEGER,\n                    table_row_count INTEGER,\n                    table_cell_count INTEGER,\n                    guideline_seed JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    paper_seed JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    direct_extraction JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    raw_record JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS model_traces (\n                    model_trace_id TEXT PRIMARY KEY,\n                    task_type TEXT NOT NULL,\n                    method TEXT NOT NULL,\n                    model_name TEXT NOT NULL,\n                    input_entity_type TEXT NOT NULL,\n                    input_entity_id TEXT NOT NULL,\n                    input_text TEXT NOT NULL,\n                    model_version TEXT,\n                    prompt_version TEXT,\n                    raw_output JSONB,\n                    parsed_output JSONB,\n                    confidence DOUBLE PRECISION,\n                    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    token_usage JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    runtime_ms INTEGER,\n                    code_version TEXT,\n                    success BOOLEAN NOT NULL DEFAULT TRUE,\n                    error_message TEXT,\n                    human_verified BOOLEAN NOT NULL DEFAULT FALSE,\n                    verified_by TEXT,\n                    verified_at TEXT,\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS pico_questions (\n                    pico_id TEXT PRIMARY KEY,\n                    guideline_id TEXT NOT NULL REFERENCES guidelines(guideline_id) ON UPDATE CASCADE,\n                    clinical_question TEXT NOT NULL,\n                    population TEXT NOT NULL,\n                    intervention TEXT NOT NULL,\n                    comparator TEXT,\n                    outcomes JSONB NOT NULL DEFAULT '[]'::jsonb,\n                    priority TEXT,\n                    status TEXT NOT NULL DEFAULT 'active',\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS recommendation_candidates (\n                    candidate_id TEXT PRIMARY KEY,\n                    model_trace_id TEXT NOT NULL REFERENCES model_traces(model_trace_id) ON UPDATE CASCADE,\n                    statement TEXT NOT NULL,\n                    chunk_id TEXT,\n                    source_record_id TEXT REFERENCES cleaned_records(record_id) ON UPDATE CASCADE,\n                    guideline_id TEXT REFERENCES guidelines(guideline_id) ON UPDATE CASCADE,\n                    pico_id TEXT REFERENCES pico_questions(pico_id) ON UPDATE CASCADE,\n                    source TEXT,\n                    issuer TEXT,\n                    title TEXT,\n                    source_url TEXT,\n                    source_section TEXT,\n                    direction TEXT NOT NULL DEFAULT 'unclear',\n                    strength TEXT NOT NULL DEFAULT 'unclear',\n                    population TEXT,\n                    intervention TEXT,\n                    comparator TEXT,\n                    outcome TEXT,\n                    rationale TEXT,\n                    contraindications TEXT,\n                    adverse_effects TEXT,\n                    raw_text TEXT,\n                    char_start INTEGER,\n                    char_end INTEGER,\n                    page INTEGER,\n                    confidence DOUBLE PRECISION,\n                    extraction_method TEXT NOT NULL DEFAULT 'rule',\n                    status TEXT NOT NULL DEFAULT 'pending',\n                    duplicate_of_candidate_id TEXT,\n                    review_note TEXT,\n                    reviewed_by TEXT,\n                    reviewed_at TEXT,\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS grade_candidates (\n                    grade_candidate_id TEXT PRIMARY KEY,\n                    model_trace_id TEXT NOT NULL REFERENCES model_traces(model_trace_id) ON UPDATE CASCADE,\n                    source_grade_raw TEXT NOT NULL,\n                    recommendation_candidate_id TEXT REFERENCES recommendation_candidates(candidate_id) ON UPDATE CASCADE,\n                    chunk_id TEXT,\n                    guideline_id TEXT REFERENCES guidelines(guideline_id) ON UPDATE CASCADE,\n                    grade_system TEXT NOT NULL DEFAULT 'unknown',\n                    mapped_from_system TEXT,\n                    mapping_note TEXT,\n                    certainty TEXT NOT NULL DEFAULT 'unclear',\n                    strength TEXT NOT NULL DEFAULT 'unclear',\n                    direction TEXT NOT NULL DEFAULT 'unclear',\n                    outcome_name TEXT,\n                    risk_of_bias TEXT NOT NULL DEFAULT 'not_reported',\n                    inconsistency TEXT NOT NULL DEFAULT 'not_reported',\n                    indirectness TEXT NOT NULL DEFAULT 'not_reported',\n                    imprecision TEXT NOT NULL DEFAULT 'not_reported',\n                    publication_bias TEXT NOT NULL DEFAULT 'not_reported',\n                    judgement_rationale TEXT,\n                    confidence DOUBLE PRECISION,\n                    extraction_method TEXT NOT NULL DEFAULT 'rule',\n                    status TEXT NOT NULL DEFAULT 'pending',\n                    review_note TEXT,\n                    reviewed_by TEXT,\n                    reviewed_at TEXT,\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS recommendation_versions (\n                    recommendation_version_id TEXT PRIMARY KEY,\n                    recommendation_candidate_id TEXT NOT NULL REFERENCES recommendation_candidates(candidate_id) ON UPDATE CASCADE,\n                    guideline_id TEXT NOT NULL REFERENCES guidelines(guideline_id) ON UPDATE CASCADE,\n                    version_number TEXT NOT NULL,\n                    recommendation_text TEXT NOT NULL,\n                    pico_id TEXT REFERENCES pico_questions(pico_id) ON UPDATE CASCADE,\n                    record_id TEXT REFERENCES cleaned_records(record_id) ON UPDATE CASCADE,\n                    grade_candidate_id TEXT REFERENCES grade_candidates(grade_candidate_id) ON UPDATE CASCADE,\n                    profile_id TEXT,\n                    issuer TEXT,\n                    grading_system TEXT,\n                    quality_status TEXT NOT NULL DEFAULT 'needs_review',\n                    recommendation_code TEXT,\n                    direction TEXT NOT NULL DEFAULT 'unclear',\n                    strength TEXT NOT NULL DEFAULT 'unclear',\n                    certainty TEXT NOT NULL DEFAULT 'unclear',\n                    rationale TEXT,\n                    remarks TEXT,\n                    population TEXT,\n                    intervention TEXT,\n                    comparator TEXT,\n                    outcome_summary TEXT,\n                    source_text TEXT,\n                    source_span TEXT,\n                    source_span_ref TEXT,\n                    start_char INTEGER,\n                    end_char INTEGER,\n                    source_section TEXT,\n                    source_url TEXT,\n                    change_type TEXT NOT NULL DEFAULT 'new',\n                    change_summary TEXT,\n                    normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n                    published_at TEXT\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS recommendations (\n                    recommendation_id TEXT PRIMARY KEY,\n                    guideline_id TEXT NOT NULL REFERENCES guidelines(guideline_id) ON UPDATE CASCADE,\n                    current_version_id TEXT REFERENCES recommendation_versions(recommendation_version_id) ON UPDATE CASCADE,\n                    status TEXT NOT NULL DEFAULT 'active',\n                    recommendation_code TEXT,\n                    title TEXT,\n                    population TEXT,\n                    intervention TEXT,\n                    comparator TEXT,\n                    direction TEXT NOT NULL DEFAULT 'unclear',\n                    strength TEXT NOT NULL DEFAULT 'unclear',\n                    certainty TEXT NOT NULL DEFAULT 'unclear',\n                    first_created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n                    last_published_at TEXT,\n                    last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n                    withdrawn_at TEXT,\n                    withdrawn_reason TEXT,\n                    superseded_by_recommendation_id TEXT REFERENCES recommendations(recommendation_id) ON UPDATE CASCADE,\n                    normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS recommendation_version_review_queue (\n                    review_item_id TEXT PRIMARY KEY,\n                    recommendation_version_id TEXT NOT NULL,\n                    recommendation_candidate_id TEXT,\n                    guideline_id TEXT,\n                    record_id TEXT,\n                    quality_status TEXT NOT NULL,\n                    review_status TEXT NOT NULL DEFAULT 'pending',\n                    priority TEXT NOT NULL DEFAULT 'normal',\n                    blocking_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,\n                    warning_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,\n                    publish_gate JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    version_payload JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    reviewer TEXT,\n                    review_decision TEXT,\n                    review_reason TEXT,\n                    reviewed_at TEXT,\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS evidence_items (\n                    evidence_id TEXT PRIMARY KEY,\n                    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON UPDATE CASCADE,\n                    pico_id TEXT NOT NULL REFERENCES pico_questions(pico_id) ON UPDATE CASCADE,\n                    recommendation_version_id TEXT REFERENCES recommendation_versions(recommendation_version_id) ON UPDATE CASCADE,\n                    recommendation_candidate_id TEXT REFERENCES recommendation_candidates(candidate_id) ON UPDATE CASCADE,\n                    study_design TEXT,\n                    sample_size INTEGER,\n                    population_extracted TEXT,\n                    intervention_extracted TEXT,\n                    comparator_extracted TEXT,\n                    outcomes_extracted JSONB NOT NULL DEFAULT '[]'::jsonb,\n                    effect_size JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    confidence_interval TEXT,\n                    effect_direction TEXT NOT NULL DEFAULT 'uncertain',\n                    extraction_method TEXT NOT NULL DEFAULT 'manual',\n                    extraction_confidence DOUBLE PRECISION,\n                    screening_status TEXT NOT NULL DEFAULT 'uncertain',\n                    exclusion_reason TEXT,\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\n                )\n                ",
    "                CREATE TABLE IF NOT EXISTS update_logs (\n                    update_log_id TEXT PRIMARY KEY,\n                    guideline_id TEXT NOT NULL REFERENCES guidelines(guideline_id) ON UPDATE CASCADE,\n                    new_recommendation_version_id TEXT NOT NULL REFERENCES recommendation_versions(recommendation_version_id) ON UPDATE CASCADE,\n                    recommendation_version_id TEXT REFERENCES recommendation_versions(recommendation_version_id) ON UPDATE CASCADE,\n                    old_recommendation_version_id TEXT REFERENCES recommendation_versions(recommendation_version_id) ON UPDATE CASCADE,\n                    update_type TEXT NOT NULL DEFAULT 'evidence_updated',\n                    change_summary TEXT NOT NULL DEFAULT '',\n                    change_reason TEXT,\n                    triggering_evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,\n                    updated_by TEXT,\n                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n                    published_at TEXT\n                )\n                ",
]


TABLE_MIGRATIONS = [
    "ALTER TABLE model_traces ADD COLUMN IF NOT EXISTS target_table TEXT",
    "ALTER TABLE model_traces ADD COLUMN IF NOT EXISTS target_entity_id TEXT",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS source_record_id TEXT REFERENCES cleaned_records(record_id) ON UPDATE CASCADE",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS source_text TEXT",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS source_span TEXT",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS source_section TEXT",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS source_block_id TEXT",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS source_order INTEGER",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS extraction_confidence DOUBLE PRECISION",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS extraction_method TEXT NOT NULL DEFAULT 'manual'",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS review_note TEXT",
    "ALTER TABLE pico_questions ALTER COLUMN guideline_id DROP NOT NULL",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE pico_questions ADD COLUMN IF NOT EXISTS raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE evidence_items ALTER COLUMN paper_id DROP NOT NULL",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS source_record_id TEXT REFERENCES cleaned_records(record_id) ON UPDATE CASCADE",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS model_trace_id TEXT REFERENCES model_traces(model_trace_id) ON UPDATE CASCADE",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS source_text TEXT",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS source_span TEXT",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS source_section TEXT",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS source_url TEXT",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS source_block_id TEXT",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS source_order INTEGER",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS review_note TEXT",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE evidence_items ADD COLUMN IF NOT EXISTS raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE grade_candidates ADD COLUMN IF NOT EXISTS pico_id TEXT REFERENCES pico_questions(pico_id) ON UPDATE CASCADE",
    "ALTER TABLE grade_candidates ADD COLUMN IF NOT EXISTS outcome_name TEXT",
    "ALTER TABLE grade_candidates ADD COLUMN IF NOT EXISTS source_span TEXT",
    "ALTER TABLE recommendation_candidates ADD COLUMN IF NOT EXISTS normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE recommendation_candidates ADD COLUMN IF NOT EXISTS raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE grade_candidates ADD COLUMN IF NOT EXISTS normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE grade_candidates ADD COLUMN IF NOT EXISTS raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS recommendation_id TEXT",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS previous_version_id TEXT REFERENCES recommendation_versions(recommendation_version_id) ON UPDATE CASCADE",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_recommendation_version_identity ON recommendation_versions(recommendation_id, version_number)",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS record_id TEXT REFERENCES cleaned_records(record_id) ON UPDATE CASCADE",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS grade_candidate_id TEXT REFERENCES grade_candidates(grade_candidate_id) ON UPDATE CASCADE",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS profile_id TEXT",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS issuer TEXT",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS grading_system TEXT",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS quality_status TEXT NOT NULL DEFAULT 'needs_review'",
    "ALTER TABLE recommendation_versions ALTER COLUMN quality_status SET DEFAULT 'needs_review'",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS source_span TEXT",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS source_span_ref TEXT",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS start_char INTEGER",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS end_char INTEGER",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE recommendation_versions ADD COLUMN IF NOT EXISTS raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb",
    "CREATE TABLE IF NOT EXISTS recommendations (\n                    recommendation_id TEXT PRIMARY KEY,\n                    guideline_id TEXT NOT NULL REFERENCES guidelines(guideline_id) ON UPDATE CASCADE,\n                    current_version_id TEXT REFERENCES recommendation_versions(recommendation_version_id) ON UPDATE CASCADE,\n                    status TEXT NOT NULL DEFAULT 'active',\n                    recommendation_code TEXT,\n                    title TEXT,\n                    population TEXT,\n                    intervention TEXT,\n                    comparator TEXT,\n                    direction TEXT NOT NULL DEFAULT 'unclear',\n                    strength TEXT NOT NULL DEFAULT 'unclear',\n                    certainty TEXT NOT NULL DEFAULT 'unclear',\n                    first_created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n                    last_published_at TEXT,\n                    last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,\n                    withdrawn_at TEXT,\n                    withdrawn_reason TEXT,\n                    superseded_by_recommendation_id TEXT REFERENCES recommendations(recommendation_id) ON UPDATE CASCADE,\n                    normalized_payload JSONB NOT NULL DEFAULT '{}'::jsonb\n                )",
    "CREATE TABLE IF NOT EXISTS recommendation_version_review_queue (\n                    review_item_id TEXT PRIMARY KEY,\n                    recommendation_version_id TEXT NOT NULL,\n                    recommendation_candidate_id TEXT,\n                    guideline_id TEXT,\n                    record_id TEXT,\n                    quality_status TEXT NOT NULL,\n                    review_status TEXT NOT NULL DEFAULT 'pending',\n                    priority TEXT NOT NULL DEFAULT 'normal',\n                    blocking_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,\n                    warning_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,\n                    publish_gate JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    version_payload JSONB NOT NULL DEFAULT '{}'::jsonb,\n                    reviewer TEXT,\n                    review_decision TEXT,\n                    review_reason TEXT,\n                    reviewed_at TEXT,\n                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP\n                )",
    "ALTER TABLE recommendation_version_review_queue ADD COLUMN IF NOT EXISTS published_at TEXT",
    "ALTER TABLE recommendation_version_review_queue ADD COLUMN IF NOT EXISTS published_version_id TEXT",
    "ALTER TABLE recommendation_version_review_queue ADD COLUMN IF NOT EXISTS publication_summary TEXT",
    "ALTER TABLE recommendation_version_review_queue ADD COLUMN IF NOT EXISTS publication_update_log_ids JSONB NOT NULL DEFAULT '[]'::jsonb",
]


CONSTRAINT_MIGRATIONS = [
    _add_check_constraint(
        "guidelines",
        "ck_guidelines_type",
        f"guideline_type IN ({_sql_literal_list(('living', 'standard', 'rapid', 'consensus'))})",
    ),
    _add_check_constraint(
        "guidelines",
        "ck_guidelines_status",
        f"status IN ({_sql_literal_list(('draft', 'active', 'archived', 'retired'))})",
    ),
    _add_check_constraint(
        "papers",
        "ck_papers_screening_status",
        f"screening_status IN ({_sql_literal_list(('unscreened', 'included', 'excluded', 'uncertain', 'duplicate', 'pending'))})",
    ),
    _add_check_constraint("model_traces", "ck_model_traces_confidence_range", "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)"),
    _add_check_constraint(
        "recommendation_candidates",
        "ck_recommendation_candidates_direction",
        f"direction IN ({_sql_literal_list(('for', 'against', 'neutral', 'no_recommendation', 'unclear'))})",
    ),
    _add_check_constraint(
        "recommendation_candidates",
        "ck_recommendation_candidates_strength",
        f"strength IN ({_sql_literal_list(('strong', 'conditional', 'weak', 'good_practice', 'none', 'unclear'))})",
    ),
    _add_check_constraint(
        "recommendation_candidates",
        "ck_recommendation_candidates_status",
        f"status IN ({_sql_literal_list(('pending', 'accepted', 'rejected', 'needs_review', 'merged'))})",
    ),
    _add_check_constraint(
        "recommendation_candidates",
        "ck_recommendation_candidates_confidence_range",
        "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
    ),
    _add_check_constraint(
        "grade_candidates",
        "ck_grade_candidates_certainty",
        f"certainty IN ({_sql_literal_list(('high', 'moderate', 'low', 'very_low', 'none', 'unclear', 'not_reported'))})",
    ),
    _add_check_constraint(
        "grade_candidates",
        "ck_grade_candidates_status",
        f"status IN ({_sql_literal_list(('pending', 'accepted', 'rejected', 'needs_review', 'merged'))})",
    ),
    _add_check_constraint("grade_candidates", "ck_grade_candidates_confidence_range", "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)"),
    _add_check_constraint(
        "pico_questions",
        "ck_pico_questions_status",
        f"status IN ({_sql_literal_list(('active', 'retired', 'merged', 'under_review'))})",
    ),
    _add_check_constraint(
        "pico_questions",
        "ck_pico_questions_extraction_confidence_range",
        "extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)",
    ),
    _add_check_constraint(
        "recommendation_versions",
        "ck_recommendation_versions_quality_status",
        f"quality_status IN ({_sql_literal_list(('publishable', 'needs_review', 'blocked'))})",
    ),
    _add_check_constraint(
        "recommendation_versions",
        "ck_recommendation_versions_direction",
        f"direction IN ({_sql_literal_list(('for', 'against', 'neutral', 'no_recommendation', 'unclear'))})",
    ),
    _add_check_constraint(
        "recommendation_versions",
        "ck_recommendation_versions_change_type",
        f"change_type IN ({_sql_literal_list(('new', 'modified', 'unchanged', 'withdrawn', 'reaffirmed'))})",
    ),
    _add_check_constraint(
        "recommendations",
        "ck_recommendations_status",
        f"status IN ({_sql_literal_list(('draft', 'active', 'withdrawn', 'superseded'))})",
    ),
    _add_check_constraint(
        "recommendation_version_review_queue",
        "ck_recommendation_version_review_queue_status",
        f"review_status IN ({_sql_literal_list(('pending', 'open', 'reviewed', 'closed', 'approved', 'applied', 'rejected'))})",
    ),
    _add_check_constraint(
        "evidence_items",
        "ck_evidence_items_effect_direction",
        f"effect_direction IN ({_sql_literal_list(('benefit', 'harm', 'no_effect', 'uncertain', 'mixed'))})",
    ),
    _add_check_constraint(
        "evidence_items",
        "ck_evidence_items_extraction_confidence_range",
        "extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)",
    ),
    _add_check_constraint(
        "update_logs",
        "ck_update_logs_update_type",
        f"update_type IN ({_sql_literal_list(('new_recommendation', 'text_modified', 'strength_changed', 'direction_changed', 'certainty_changed', 'withdrawn', 'reaffirmed', 'evidence_updated'))})",
    ),
    f"INSERT INTO storage_schema_migrations (migration_id) VALUES ('{STORAGE_SCHEMA_VERSION}') ON CONFLICT (migration_id) DO NOTHING",
]
