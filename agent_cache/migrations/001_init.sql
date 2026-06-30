CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompts (
    prompt_id TEXT PRIMARY KEY,
    raw_prompt TEXT NOT NULL,
    normalized_prompt TEXT NOT NULL,
    structure_signature TEXT NOT NULL,
    embedding TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS responses (
    response_id TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL REFERENCES prompts(prompt_id) ON DELETE CASCADE,
    raw_response TEXT NOT NULL,
    response_schema TEXT NOT NULL,
    response_skeleton TEXT NOT NULL,
    created_at TEXT NOT NULL,
    token_count INTEGER,
    metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS capsules (
    capsule_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    prompt_structure_signature TEXT NOT NULL,
    regex_rules TEXT NOT NULL,
    parser_rules TEXT NOT NULL,
    response_schema TEXT NOT NULL,
    allowed_dynamic_fields TEXT NOT NULL,
    forbidden_fields TEXT NOT NULL,
    ttl_seconds INTEGER NOT NULL,
    max_age_seconds INTEGER NOT NULL,
    max_examples INTEGER NOT NULL,
    generated_program_id TEXT,
    validation_policy TEXT NOT NULL,
    safety_flags TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS validation_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    capsule_id TEXT,
    prompt_id TEXT,
    validator TEXT NOT NULL,
    passed INTEGER NOT NULL,
    reasons TEXT NOT NULL,
    details TEXT NOT NULL,
    created_at TEXT NOT NULL
);
