from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import CacheCapsule, PromptRecord, ResponseRecord, ValidationResult, utc_now


SCHEMA_VERSION = 2


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _json_loads(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def _dt_to_str(value):
    return value.isoformat() if value is not None else None


def _str_to_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


class SQLiteCacheStore:
    def __init__(self, db_path: str | Path, *, redaction_hook=None):
        self.db_path = Path(db_path)
        self.redaction_hook = redaction_hook
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.initialize()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def initialize(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
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

                CREATE TABLE IF NOT EXISTS docs_indexes (
                    index_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    rows_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                """
            )
            current_version = self.get_meta_value("schema_version")
            if current_version is None:
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(SCHEMA_VERSION)),
                )
            elif int(current_version) < SCHEMA_VERSION:
                self.set_meta_value("schema_version", str(SCHEMA_VERSION))
            self._conn.commit()

    def save_prompt(self, prompt: PromptRecord) -> None:
        prompt = self._maybe_redact_prompt(prompt)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO prompts(
                    prompt_id, raw_prompt, normalized_prompt, structure_signature, embedding, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prompt.prompt_id,
                    prompt.raw_prompt,
                    prompt.normalized_prompt,
                    prompt.structure_signature,
                    _json_dumps(list(prompt.embedding)),
                    _dt_to_str(prompt.created_at),
                    _json_dumps(prompt.metadata),
                ),
            )
            self._conn.commit()

    def save_response(self, response: ResponseRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO responses(
                    response_id, prompt_id, raw_response, response_schema, response_skeleton, created_at, token_count, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.response_id,
                    response.prompt_id,
                    response.raw_response,
                    _json_dumps(response.response_schema),
                    _json_dumps(response.response_skeleton),
                    _dt_to_str(response.created_at),
                    response.token_count,
                    _json_dumps(response.metadata),
                ),
            )
            self._conn.commit()

    def save_capsule(self, capsule: CacheCapsule) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO capsules(
                    capsule_id, name, prompt_structure_signature, regex_rules, parser_rules, response_schema,
                    allowed_dynamic_fields, forbidden_fields, ttl_seconds, max_age_seconds, max_examples,
                    generated_program_id, validation_policy, safety_flags, enabled, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capsule.capsule_id,
                    capsule.name,
                    capsule.prompt_structure_signature,
                    _json_dumps(list(capsule.regex_rules)),
                    _json_dumps(capsule.parser_rules),
                    _json_dumps(capsule.response_schema),
                    _json_dumps(list(capsule.allowed_dynamic_fields)),
                    _json_dumps(list(capsule.forbidden_fields)),
                    capsule.ttl_seconds,
                    capsule.max_age_seconds,
                    capsule.max_examples,
                    capsule.generated_program_id,
                    _json_dumps(capsule.validation_policy),
                    _json_dumps(list(capsule.safety_flags)),
                    1 if capsule.enabled else 0,
                    _dt_to_str(capsule.created_at),
                    _dt_to_str(capsule.updated_at),
                    _json_dumps(capsule.metadata),
                ),
            )
            self._conn.commit()

    def save_docs_index(
        self,
        index_id: str,
        *,
        name: str,
        source: str,
        rows: Sequence[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        metadata = dict(metadata or {})
        now = utc_now().isoformat()
        capsule = CacheCapsule(
            capsule_id=f"docs::{index_id}",
            name=name,
            prompt_structure_signature=f"docs-index::{index_id}",
            response_schema={"type": "object"},
            ttl_seconds=0,
            max_age_seconds=0,
            max_examples=len(rows),
            generated_program_id=None,
            validation_policy={"kind": "docs_index"},
            safety_flags=("docs_index",),
            enabled=enabled,
            created_at=_str_to_dt(metadata.get("created_at")) or utc_now(),
            updated_at=_str_to_dt(metadata.get("updated_at")) or utc_now(),
            metadata={
                "type": "docs_index",
                "docs_index_id": index_id,
                "source": source,
                "row_count": len(rows),
                **metadata,
            },
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO docs_indexes(
                    index_id, name, source, rows_json, enabled, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    index_id,
                    name,
                    source,
                    _json_dumps(list(rows)),
                    1 if enabled else 0,
                    now,
                    now,
                    _json_dumps(capsule.metadata),
                ),
            )
            self._conn.commit()
        self.save_capsule(capsule)

    def iter_docs_indexes(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM docs_indexes ORDER BY created_at ASC").fetchall()
        return [self._row_to_docs_index(row) for row in rows]

    def count_docs_indexes(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM docs_indexes").fetchone()
        return int(row["n"])

    def log_validation(self, capsule_id: str | None, prompt_id: str | None, result: ValidationResult) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO validation_logs(capsule_id, prompt_id, validator, passed, reasons, details, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    capsule_id,
                    prompt_id,
                    result.validator,
                    1 if result.passed else 0,
                    _json_dumps(list(result.reasons)),
                    _json_dumps(result.details),
                    utc_now().isoformat(),
                ),
            )
            self._conn.commit()

    def iter_capsules(self) -> list[CacheCapsule]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM capsules ORDER BY created_at ASC").fetchall()
        return [self._row_to_capsule(row) for row in rows]

    def get_capsule(self, capsule_id: str) -> CacheCapsule | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM capsules WHERE capsule_id = ?", (capsule_id,)).fetchone()
        return self._row_to_capsule(row) if row else None

    def count_prompts(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM prompts").fetchone()
        return int(row["n"])

    def count_responses(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM responses").fetchone()
        return int(row["n"])

    def count_capsules(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM capsules").fetchone()
        return int(row["n"])

    def count_disabled_capsules(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM capsules WHERE enabled = 0").fetchone()
        return int(row["n"])

    def count_validation_logs(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM validation_logs").fetchone()
        return int(row["n"])

    def capsule_validation_summary(self, capsule_id: str) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) AS passed,
                    SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) AS failed
                FROM validation_logs
                WHERE capsule_id = ?
                """,
                (capsule_id,),
            ).fetchone()
        return {
            "total": int(row["total"] or 0),
            "passed": int(row["passed"] or 0),
            "failed": int(row["failed"] or 0),
        }

    def get_meta_value(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta_value(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (key, value),
            )
            self._conn.commit()

    def get_meta_json(self, key: str) -> Any:
        value = self.get_meta_value(key)
        return _json_loads(value, {}) if value is not None else {}

    def set_meta_json(self, key: str, value: Any) -> None:
        self.set_meta_value(key, _json_dumps(value))

    def _maybe_redact_prompt(self, prompt: PromptRecord) -> PromptRecord:
        if self.redaction_hook is None:
            return prompt
        redacted = self.redaction_hook(prompt)
        return redacted if isinstance(redacted, PromptRecord) else prompt

    @staticmethod
    def _row_to_capsule(row: sqlite3.Row) -> CacheCapsule:
        return CacheCapsule(
            capsule_id=row["capsule_id"],
            name=row["name"],
            prompt_structure_signature=row["prompt_structure_signature"],
            regex_rules=tuple(_json_loads(row["regex_rules"], [])),
            parser_rules=_json_loads(row["parser_rules"], {}),
            response_schema=_json_loads(row["response_schema"], {}),
            allowed_dynamic_fields=tuple(_json_loads(row["allowed_dynamic_fields"], [])),
            forbidden_fields=tuple(_json_loads(row["forbidden_fields"], [])),
            ttl_seconds=row["ttl_seconds"],
            max_age_seconds=row["max_age_seconds"],
            max_examples=row["max_examples"],
            generated_program_id=row["generated_program_id"],
            validation_policy=_json_loads(row["validation_policy"], {}),
            safety_flags=tuple(_json_loads(row["safety_flags"], [])),
            enabled=bool(row["enabled"]),
            created_at=_str_to_dt(row["created_at"]) or utc_now(),
            updated_at=_str_to_dt(row["updated_at"]) or utc_now(),
            metadata=_json_loads(row["metadata"], {}),
        )

    @staticmethod
    def _row_to_docs_index(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "index_id": row["index_id"],
            "name": row["name"],
            "source": row["source"],
            "rows": _json_loads(row["rows_json"], []),
            "enabled": bool(row["enabled"]),
            "created_at": _str_to_dt(row["created_at"]) or utc_now(),
            "updated_at": _str_to_dt(row["updated_at"]) or utc_now(),
            "metadata": _json_loads(row["metadata"], {}),
        }
