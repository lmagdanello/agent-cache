from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CacheStatus(str, Enum):
    HIT_GENERATED = "HIT_GENERATED"
    HIT_EXACT = "HIT_EXACT"
    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"
    MISS = "MISS"
    EXPIRED = "EXPIRED"
    VALIDATION_FAILED = "VALIDATION_FAILED"


class CacheDecisionKind(str, Enum):
    MISS = "MISS"
    CANDIDATE = "CANDIDATE"
    POSSIBLE_HIT = "POSSIBLE_HIT"


class CacheHitKind(str, Enum):
    GENERATED = "GENERATED"
    EXACT = "EXACT"


@dataclass(slots=True)
class PromptRecord:
    prompt_id: str
    raw_prompt: str
    normalized_prompt: str
    structure_signature: str
    embedding: tuple[float, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResponseRecord:
    response_id: str
    prompt_id: str
    raw_response: str
    response_schema: dict[str, Any] = field(default_factory=dict)
    response_skeleton: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedProgram:
    program_id: str
    capsule_id: str
    kind: str
    template: str
    required_fields: tuple[str, ...] = field(default_factory=tuple)
    allowed_fields: tuple[str, ...] = field(default_factory=tuple)
    transformations: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CacheCapsule:
    capsule_id: str
    name: str
    prompt_structure_signature: str
    regex_rules: tuple[str, ...] = field(default_factory=tuple)
    parser_rules: dict[str, Any] = field(default_factory=dict)
    response_schema: dict[str, Any] = field(default_factory=dict)
    allowed_dynamic_fields: tuple[str, ...] = field(default_factory=tuple)
    forbidden_fields: tuple[str, ...] = field(default_factory=tuple)
    ttl_seconds: int = 0
    max_age_seconds: int = 0
    max_examples: int = 0
    generated_program_id: str | None = None
    validation_policy: dict[str, Any] = field(default_factory=dict)
    safety_flags: tuple[str, ...] = field(default_factory=tuple)
    enabled: bool = True
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CacheCandidate:
    prompt: PromptRecord
    capsule: CacheCapsule | None
    score: float
    reason: str = ""
    matched_fields: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class ValidationResult:
    passed: bool
    validator: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CacheDecision:
    status: CacheStatus
    score: float
    prompt: PromptRecord
    candidate: CacheCandidate | None = None
    capsule: CacheCapsule | None = None
    response: ResponseRecord | None = None
    validation_results: tuple[ValidationResult, ...] = field(default_factory=tuple)
    estimated_tokens_saved: int = 0
    estimated_cost_saved: float = 0.0
    reason: str = ""
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class AgentCacheConfig:
    similarity_miss_threshold: float = 0.75
    similarity_possible_hit_threshold: float = 0.91
    default_ttl_seconds: int = 3600
    max_examples_per_capsule: int = 5
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "embedding": 0.45,
            "structural": 0.30,
            "schema": 0.10,
            "recency": 0.10,
            "health": 0.05,
        }
    )
    allow_generated_programs: bool = True
    cache_namespace: str = "default"
    redaction_fields: tuple[str, ...] = field(default_factory=tuple)
    max_validation_logs: int = 1000
    dynamic_field_names: tuple[str, ...] = ("price", "availability", "date", "timestamp", "metrics", "account")
    extra: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def structural_signature_from_prompt(prompt: str) -> str:
    tokens = []
    for raw in prompt.split():
        if raw.isdigit():
            tokens.append("<NUM>")
        elif any(ch.isdigit() for ch in raw):
            tokens.append("<ALNUM>")
        elif len(raw) > 6:
            tokens.append("<LONG>")
        else:
            tokens.append(raw.lower())
    return " ".join(tokens)


def contains_dynamic_data(prompt: str, dynamic_field_names: Sequence[str]) -> bool:
    normalized = normalize_text(prompt)
    return any(field in normalized for field in dynamic_field_names)


def now_iso() -> str:
    return utc_now().isoformat()

