from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from typing import Sequence

from .models import CacheCapsule, PromptRecord, normalize_text, structural_signature_from_prompt
from .programs import SafeProgramDSL


@dataclass(slots=True)
class ScoreWeights:
    embedding: float = 0.45
    structural: float = 0.30
    schema: float = 0.10
    recency: float = 0.10
    health: float = 0.05

    @classmethod
    def from_mapping(cls, mapping: dict[str, float]) -> "ScoreWeights":
        return cls(
            embedding=mapping.get("embedding", 0.45),
            structural=mapping.get("structural", 0.30),
            schema=mapping.get("schema", 0.10),
            recency=mapping.get("recency", 0.10),
            health=mapping.get("health", 0.05),
        )


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


@dataclass(slots=True)
class ScoredCapsule:
    capsule: CacheCapsule | None
    score: float
    embedding_score: float
    structural_score: float
    schema_score: float
    recency_score: float
    health_score: float


class CacheScorer:
    def __init__(self, weights: ScoreWeights | None = None):
        self.weights = weights or ScoreWeights()
        self._dsl = SafeProgramDSL()

    def score(
        self,
        prompt: PromptRecord,
        capsule: CacheCapsule | None,
        *,
        capsule_health: float = 1.0,
        schema_overlap: float = 0.0,
        recency_seconds: float | None = None,
    ) -> ScoredCapsule:
        if capsule is None:
            return ScoredCapsule(None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        embedding_score = _cosine_similarity(prompt.embedding, capsule.metadata.get("centroid_embedding", ()))
        if embedding_score == 0.0:
            embedding_score = max(
                _token_jaccard(prompt.normalized_prompt, capsule.prompt_structure_signature),
                _token_jaccard(prompt.structure_signature, capsule.prompt_structure_signature),
            )

        structural_score = _token_jaccard(
            prompt.structure_signature,
            capsule.prompt_structure_signature,
        )
        template_match = self._template_match_score(prompt, capsule)
        embedding_score = max(embedding_score, template_match)
        structural_score = max(structural_score, template_match)
        schema_score = max(0.0, min(1.0, schema_overlap))
        recency_score = self._recency_score(recency_seconds)
        health_score = max(0.0, min(1.0, capsule_health))

        total = (
            self.weights.embedding * embedding_score
            + self.weights.structural * structural_score
            + self.weights.schema * schema_score
            + self.weights.recency * recency_score
            + self.weights.health * health_score
        )
        return ScoredCapsule(
            capsule=capsule,
            score=max(0.0, min(1.0, total)),
            embedding_score=embedding_score,
            structural_score=structural_score,
            schema_score=schema_score,
            recency_score=recency_score,
            health_score=health_score,
        )

    def score_prompt_pair(self, left: PromptRecord, right: PromptRecord) -> float:
        left_sig = left.structure_signature or structural_signature_from_prompt(left.normalized_prompt)
        right_sig = right.structure_signature or structural_signature_from_prompt(right.normalized_prompt)
        structural = _token_jaccard(left_sig, right_sig)
        embedding = _cosine_similarity(left.embedding, right.embedding)
        if embedding == 0.0:
            embedding = _token_jaccard(left.normalized_prompt, right.normalized_prompt)
        return max(0.0, min(1.0, 0.6 * embedding + 0.4 * structural))

    @staticmethod
    def _recency_score(recency_seconds: float | None) -> float:
        if recency_seconds is None:
            return 0.5
        if recency_seconds <= 0:
            return 1.0
        decay = 86400.0
        return max(0.0, min(1.0, 1.0 / (1.0 + recency_seconds / decay)))

    def _template_match_score(self, prompt: PromptRecord, capsule: CacheCapsule) -> float:
        template = capsule.metadata.get("prompt_template")
        if not isinstance(template, str) or not template.strip():
            return 0.0
        extracted = self._dsl.extract_fields(prompt.raw_prompt, template)
        return 1.0 if not extracted.reasons else 0.0
