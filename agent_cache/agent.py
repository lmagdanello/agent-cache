from __future__ import annotations

from dataclasses import dataclass
import re
from uuid import uuid4

from .llm import LLMClient, LLMResponse
from .models import (
    AgentCacheConfig,
    CacheCandidate,
    CacheDecision,
    CacheDecisionKind,
    CacheHitKind,
    CacheStatus,
    GeneratedProgram,
    PromptRecord,
    ResponseRecord,
    normalize_text,
    structural_signature_from_prompt,
    utc_now,
)
from .programs import FieldExtractor, GeneratedProgramRunner, SafeProgramDSL
from .scoring import CacheScorer, ScoreWeights
from .telemetry import TelemetryCounters
from .validators import CapsulePolicyValidator, OutputSchemaValidator, StructuralValidator
from .store import SQLiteCacheStore


@dataclass(slots=True)
class AskResult:
    decision: CacheDecision
    llm_response: LLMResponse | None = None


def classify_similarity(score: float, config: AgentCacheConfig) -> CacheDecisionKind:
    if score < config.similarity_miss_threshold:
        return CacheDecisionKind.MISS
    if score < config.similarity_possible_hit_threshold:
        return CacheDecisionKind.CANDIDATE
    return CacheDecisionKind.POSSIBLE_HIT


class AgentCacheAgent:
    def __init__(
        self,
        *,
        store: SQLiteCacheStore,
        llm_client: LLMClient,
        config: AgentCacheConfig | None = None,
        scorer: CacheScorer | None = None,
        telemetry: TelemetryCounters | None = None,
    ):
        self.store = store
        self.llm_client = llm_client
        self.config = config or AgentCacheConfig()
        self.scorer = scorer or CacheScorer(ScoreWeights.from_mapping(self.config.score_weights))
        self.telemetry = telemetry or TelemetryCounters.from_dict(self.store.get_meta_json("telemetry_counters"))
        self.structural_validator = StructuralValidator()
        self.policy_validator = CapsulePolicyValidator()
        self.output_schema_validator = OutputSchemaValidator()
        self.program_runner = GeneratedProgramRunner()
        self.field_extractor = FieldExtractor()
        self.dsl = SafeProgramDSL()

    def ask(self, prompt: str) -> AskResult:
        normalized = normalize_text(prompt)
        structure_signature = structural_signature_from_prompt(normalized)
        prompt_record = PromptRecord(
            prompt_id=str(uuid4()),
            raw_prompt=prompt,
            normalized_prompt=normalized,
            structure_signature=structure_signature,
        )
        self.store.save_prompt(prompt_record)

        docs_hit = self._best_docs_index_for_prompt(prompt_record)
        if docs_hit is not None:
            docs_index, docs_item, docs_capsule = docs_hit
            response_text = self._docs_answer_from_item(docs_item)
            response = ResponseRecord(
                response_id=str(uuid4()),
                prompt_id=prompt_record.prompt_id,
                raw_response=response_text or "",
                response_schema={"type": "object"},
                response_skeleton={
                    "answer": response_text,
                    "source": self._docs_source_from_item(docs_item),
                    "index_id": docs_index["index_id"],
                },
                token_count=len((response_text or "").split()),
                metadata={
                    "kind": "docs_index",
                    "docs_index_id": docs_index["index_id"],
                    "source": self._docs_source_from_item(docs_item),
                    "matched_title": docs_item.get("title") if isinstance(docs_item, dict) else None,
                },
            )
            self.store.save_response(response)
            self.telemetry.record("hits")
            self.telemetry.record("candidate_accepts")
            self.telemetry.record("capsule_hits")
            self.telemetry.record("estimated_tokens_saved", value=response.token_count or 0)
            self._persist_telemetry()
            decision = CacheDecision(
                status=CacheStatus.HIT_EXACT,
                score=1.0,
                prompt=prompt_record,
                capsule=docs_capsule,
                response=response,
                reason=f"matched persisted docs index {docs_index['index_id']}",
            )
            return AskResult(decision=decision, llm_response=LLMResponse(text=response.raw_response, model="docs-index"))

        capsule = self._best_capsule_for_prompt(prompt_record)
        if capsule is None:
            return self._miss(prompt_record, score=0.0, reason="no capsule matched")

        scored = self._score_capsule(prompt_record, capsule)
        band = classify_similarity(scored.score, self.config)
        if band == CacheDecisionKind.MISS:
            return self._miss(prompt_record, score=scored.score, reason="below miss threshold")

        if self._is_expired(capsule):
            self.telemetry.record("expired")
            self._persist_telemetry()
            llm_response = self.llm_client.complete(prompt)
            response = self._store_response(prompt_record, llm_response)
            decision = CacheDecision(
                status=CacheStatus.EXPIRED,
                score=scored.score,
                prompt=prompt_record,
                capsule=capsule,
                response=response,
                reason="capsule ttl exceeded",
            )
            return AskResult(decision=decision, llm_response=llm_response)

        structural_result = self.structural_validator.validate(prompt_record, capsule)
        policy_result = self.policy_validator.validate(prompt_record, capsule)
        validation_results = (structural_result, policy_result)
        for result in validation_results:
            self.store.log_validation(capsule.capsule_id, prompt_record.prompt_id, result)
        if not all(result.passed for result in validation_results):
            self.telemetry.record("validation_failed")
            self.telemetry.record("capsule_rejections")
            if band == CacheDecisionKind.CANDIDATE:
                self.telemetry.record("candidate_rejects")
            self._persist_telemetry()
            llm_response = self.llm_client.complete(prompt)
            response = self._store_response(prompt_record, llm_response)
            decision = CacheDecision(
                status=CacheStatus.VALIDATION_FAILED if band == CacheDecisionKind.POSSIBLE_HIT else CacheStatus.CANDIDATE_REJECTED,
                score=scored.score,
                prompt=prompt_record,
                capsule=capsule,
                response=response,
                validation_results=validation_results,
                reason="; ".join(reason for result in validation_results for reason in result.reasons) or "validation failed",
            )
            return AskResult(decision=decision, llm_response=llm_response)

        if band == CacheDecisionKind.CANDIDATE:
            self.telemetry.record("candidate_rejects")
            self.telemetry.record("capsule_rejections")
            self._persist_telemetry()
            candidate = CacheCandidate(prompt=prompt_record, capsule=capsule, score=scored.score, reason="candidate")
            llm_response = self.llm_client.complete(prompt)
            response = self._store_response(prompt_record, llm_response)
            decision = CacheDecision(
                status=CacheStatus.CANDIDATE_REJECTED,
                score=scored.score,
                prompt=prompt_record,
                candidate=candidate,
                capsule=capsule,
                response=response,
                validation_results=validation_results,
                reason="candidate path validated but no safe generated program available",
            )
            return AskResult(decision=decision, llm_response=llm_response)

        generated = self._maybe_generate(prompt_record, capsule)
        if generated is not None:
            schema_result = self.output_schema_validator.validate(generated.response_skeleton, capsule)
            self.store.log_validation(capsule.capsule_id, prompt_record.prompt_id, schema_result)
            if not schema_result.passed:
                self.telemetry.record("validation_failed")
                self.telemetry.record("capsule_rejections")
                self._persist_telemetry()
                llm_response = self.llm_client.complete(prompt)
                response = self._store_response(prompt_record, llm_response)
                decision = CacheDecision(
                    status=CacheStatus.VALIDATION_FAILED,
                    score=scored.score,
                    prompt=prompt_record,
                    capsule=capsule,
                    response=response,
                    validation_results=validation_results + (schema_result,),
                    reason="generated output failed schema validation",
                )
                return AskResult(decision=decision, llm_response=llm_response)
            response = self._store_generated_response(prompt_record, generated)
            self.telemetry.record("hits")
            self.telemetry.record("candidate_accepts")
            self.telemetry.record("capsule_hits")
            self.telemetry.record("estimated_tokens_saved", value=response.token_count or 0)
            self._persist_telemetry()
            decision = CacheDecision(
                status=CacheStatus.HIT_GENERATED,
                score=scored.score,
                prompt=prompt_record,
                capsule=capsule,
                response=response,
                validation_results=validation_results + (schema_result,),
                reason="safe generated program executed",
            )
            return AskResult(decision=decision, llm_response=LLMResponse(text=response.raw_response, model="generated"))

        llm_response = self.llm_client.complete(prompt)
        response = self._store_response(prompt_record, llm_response)
        decision = CacheDecision(
            status=CacheStatus.MISS,
            score=scored.score,
            prompt=prompt_record,
            capsule=capsule,
            response=response,
            reason="phase1 safe fallback to llm",
        )
        return AskResult(decision=decision, llm_response=llm_response)

    def _best_capsule_for_prompt(self, prompt: PromptRecord):
        capsules = [capsule for capsule in self.store.iter_capsules() if capsule.metadata.get("type") != "docs_index"]
        if not capsules:
            return None
        best = None
        best_score = -1.0
        for capsule in capsules:
            scored = self._score_capsule(prompt, capsule)
            if scored.score > best_score:
                best = capsule
                best_score = scored.score
        return best

    def _best_docs_index_for_prompt(self, prompt: PromptRecord):
        best_index = None
        best_item = None
        best_capsule = None
        best_score = 0.0
        for docs_index in self.store.iter_docs_indexes():
            if not docs_index.get("enabled", True):
                continue
            item, score = self._match_docs_rows(prompt.raw_prompt, docs_index.get("rows", []))
            if item is None or score <= best_score:
                continue
            best_index = docs_index
            best_item = item
            best_score = score
            best_capsule = self.store.get_capsule(f"docs::{docs_index['index_id']}")
        if best_index is None or best_item is None:
            return None
        return best_index, best_item, best_capsule

    def _match_docs_rows(self, prompt: str, rows: list[dict]) -> tuple[dict | None, float]:
        normalized = " ".join(prompt.lower().split())
        match = re.search(r"find\s+(.*?)\s+under\s+(\d+)", normalized)
        if match:
            item_query = match.group(1).strip()
            budget = int(match.group(2))
            query_tokens = [token for token in item_query.split() if token]
            best_row = None
            best_score = 0.0
            for row in rows:
                name = str(row.get("name", "")).lower()
                price = row.get("price")
                if not isinstance(price, (int, float)) or price > budget:
                    continue
                if all(token in name for token in query_tokens):
                    score = 1.0 - min(float(price) / max(float(budget), 1.0), 1.0) * 0.1
                    if score > best_score:
                        best_row = row
                        best_score = score
            return best_row, best_score

        query_tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) > 2]
        best_row = None
        best_score = 0.0
        for row in rows:
            haystack = " ".join(
                str(row.get(field, ""))
                for field in ("name", "title", "summary", "description", "url", "content")
            ).lower()
            haystack += " " + " ".join(str(item).lower() for item in row.get("keywords", []))
            score = sum(1 for token in query_tokens if token in haystack)
            normalized_score = score / max(len(query_tokens), 1)
            if normalized_score > best_score:
                best_row = row
                best_score = normalized_score
        return best_row, best_score

    @staticmethod
    def _docs_answer_from_item(item: dict | None) -> str:
        if not item:
            return ""
        if "price" in item:
            return f"{item['name']} costs {item['price']} and is {item.get('availability', 'unknown')}"
        return item.get("summary") or item.get("description") or item.get("title") or ""

    @staticmethod
    def _docs_source_from_item(item: dict | None) -> str | None:
        if not item:
            return None
        return item.get("url") or item.get("source")

    def _score_capsule(self, prompt: PromptRecord, capsule):
        schema_overlap = 1.0 if capsule.response_schema else 0.0
        try:
            return self.scorer.score(prompt, capsule, schema_overlap=schema_overlap)
        except TypeError:
            return self.scorer.score(prompt, capsule)

    def _is_expired(self, capsule) -> bool:
        ttl = capsule.ttl_seconds or self.config.default_ttl_seconds
        if ttl <= 0:
            return False
        age = (utc_now() - capsule.created_at).total_seconds()
        return age > ttl or (capsule.max_age_seconds > 0 and age > capsule.max_age_seconds)

    def _maybe_generate(self, prompt: PromptRecord, capsule) -> ResponseRecord | None:
        if capsule.generated_program_id is None and "generated_program" not in capsule.metadata and "program_template" not in capsule.metadata:
            return None
        program = self._program_from_capsule(capsule)
        if program is None:
            return None
        extracted = self.dsl.extract_fields(prompt.raw_prompt, capsule.metadata.get("prompt_template", prompt.raw_prompt))
        if extracted.reasons:
            return None
        execution = self.program_runner.execute(program, extracted.fields)
        if not execution.passed or execution.output is None:
            return None
        return ResponseRecord(
            response_id=str(uuid4()),
            prompt_id=prompt.prompt_id,
            raw_response=execution.output["text"],
            response_schema=capsule.response_schema,
            response_skeleton=execution.output,
            token_count=len(str(execution.output["text"]).split()),
        )

    def _program_from_capsule(self, capsule) -> GeneratedProgram | None:
        program_data = capsule.metadata.get("generated_program")
        if isinstance(program_data, GeneratedProgram):
            return program_data
        if isinstance(program_data, dict):
            try:
                return GeneratedProgram(
                    program_id=program_data["program_id"],
                    capsule_id=program_data["capsule_id"],
                    kind=program_data["kind"],
                    template=program_data["template"],
                    required_fields=tuple(program_data.get("required_fields", ())),
                    allowed_fields=tuple(program_data.get("allowed_fields", ())),
                    transformations=dict(program_data.get("transformations", {})),
                    metadata=dict(program_data.get("metadata", {})),
                )
            except KeyError:
                return None
        return None

    def _store_response(self, prompt: PromptRecord, llm_response: LLMResponse) -> ResponseRecord:
        response = ResponseRecord(
            response_id=str(uuid4()),
            prompt_id=prompt.prompt_id,
            raw_response=llm_response.text,
            response_skeleton={"text": llm_response.text[:128]},
            token_count=llm_response.completion_tokens or len(llm_response.text.split()),
        )
        self.store.save_response(response)
        return response

    def _store_generated_response(self, prompt: PromptRecord, response: ResponseRecord) -> ResponseRecord:
        self.store.save_response(response)
        return response

    def _persist_telemetry(self) -> None:
        self.store.set_meta_json("telemetry_counters", self.telemetry.to_dict())

    def _miss(self, prompt: PromptRecord, *, score: float, reason: str) -> AskResult:
        llm_response = self.llm_client.complete(prompt.raw_prompt)
        response = self._store_response(prompt, llm_response)
        self.telemetry.record("misses")
        self._persist_telemetry()
        decision = CacheDecision(
            status=CacheStatus.MISS,
            score=score,
            prompt=prompt,
            response=response,
            reason=reason,
        )
        return AskResult(decision=decision, llm_response=llm_response)
