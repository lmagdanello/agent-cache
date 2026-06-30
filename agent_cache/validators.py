from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Protocol

from .models import CacheCapsule, PromptRecord, ValidationResult, contains_dynamic_data, normalize_text
from .programs import SafeProgramDSL


class ExplicitLLMValidator(Protocol):
    def validate(self, prompt: PromptRecord, candidate_output: object, capsule: CacheCapsule) -> ValidationResult: ...


@dataclass(slots=True)
class StructuralValidator:
    dsl: SafeProgramDSL = field(default_factory=SafeProgramDSL)

    def validate(self, prompt: PromptRecord, capsule: CacheCapsule) -> ValidationResult:
        reasons: list[str] = []
        regex_pass = True
        if capsule.regex_rules:
            normalized = normalize_text(prompt.raw_prompt)
            regex_pass = any(re.search(pattern, normalized) for pattern in capsule.regex_rules)
            if not regex_pass:
                reasons.append("regex rule mismatch")

        template = capsule.metadata.get("prompt_template")
        template_pass = False
        if isinstance(template, str) and template.strip():
            template_pass = not self.dsl.extract_fields(prompt.raw_prompt, template).reasons

        signature_pass = prompt.structure_signature == capsule.prompt_structure_signature or template_pass
        if not signature_pass:
            reasons.append("structure signature mismatch")

        passed = regex_pass and signature_pass
        return ValidationResult(
            passed=passed,
            validator="StructuralValidator",
            reasons=tuple(reasons),
            details={
                "prompt_signature": prompt.structure_signature,
                "capsule_signature": capsule.prompt_structure_signature,
                "template_matched": template_pass,
            },
        )


@dataclass(slots=True)
class CapsulePolicyValidator:
    def validate(self, prompt: PromptRecord, capsule: CacheCapsule) -> ValidationResult:
        reasons: list[str] = []
        passed = capsule.enabled and capsule.ttl_seconds >= 0
        if contains_dynamic_data(prompt.raw_prompt, capsule.allowed_dynamic_fields or ("price", "availability", "date", "metrics", "account")):
            if not capsule.allowed_dynamic_fields:
                reasons.append("dynamic data requires explicit policy")
                passed = False
        if not capsule.enabled:
            reasons.append("capsule disabled")
        if capsule.ttl_seconds < 0:
            reasons.append("invalid ttl")
        for forbidden in capsule.forbidden_fields:
            if forbidden and forbidden in normalize_text(prompt.raw_prompt):
                reasons.append(f"forbidden field present: {forbidden}")
                passed = False
        return ValidationResult(
            passed=passed,
            validator="CapsulePolicyValidator",
            reasons=tuple(reasons),
            details={"ttl_seconds": capsule.ttl_seconds, "enabled": capsule.enabled},
        )


@dataclass(slots=True)
class OutputSchemaValidator:
    def validate(self, candidate_output: object, capsule: CacheCapsule) -> ValidationResult:
        expected = capsule.response_schema
        if not expected:
            return ValidationResult(True, "OutputSchemaValidator", ("no schema configured",), {})
        if not isinstance(candidate_output, dict):
            return ValidationResult(False, "OutputSchemaValidator", ("output must be an object",), {"expected": expected})
        required = expected.get("required", [])
        missing = [field for field in required if field not in candidate_output]
        if missing:
            return ValidationResult(False, "OutputSchemaValidator", (f"missing required fields: {', '.join(missing)}",), {"expected": expected})
        schema_type = expected.get("type")
        if schema_type == "object" and not isinstance(candidate_output, dict):
            return ValidationResult(False, "OutputSchemaValidator", ("expected object output",), {"expected": expected})
        return ValidationResult(True, "OutputSchemaValidator", (), {"expected": expected})


@dataclass(slots=True)
class ValidatorPipeline:
    structural: StructuralValidator = field(default_factory=StructuralValidator)
    policy: CapsulePolicyValidator = field(default_factory=CapsulePolicyValidator)
    schema: OutputSchemaValidator = field(default_factory=OutputSchemaValidator)

    def validate(self, prompt: PromptRecord, capsule: CacheCapsule, candidate_output: object) -> tuple[ValidationResult, ...]:
        results = [
            self.structural.validate(prompt, capsule),
            self.policy.validate(prompt, capsule),
        ]
        if all(result.passed for result in results):
            results.append(self.schema.validate(candidate_output, capsule))
        return tuple(results)
