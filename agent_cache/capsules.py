from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter
from typing import Iterable

from .models import CacheCapsule
from .models import PromptRecord
from .programs import GeneratedProgram


@dataclass(slots=True)
class CapsuleHealth:
    hit_rate: float = 0.0
    rejection_rate: float = 0.0
    enabled: bool = True


def capsule_is_dynamic(capsule: CacheCapsule) -> bool:
    return bool(capsule.allowed_dynamic_fields or capsule.forbidden_fields)


def derive_regex_from_prompt(prompt: str) -> tuple[str, ...]:
    normalized = prompt.lower().strip()
    parts = []
    for token in normalized.split():
        if any(ch.isdigit() for ch in token):
            parts.append(r"\S+")
        else:
            parts.append(re.escape(token))
    return ("^" + r"\s+".join(parts) + "$",)


def build_capsule_from_examples(
    capsule_id: str,
    name: str,
    prompts: Iterable[PromptRecord],
    *,
    generated_program: GeneratedProgram | None = None,
    ttl_seconds: int = 3600,
    max_examples: int = 5,
) -> CacheCapsule:
    prompt_list = list(prompts)
    if not prompt_list:
        raise ValueError("at least one prompt is required")

    signatures = Counter(prompt.structure_signature for prompt in prompt_list)
    signature = signatures.most_common(1)[0][0]
    sample_prompt = prompt_list[0].raw_prompt
    return CacheCapsule(
        capsule_id=capsule_id,
        name=name,
        prompt_structure_signature=signature,
        regex_rules=derive_regex_from_prompt(sample_prompt),
        parser_rules={"kind": "template"},
        response_schema={"type": "object", "required": []},
        allowed_dynamic_fields=(),
        forbidden_fields=(),
        ttl_seconds=ttl_seconds,
        max_age_seconds=ttl_seconds,
        max_examples=max_examples,
        generated_program_id=generated_program.program_id if generated_program else None,
        validation_policy={"mode": "strict"},
        safety_flags=("safe-template",),
        enabled=True,
        metadata={"example_count": len(prompt_list)},
    )
