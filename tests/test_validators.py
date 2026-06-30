from agent_cache.models import CacheCapsule, PromptRecord, structural_signature_from_prompt
from agent_cache.validators import CapsulePolicyValidator, OutputSchemaValidator, StructuralValidator, ValidatorPipeline


def make_prompt(text: str) -> PromptRecord:
    normalized = " ".join(text.lower().split())
    return PromptRecord(
        prompt_id="p",
        raw_prompt=text,
        normalized_prompt=normalized,
        structure_signature=structural_signature_from_prompt(normalized),
    )


def test_structural_validator_rejects_mismatch():
    prompt = make_prompt("Summarize incident INC-123 for region us-east")
    capsule = CacheCapsule(
        capsule_id="c",
        name="incident",
        prompt_structure_signature="find aaa batteries under <num>",
    )
    result = StructuralValidator().validate(prompt, capsule)
    assert not result.passed
    assert "structure signature mismatch" in result.reasons


def test_policy_validator_rejects_dynamic_data_without_policy():
    prompt = make_prompt("Find USB-C cable with price under 50")
    capsule = CacheCapsule(
        capsule_id="c",
        name="shopping",
        prompt_structure_signature=prompt.structure_signature,
        ttl_seconds=60,
    )
    result = CapsulePolicyValidator().validate(prompt, capsule)
    assert not result.passed
    assert "dynamic data requires explicit policy" in result.reasons


def test_output_schema_validator_requires_keys():
    capsule = CacheCapsule(
        capsule_id="c",
        name="schema",
        prompt_structure_signature="x",
        response_schema={"type": "object", "required": ["text", "source"]},
    )
    result = OutputSchemaValidator().validate({"text": "ok"}, capsule)
    assert not result.passed
    assert "missing required fields: source" in result.reasons[0]


def test_pipeline_runs_deterministic_validators_first():
    prompt = make_prompt("Hello world")
    capsule = CacheCapsule(capsule_id="c", name="x", prompt_structure_signature="different", ttl_seconds=-1)
    results = ValidatorPipeline().validate(prompt, capsule, {})
    assert len(results) == 2
    assert not results[0].passed
    assert not results[1].passed
