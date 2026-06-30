from agent_cache.models import GeneratedProgram
from agent_cache.programs import FieldExtractor, GeneratedProgramRunner, SafeProgramDSL


def test_dsl_extracts_named_fields():
    dsl = SafeProgramDSL()
    extracted = dsl.extract_fields("Summarize incident INC-123 for region us-east", "Summarize incident {incident_id} for region {region}")
    assert extracted.fields == {"incident_id": "INC-123", "region": "us-east"}


def test_program_runner_applies_transformations():
    program = GeneratedProgram(
        program_id="gp1",
        capsule_id="c1",
        kind="template",
        template="Incident {incident_id} in {region}",
        required_fields=("incident_id", "region"),
        allowed_fields=("incident_id", "region"),
        transformations={"region": "upper"},
    )
    result = GeneratedProgramRunner().execute(program, {"incident_id": "INC-123", "region": "us-east"})
    assert result.passed
    assert result.output == {"text": "Incident INC-123 in US-EAST"}


def test_program_runner_rejects_missing_fields():
    program = GeneratedProgram(
        program_id="gp1",
        capsule_id="c1",
        kind="template",
        template="Incident {incident_id} in {region}",
        required_fields=("incident_id", "region"),
        allowed_fields=("incident_id", "region"),
    )
    result = GeneratedProgramRunner().execute(program, {"incident_id": "INC-123"})
    assert not result.passed

