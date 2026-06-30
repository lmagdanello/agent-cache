from agent_cache.agent import AgentCacheAgent
from agent_cache.llm import FakeLLMClient
from agent_cache.models import AgentCacheConfig, CacheCapsule, structural_signature_from_prompt
from agent_cache.store import SQLiteCacheStore


class FixedScorer:
    def __init__(self, score: float):
        self._score = score

    def score(self, prompt, capsule):
        return type("Scored", (), {"score": self._score})()


def test_agent_generates_hit_when_program_is_safe(tmp_path):
    store = SQLiteCacheStore(tmp_path / "cache.sqlite3")
    capsule = CacheCapsule(
        capsule_id="c1",
        name="support",
        prompt_structure_signature=structural_signature_from_prompt("Summarize incident INC-123 for region us-east"),
        response_schema={"type": "object", "required": ["text"]},
        ttl_seconds=60,
        metadata={
            "prompt_template": "Summarize incident {incident_id} for region {region}",
            "generated_program": {
                "program_id": "gp1",
                "capsule_id": "c1",
                "kind": "template",
                "template": "Incident {incident_id} in {region}",
                "required_fields": ["incident_id", "region"],
                "allowed_fields": ["incident_id", "region"],
                "transformations": {"region": "upper"},
            },
        },
    )
    store.save_capsule(capsule)
    agent = AgentCacheAgent(store=store, llm_client=FakeLLMClient(), config=AgentCacheConfig(), scorer=FixedScorer(0.95))
    result = agent.ask("Summarize incident INC-123 for region us-east")
    assert result.decision.status.value == "HIT_GENERATED"
    assert result.decision.response.raw_response == "Incident INC-123 in US-EAST"


def test_agent_expires_old_capsules(tmp_path):
    store = SQLiteCacheStore(tmp_path / "cache.sqlite3")
    capsule = CacheCapsule(
        capsule_id="c1",
        name="support",
        prompt_structure_signature="summarize incident <ALNUM> for region <LONG>",
        ttl_seconds=1,
    )
    capsule.created_at = capsule.created_at.replace(year=2020)
    store.save_capsule(capsule)
    agent = AgentCacheAgent(store=store, llm_client=FakeLLMClient(), config=AgentCacheConfig(), scorer=FixedScorer(0.95))
    result = agent.ask("Summarize incident INC-123 for region us-east")
    assert result.decision.status.value == "EXPIRED"
