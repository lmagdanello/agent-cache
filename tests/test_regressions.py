from agent_cache.agent import AgentCacheAgent
from agent_cache.llm import FakeLLMClient
from agent_cache.models import AgentCacheConfig, CacheCapsule
from agent_cache.store import SQLiteCacheStore


class FixedScorer:
    def __init__(self, score: float):
        self._score = score

    def score(self, prompt, capsule):
        return type("Scored", (), {"score": self._score})()


def test_semantically_similar_but_unsafe_prompt_is_rejected(tmp_path):
    store = SQLiteCacheStore(tmp_path / "cache.sqlite3")
    capsule = CacheCapsule(
        capsule_id="c1",
        name="shopping",
        prompt_structure_signature="find aaa batteries under <num>",
        ttl_seconds=60,
        forbidden_fields=("availability",),
    )
    store.save_capsule(capsule)
    agent = AgentCacheAgent(store=store, llm_client=FakeLLMClient(), config=AgentCacheConfig(), scorer=FixedScorer(0.95))
    result = agent.ask("Find AAA batteries under 30 availability now")
    assert result.decision.status.value == "VALIDATION_FAILED"

