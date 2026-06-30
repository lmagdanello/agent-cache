from agent_cache.agent import AgentCacheAgent
from agent_cache.agent import classify_similarity
from agent_cache.llm import FakeLLMClient
from agent_cache.models import AgentCacheConfig, CacheCapsule, CacheDecisionKind, PromptRecord, structural_signature_from_prompt
from agent_cache.store import SQLiteCacheStore


def make_prompt(text: str) -> PromptRecord:
    normalized = " ".join(text.lower().split())
    return PromptRecord(
        prompt_id="p",
        raw_prompt=text,
        normalized_prompt=normalized,
        structure_signature=structural_signature_from_prompt(normalized),
    )


def test_below_threshold_is_miss(tmp_path):
    store = SQLiteCacheStore(tmp_path / "cache.sqlite3")
    capsule = CacheCapsule(
        capsule_id="c1",
        name="match",
        prompt_structure_signature="summarize incident <num> for region <long>",
    )
    store.save_capsule(capsule)
    agent = AgentCacheAgent(store=store, llm_client=FakeLLMClient(), config=AgentCacheConfig())
    result = agent.ask("hello world")
    assert result.decision.status.value == "MISS"


def test_threshold_classification():
    config = AgentCacheConfig()
    assert classify_similarity(0.74, config) == CacheDecisionKind.MISS
    assert classify_similarity(0.75, config) == CacheDecisionKind.CANDIDATE
    assert classify_similarity(0.90, config) == CacheDecisionKind.CANDIDATE
    assert classify_similarity(0.91, config) == CacheDecisionKind.POSSIBLE_HIT


def test_candidate_band_is_not_returned_directly(tmp_path):
    class FixedScorer:
        def score(self, prompt, capsule):
            return type("Scored", (), {"score": 0.8})()

    store = SQLiteCacheStore(tmp_path / "cache.sqlite3")
    capsule = CacheCapsule(
        capsule_id="c1",
        name="match",
        prompt_structure_signature="summarize incident inc-123 for region us-east",
    )
    store.save_capsule(capsule)
    agent = AgentCacheAgent(store=store, llm_client=FakeLLMClient(), config=AgentCacheConfig(), scorer=FixedScorer())
    result = agent.ask("Summarize incident INC-123 for region us-east")
    assert result.decision.status.value == "CANDIDATE_REJECTED"
    assert result.llm_response is not None
    assert result.llm_response.text.startswith("FAKE_RESPONSE:")
