from agent_cache.models import CacheCapsule, PromptRecord, ResponseRecord, structural_signature_from_prompt
from agent_cache.store import SQLiteCacheStore


def test_store_roundtrip(tmp_path):
    store = SQLiteCacheStore(tmp_path / "cache.sqlite3")
    prompt = PromptRecord(
        prompt_id="p1",
        raw_prompt="Find AAA batteries under 30",
        normalized_prompt="find aaa batteries under 30",
        structure_signature=structural_signature_from_prompt("find aaa batteries under 30"),
    )
    response = ResponseRecord(
        response_id="r1",
        prompt_id="p1",
        raw_response="result",
        response_skeleton={"text": "result"},
    )
    capsule = CacheCapsule(
        capsule_id="c1",
        name="shopping",
        prompt_structure_signature="find aaa batteries under <num>",
    )
    store.save_prompt(prompt)
    store.save_response(response)
    store.save_capsule(capsule)
    assert store.count_prompts() == 1
    assert store.count_responses() == 1
    assert store.count_capsules() == 1
    assert store.get_capsule("c1").name == "shopping"

