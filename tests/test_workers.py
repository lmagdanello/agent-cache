import threading
import time

from agent_cache.buffers import CandidateBuffer, CapsuleBuildBuffer, MissBuffer
from agent_cache.models import PromptRecord, ResponseRecord, structural_signature_from_prompt
from agent_cache.store import SQLiteCacheStore
from agent_cache.workers import BackgroundWorkerSystem


def make_prompt(text: str) -> PromptRecord:
    normalized = " ".join(text.lower().split())
    return PromptRecord(
        prompt_id=text,
        raw_prompt=text,
        normalized_prompt=normalized,
        structure_signature=structural_signature_from_prompt(normalized),
    )


def test_bounded_buffer_is_thread_safe():
    buffer = CandidateBuffer(128)

    def writer(index: int) -> None:
        buffer.append(index)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(buffer.snapshot()) == 50


def test_background_worker_builds_capsule(tmp_path):
    store = SQLiteCacheStore(tmp_path / "cache.sqlite3")
    worker = BackgroundWorkerSystem(store=store, build_threshold=2)
    worker.start()
    try:
        prompts = [make_prompt("Find AAA batteries under 30"), make_prompt("Find USB-C cable under 50")]
        for prompt in prompts:
            response = ResponseRecord(
                response_id=f"r-{prompt.prompt_id}",
                prompt_id=prompt.prompt_id,
                raw_response="x",
                response_skeleton={"text": "x"},
            )
            worker.submit_miss(prompt, response)

        deadline = time.time() + 5
        while time.time() < deadline and store.count_capsules() == 0:
            time.sleep(0.05)
        assert store.count_capsules() >= 1
    finally:
        worker.stop()

