from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .buffers import CandidateBuffer, CapsuleBuildBuffer, MissBuffer
from .capsules import build_capsule_from_examples
from .models import PromptRecord, ResponseRecord, utc_now
from .store import SQLiteCacheStore


@dataclass(slots=True)
class MissEvent:
    prompt: PromptRecord
    response: ResponseRecord


@dataclass(slots=True)
class CandidateEvent:
    prompt: PromptRecord
    reason: str
    score: float


@dataclass(slots=True)
class CapsuleBuildEvent:
    capsule_name: str
    prompt_examples: list[PromptRecord]


class BackgroundWorkerSystem:
    def __init__(
        self,
        *,
        store: SQLiteCacheStore,
        capsule_builder: Callable[[str, str, list[PromptRecord]], object] | None = None,
        build_threshold: int = 3,
    ):
        self.store = store
        self.capsule_builder = capsule_builder or (lambda capsule_id, name, prompts: build_capsule_from_examples(capsule_id, name, prompts))
        self.build_threshold = build_threshold
        self.miss_buffer: MissBuffer[MissEvent] = MissBuffer(256)
        self.candidate_buffer: CandidateBuffer[CandidateEvent] = CandidateBuffer(256)
        self.capsule_build_buffer: CapsuleBuildBuffer[CapsuleBuildEvent] = CapsuleBuildBuffer(64)
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="agent-cache-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        self._queue.put(("stop", None))
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit_miss(self, prompt: PromptRecord, response: ResponseRecord) -> None:
        self._queue.put(("miss", MissEvent(prompt=prompt, response=response)))

    def submit_candidate(self, prompt: PromptRecord, reason: str, score: float) -> None:
        self._queue.put(("candidate", CandidateEvent(prompt=prompt, reason=reason, score=score)))

    def submit_capsule_examples(self, capsule_name: str, prompt_examples: list[PromptRecord]) -> None:
        self._queue.put(("capsule", CapsuleBuildEvent(capsule_name=capsule_name, prompt_examples=prompt_examples)))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                kind, payload = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if kind == "stop":
                break
            if kind == "miss":
                self._handle_miss(payload)  # type: ignore[arg-type]
            elif kind == "candidate":
                self._handle_candidate(payload)  # type: ignore[arg-type]
            elif kind == "capsule":
                self._handle_capsule(payload)  # type: ignore[arg-type]
            self._queue.task_done()

    def _handle_miss(self, event: MissEvent) -> None:
        self.miss_buffer.append(event)
        if len(self.miss_buffer.snapshot()) >= self.build_threshold:
            prompt_examples = [item.prompt for item in self.miss_buffer.snapshot()[-self.build_threshold :]]
            self.capsule_build_buffer.append(CapsuleBuildEvent(capsule_name="auto-built", prompt_examples=prompt_examples))
            self._build_capsule(prompt_examples)

    def _handle_candidate(self, event: CandidateEvent) -> None:
        self.candidate_buffer.append(event)

    def _handle_capsule(self, event: CapsuleBuildEvent) -> None:
        self.capsule_build_buffer.append(event)
        self._build_capsule(event.prompt_examples, capsule_name=event.capsule_name)

    def _build_capsule(self, prompt_examples: list[PromptRecord], capsule_name: str = "auto-built") -> None:
        capsule_id = f"capsule-{len(self.store.iter_capsules()) + 1}"
        capsule = self.capsule_builder(capsule_id, capsule_name, prompt_examples)
        self.store.save_capsule(capsule)

