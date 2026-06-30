from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str = "fake"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    def complete(self, prompt: str, *, system_prompt: str | None = None) -> LLMResponse: ...


class FakeLLMClient:
    def complete(self, prompt: str, *, system_prompt: str | None = None) -> LLMResponse:
        return LLMResponse(text=f"FAKE_RESPONSE: {prompt}", model="fake", prompt_tokens=len(prompt.split()))


class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> LLMResponse:
        raise NotImplementedError("OpenAI-compatible client stub is not configured in this build")

