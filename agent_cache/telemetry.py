from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class TelemetryCounters:
    hits: int = 0
    misses: int = 0
    candidate_accepts: int = 0
    candidate_rejects: int = 0
    expired: int = 0
    validation_failed: int = 0
    estimated_tokens_saved: int = 0
    estimated_cost_saved: float = 0.0
    capsule_hits: int = 0
    capsule_rejections: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record(self, name: str, value: int | float = 1) -> None:
        with self._lock:
            current = getattr(self, name)
            setattr(self, name, current + value)

    def to_dict(self) -> dict[str, int | float]:
        with self._lock:
            total_decisions = self.hits + self.misses + self.candidate_rejects + self.validation_failed + self.expired
            total_rejections = self.candidate_rejects + self.validation_failed + self.expired + self.capsule_rejections
            return {
                "hits": self.hits,
                "misses": self.misses,
                "candidate_accepts": self.candidate_accepts,
                "candidate_rejects": self.candidate_rejects,
                "expired": self.expired,
                "validation_failed": self.validation_failed,
                "estimated_tokens_saved": self.estimated_tokens_saved,
                "estimated_cost_saved": self.estimated_cost_saved,
                "capsule_hits": self.capsule_hits,
                "capsule_rejections": self.capsule_rejections,
                "hit_rate": (self.hits / total_decisions) if total_decisions else 0.0,
                "rejection_rate": (total_rejections / total_decisions) if total_decisions else 0.0,
            }

    @classmethod
    def from_dict(cls, data: dict[str, int | float] | None) -> "TelemetryCounters":
        data = data or {}
        return cls(
            hits=int(data.get("hits", 0)),
            misses=int(data.get("misses", 0)),
            candidate_accepts=int(data.get("candidate_accepts", 0)),
            candidate_rejects=int(data.get("candidate_rejects", 0)),
            expired=int(data.get("expired", 0)),
            validation_failed=int(data.get("validation_failed", 0)),
            estimated_tokens_saved=int(data.get("estimated_tokens_saved", 0)),
            estimated_cost_saved=float(data.get("estimated_cost_saved", 0.0)),
            capsule_hits=int(data.get("capsule_hits", 0)),
            capsule_rejections=int(data.get("capsule_rejections", 0)),
        )

    def to_prometheus(self) -> str:
        data = self.to_dict()
        lines = [f"agent_cache_{k} {v}" for k, v in data.items()]
        return "\n".join(lines) + "\n"
