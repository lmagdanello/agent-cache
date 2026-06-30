from __future__ import annotations

from dataclasses import dataclass

from .models import CacheCapsule, PromptRecord


@dataclass(slots=True)
class DynamicDataPolicy:
    require_refresh_for_dynamic_fields: bool = True

    def allows(self, prompt: PromptRecord, capsule: CacheCapsule) -> bool:
        return not self.require_refresh_for_dynamic_fields or not capsule.allowed_dynamic_fields

