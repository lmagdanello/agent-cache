from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class BoundedBuffer(Generic[T]):
    maxlen: int
    _items: Deque[T] = None  # type: ignore[assignment]
    _lock: Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._items = deque(maxlen=self.maxlen)
        self._lock = Lock()

    def append(self, item: T) -> None:
        with self._lock:
            self._items.append(item)

    def snapshot(self) -> list[T]:
        with self._lock:
            return list(self._items)


class MissBuffer(BoundedBuffer[T]):
    pass


class CandidateBuffer(BoundedBuffer[T]):
    pass


class CapsuleBuildBuffer(BoundedBuffer[T]):
    pass
