from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class AlarmEvent:
    ts_s: float
    source: str
    zone_id: str
    payload: Dict[str, Any]


class AlarmSink(Protocol):
    def trigger(self, event: AlarmEvent) -> None:
        ...
