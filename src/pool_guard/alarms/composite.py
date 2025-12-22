from __future__ import annotations

from typing import List

from pool_guard.alarms.base import AlarmEvent, AlarmSink
from pool_guard.alarms.rate_limiter import RateLimiter


class CompositeAlarm:
    def __init__(self, sinks: List[AlarmSink], limiter: RateLimiter) -> None:
        self.sinks = sinks
        self.limiter = limiter

    def trigger(self, event: AlarmEvent) -> bool:
        key = f"{event.source}:{event.zone_id}"
        if not self.limiter.allow(key, event.ts_s):
            return False
        for s in self.sinks:
            try:
                s.trigger(event)
            except Exception:
                continue
        return True
