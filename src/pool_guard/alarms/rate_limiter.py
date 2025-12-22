from __future__ import annotations

from dataclasses import dataclass

from pool_guard.utils.time import now_s


@dataclass
class RateLimiter:
    cooldown_s: float
    _last: dict[str, float]

    def __init__(self, cooldown_s: float) -> None:
        self.cooldown_s = max(0.0, cooldown_s)
        self._last = {}

    def allow(self, key: str, ts_s: float | None = None) -> bool:
        if self.cooldown_s <= 0:
            return True
        ts = now_s() if ts_s is None else ts_s
        last = self._last.get(key, None)
        if last is None or (ts - last) >= self.cooldown_s:
            self._last[key] = ts
            return True
        return False
