from __future__ import annotations

from typing import Dict

import requests

from pool_guard.alarms.base import AlarmEvent


class HTTPAlarmSink:
    def __init__(self, url: str, timeout_s: float = 2.0, headers: Dict[str, str] | None = None) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.headers = headers or {}

    def trigger(self, event: AlarmEvent) -> None:
        if not self.url:
            return
        try:
            requests.post(self.url, json={"event": event.__dict__}, timeout=self.timeout_s, headers=self.headers)
        except Exception:
            # swallow to keep realtime loop alive
            return
